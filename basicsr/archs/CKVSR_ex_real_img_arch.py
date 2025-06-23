import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet

@ARCH_REGISTRY.register()
class CKVSRexRI(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment 光流对齐模块 - 使用SpyNet计算帧间运动
        self.spynet = SpyNet(spynet_path)

        # propagation === 双向传播分支 ===
        # 后向传播分支 (处理逆序帧)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        # 前向传播分支 (处理正序帧)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # === 投影头分离特征空间 ===
        # 用于将前向特征投影到新的空间
        self.forward_projection = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlockNoBN(num_feat) # 增加一些非线性变换和表达能力
        )
        # 用于将后向特征投影到新的空间
        self.backward_projection = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlockNoBN(num_feat) # 增加一些非线性变换和表达能力
        )

        # === 复数特征构建模块 ===
        # 1x1卷积压缩双向特征通道 (用于构建虚部)
        self.compress = nn.Conv2d(2*num_feat,num_feat,1,1,0)
        # 残差块处理压缩后的特征 (虚部生成)
        self.resblock1 = ResidualBlockNoBN(num_feat)

        # === 复数注意力机制 ===
        # 3D卷积+池化提取复数空间注意力
        self.conv3D = nn.Sequential(
            nn.Conv3d(num_feat,num_feat,(3,3,3),(1,1,1),(1,1,1)),  # 3D空间卷积
            nn.PReLU(), # 参数化ReLU
            nn.Conv3d(num_feat, num_feat, (2, 1, 1), (2, 1, 1), (0,0,0)), # 降维卷积
            nn.AdaptiveAvgPool3d(1) # 全局池化生成注意力向量
        )
        # 实部增强卷积
        self.convreal = nn.Conv2d(num_feat,num_feat,3,1,1,bias=True)
        # 虚部增强卷积
        self.convimg = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # === 重建模块 ===
        # 特征融合 (复数特征→实数空间)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        # 第一次上采样卷积
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        # 第二次上采样卷积
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        # 高分辨率特征精炼
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        # 最终输出层
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # 像素重排上采样 (×2)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def get_flow(self, x):
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, x):
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # === 应用投影头分离特征空间 ===
            # 将前向特征投影到其独立空间
            projected_fwd_feat = self.forward_projection(feat_prop)
            # 将后向特征（从out_l中获取）投影到其独立空间
            projected_bwd_feat = self.backward_projection(out_l[i])

            # ======================= [MODIFIED START] =======================
            # === 复数特征构建 (实验性修改: 后向为实，前向为虚) ===
            # 实部 = 后向特征 - 前向特征
            real = projected_bwd_feat - projected_fwd_feat
            # 虚部 = 融合(后向+前向) → 压缩 → 残差块
            img = self.resblock1(self.compress(torch.cat([projected_bwd_feat, projected_fwd_feat], dim=1)))
            # ======================= [MODIFIED END] =======================

            # === 复数注意力机制 ===
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal,simg],dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img *attsin
            attreal = real+self.convreal(real)
            attimg = img+self.convimg(img)
            out = torch.cat([attreal, attimg], dim=1)

            # === 超分辨率重建 ===
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out
            lastoutput = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

@ARCH_REGISTRY.register()
class CKIconVSRexRI(nn.Module):
    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        self.spynet = SpyNet(spynet_path)

        # === 传播分支 ===
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        # === 复数域处理核心 C.T ===
        self.compress = nn.Conv2d(2*num_feat, num_feat, 1, 1, 0)
        self.resblock1 = ResidualBlockNoBN(num_feat)

        # 3D复数注意力机制 C.A
        self.conv3D = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(num_feat, num_feat, (2, 1, 1), (2, 1, 1), (0, 0, 0)),
            nn.AdaptiveAvgPool3d(1)
        )
        self.convreal = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.convimg = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # ======================= [MODIFIED START] =======================
        # === 重建模块 ===
        # [新增] 增加fusion层以处理复数模块的输出
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        # ======================= [MODIFIED END] =======================
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def pad_spatial(self, x):
        n, t, c, h, w = x.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)
        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()
        x = self.pad_spatial(x)
        h, w = x.shape[3:]
        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # ======================= [MODIFIED START] =======================
            # === 复数域处理 C.T (实验性修改: 后向为实，前向为虚) ===
            # 实部 = 后向特征 - 前向特征
            real = out_l[i] - feat_prop
            # 虚部 = 融合(后向+前向) → 压缩 → 残差块
            img = self.resblock1(self.compress(torch.cat([out_l[i], feat_prop], dim=1)))
            # ======================= [MODIFIED END] =======================

            # === C.A ===
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal,simg],dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img *attsin
            attreal = real+self.convreal(real)
            attimg = img+self.convimg(img)
            out = torch.cat([attreal, attimg], dim=1)

            # ======================= [MODIFIED START] =======================
            # === 上采样 (修正BUG) ===
            # [修正] 应用特征融合层并将融合后的特征送入上采样模块
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            # ======================= [MODIFIED END] =======================

            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):
    def __init__(self, num_input_frame, num_feat, load_path):
        super(EDVRFeatureExtractor, self).__init__()
        self.center_frame_idx = num_input_frame // 2
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)
        ref_feat_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        return self.fusion(aligned_feat)