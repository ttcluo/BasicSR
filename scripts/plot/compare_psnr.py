import re
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def parse_psnr_data(file_path):
    """解析日志文件中的PSNR数据"""
    psnr_data = []  # (iteration, psnr_value)

    last_iter = None  # 用于存储最后的迭代次数

    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # 提取训练日志行中的迭代次数 (正确行)
            if 'iter:' in line and 'l_pix:' in line and 'eta:' in line:
                match_iter = re.search(r'iter:\s*(\d+,\d+)', line)
                if match_iter:
                    # 去掉逗号并转换为整数
                    iter_num = int(match_iter.group(1).replace(',', ''))
                    last_iter = iter_num

            # 当遇到"Saving models"时，使用最后的迭代次数作为验证点的迭代次数
            if 'Saving models and training states.' in line and last_iter is not None:
                # 寻找接下来的Validation行
                for j in range(i+1, min(i+10, len(lines))):
                    if 'Validation Vid4' in lines[j]:
                        # 在验证行后面找PSNR值
                        if j+1 < len(lines):
                            match_psnr = re.search(r'# psnr:\s*(\d+\.\d+)', lines[j+1])
                            if match_psnr:
                                psnr = float(match_psnr.group(1))
                                psnr_data.append((last_iter, psnr))
                        break
            i += 1

    # 去掉可能的重复项 (基于迭代次数)
    unique_data = {}
    for iter, psnr in psnr_data:
        unique_data[iter] = psnr

    return sorted([(k, unique_data[k]) for k in unique_data], key=lambda x: x[0])

# 解析两个方法的日志文件
cvsr_data = parse_psnr_data('/data/luochuan/BasicSR/experiments/CVSR_Vimeo90K_BIx4_ex_real_img/train_CVSR_Vimeo90K_BIx4_ex_real_img_20250619_224309.log')
kvsr_data = parse_psnr_data('/data/luochuan/BasicSR/experiments/KVSR_Vimeo90K_BIx4/train_KVSR_Vimeo90K_BIx4_20250621_000459.log')

# 提取数据
cvsr_iters, cvsr_psnrs = zip(*cvsr_data) if cvsr_data else ([], [])
kvsr_iters, kvsr_psnrs = zip(*kvsr_data) if kvsr_data else ([], [])

# 创建平滑曲线数据
def create_smooth_curve(x, y):
    """使用样条插值创建平滑曲线"""
    if len(x) < 4:  # 需要足够多的点才能插值
        return x, y

    # 确保x值严格递增 (使用唯一值)
    x_arr = np.array(x)
    y_arr = np.array(y)

    # 移除可能重复的点
    unique_indices = np.unique(x_arr, return_index=True)[1]
    x_unique = [x_arr[i] for i in sorted(unique_indices)]
    y_unique = [y_arr[i] for i in sorted(unique_indices)]

    if len(x_unique) < 4:
        return x_unique, y_unique

    # 创建平滑曲线
    x_smooth = np.linspace(min(x_unique), max(x_unique), 500)
    spl = make_interp_spline(x_unique, y_unique, k=3)  # 使用三次样条插值
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

# 为CVSR创建平滑数据
if cvsr_psnrs:
    cvsr_x_smooth, cvsr_y_smooth = create_smooth_curve(cvsr_iters, cvsr_psnrs)
else:
    cvsr_x_smooth, cvsr_y_smooth = [], []

# 为KVSR创建平滑数据
if kvsr_psnrs:
    kvsr_x_smooth, kvsr_y_smooth = create_smooth_curve(kvsr_iters, kvsr_psnrs)
else:
    kvsr_x_smooth, kvsr_y_smooth = [], []

# 创建比较图表
plt.figure(figsize=(12, 8))
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans',
    'axes.labelpad': 12
})

# 绘制平滑PSNR曲线
plt.plot(cvsr_x_smooth, cvsr_y_smooth, '-', linewidth=3,
         color='#1f77b4', label='CVSR')

plt.plot(kvsr_x_smooth, kvsr_y_smooth, '-', linewidth=3,
         color='#ff7f0e', label='KVSR')

# 添加原始数据点
if cvsr_psnrs:
    plt.scatter(cvsr_iters, cvsr_psnrs, s=30,
               color='#1f77b4', alpha=0.6)

if kvsr_psnrs:
    plt.scatter(kvsr_iters, kvsr_psnrs, s=30,
               color='#ff7f0e', alpha=0.6)

# 设置图表属性
plt.title('PSNR Comparison: CVSR vs KVSR',
          fontsize=16, pad=15)
plt.xlabel('Training Iteration', fontsize=14)
plt.ylabel('PSNR (dB)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例并包含最终PSNR值
cvsr_final = cvsr_psnrs[-1] if cvsr_psnrs else None
kvsr_final = kvsr_psnrs[-1] if kvsr_psnrs else None

legend_labels = [
    f'CVSR (final: {cvsr_final:.4f} dB)' if cvsr_final is not None else 'CVSR',
    f'KVSR (final: {kvsr_final:.4f} dB)' if kvsr_final is not None else 'KVSR'
]

plt.legend(legend_labels, fontsize=13, loc='lower right')

# 设置坐标轴范围
plt.xlim(0, 305000)
if cvsr_psnrs and kvsr_psnrs:
    min_psnr = min(min(cvsr_psnrs), min(kvsr_psnrs)) - 0.1
    max_psnr = max(max(cvsr_psnrs), max(kvsr_psnrs)) + 0.1
    plt.ylim(min_psnr, max_psnr)
elif cvsr_psnrs:
    plt.ylim(min(cvsr_psnrs)-0.1, max(cvsr_psnrs)+0.1)
elif kvsr_psnrs:
    plt.ylim(min(kvsr_psnrs)-0.1, max(kvsr_psnrs)+0.1)

# 添加精细的刻度
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(50000))
ax.xaxis.set_minor_locator(MultipleLocator(10000))

# Y轴刻度设置为更精细
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

# 美化布局
plt.tight_layout()

# 保存为高分辨率图片
plt.savefig('psnr_comparison_smooth.png', dpi=300, bbox_inches='tight')
print('PSNR比较图表已保存为 "psnr_comparison_smooth.png"')