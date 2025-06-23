import re
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def parse_psnr_data(file_path):
    """解析日志文件中的PSNR数据"""
    psnr_data = []  # (iteration, psnr_value)

    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if 'Validation Vid4' in lines[i]:
                # 查找下一行中的PSNR值
                if i+1 < len(lines):
                    match_psnr = re.search(r'# psnr:\s*(\d+\.\d+)', lines[i+1])
                    match_iter = None

                    # 在当前行或之后的"Best"行中查找迭代次数
                    for j in range(i, min(i+3, len(lines))):
                        if 'Best:' in lines[j]:
                            match_iter = re.search(r'Best:.*@ (\d+) iter', lines[j])
                            if match_iter:
                                break

                    if match_psnr and match_iter:
                        iteration = int(match_iter.group(1))
                        psnr = float(match_psnr.group(1))
                        psnr_data.append((iteration, psnr))
            i += 1

    return sorted(psnr_data, key=lambda x: x[0])

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

    x_smooth = np.linspace(min(x), max(x), 500)
    spl = make_interp_spline(x, y, k=3)  # 使用三次样条插值
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
plt.title('PSNR Comparison: CVSR vs KVSR (Vid4 Benchmark)',
          fontsize=16, pad=15)
plt.xlabel('Training Iteration', fontsize=14)
plt.ylabel('PSNR (dB)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例并包含最终PSNR值
if cvsr_psnrs and kvsr_psnrs:
    cvsr_final = f'{cvsr_psnrs[-1]:.4f} dB'
    kvsr_final = f'{kvsr_psnrs[-1]:.4f} dB'
    legend_labels = [
        f'CVSR (final: {cvsr_final})',
        f'KVSR (final: {kvsr_final})'
    ]
else:
    legend_labels = ['CVSR', 'KVSR']

plt.legend(legend_labels, fontsize=13, loc='lower right')

# 设置坐标轴范围
plt.xlim(0, 305000)
if cvsr_psnrs and kvsr_psnrs:
    min_psnr = min(min(cvsr_psnrs), min(kvsr_psnrs)) - 0.2
    max_psnr = max(max(cvsr_psnrs), max(kvsr_psnrs)) + 0.2
    plt.ylim(min_psnr, max_psnr)

# 添加精细的刻度
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(50000))
ax.xaxis.set_minor_locator(MultipleLocator(10000))

# 添加精细的Y轴刻度（0.1 dB间隔）
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# 美化布局
plt.tight_layout()

# 保存为高分辨率图片
plt.savefig('psnr_comparison_smooth.png', dpi=300, bbox_inches='tight')
print('PSNR比较图表已保存为 "psnr_comparison_smooth.png"')