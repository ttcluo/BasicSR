import re
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

# 提取数据用于绘图
cvsr_iters, cvsr_psnrs = zip(*cvsr_data) if cvsr_data else ([], [])
kvsr_iters, kvsr_psnrs = zip(*kvsr_data) if kvsr_data else ([], [])

# 创建比较图表
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

# 绘制PSNR曲线
plt.plot(cvsr_iters, cvsr_psnrs, 's-', linewidth=2.5, markersize=8,
         color='#1f77b4', markerfacecolor='white', markeredgewidth=2,
         label='CVSR')

plt.plot(kvsr_iters, kvsr_psnrs, 'o-', linewidth=2.5, markersize=8,
         color='#ff7f0e', markerfacecolor='white', markeredgewidth=2,
         label='KVSR')

# 设置图表属性
plt.title('PSNR Comparison: CVSR vs KVSR (Vid4 Benchmark)', fontsize=16, pad=20)
plt.xlabel('Training Iteration', fontsize=14, labelpad=10)
plt.ylabel('PSNR (dB)', fontsize=14, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=13, loc='lower right')

# 设置坐标轴范围
plt.xlim(0, 305000)
if cvsr_psnrs and kvsr_psnrs:
    min_psnr = min(min(cvsr_psnrs), min(kvsr_psnrs)) - 0.5
    max_psnr = max(max(cvsr_psnrs), max(kvsr_psnrs)) + 0.5
    plt.ylim(min_psnr, max_psnr)

# 添加主要和次要刻度
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(50000))
ax.xaxis.set_minor_locator(MultipleLocator(10000))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

# 突出最终结果
if cvsr_psnrs:
    plt.scatter([cvsr_iters[-1]], [cvsr_psnrs[-1]], s=120, zorder=10,
                facecolor='white', edgecolor='#1f77b4', linewidth=2.5)
    plt.annotate(f'CVSR Final: {cvsr_psnrs[-1]:.4f} dB',
                (cvsr_iters[-1], cvsr_psnrs[-1]),
                textcoords="offset points", xytext=(-10,-15),
                ha='right', fontsize=11, arrowprops=dict(arrowstyle="->", color='#1f77b4'))

if kvsr_psnrs:
    plt.scatter([kvsr_iters[-1]], [kvsr_psnrs[-1]], s=120, zorder=10,
                facecolor='white', edgecolor='#ff7f0e', linewidth=2.5)
    plt.annotate(f'KVSR Final: {kvsr_psnrs[-1]:.4f} dB',
                (kvsr_iters[-1], kvsr_psnrs[-1]),
                textcoords="offset points", xytext=(10,-15),
                ha='left', fontsize=11, arrowprops=dict(arrowstyle="->", color='#ff7f0e'))

# 美化布局
plt.tight_layout()

# 保存为高分辨率图片
plt.savefig('psnr_comparison.png', dpi=300, bbox_inches='tight')
print('PSNR比较图表已保存为 "psnr_comparison.png"')