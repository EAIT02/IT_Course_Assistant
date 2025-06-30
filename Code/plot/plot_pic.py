import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

'''
我的想法是学习参与度、计算思维能力、学习成绩放一个图里，3个子图
然后计算思维的5个子维度再放一个图，5个子图
'''



# 设置Times New Roman字体（期刊标准）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

# 全局图表设置
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 300
# mpl.rcParams['figure.figsize'] = (15, 8)

# 读取Excel数据
df = pd.read_excel('experiment_data.xlsx')

# 创建图表和子图
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
# fig.suptitle('Distribution of Scores: Pre-test vs. Post-test', fontsize=14, fontweight='bold', y=1.02, fontname='Times New Roman')

# 定义颜色和线型
colors = ['magenta', 'b', '#2ca02c', '#d62728']  # 专业学术配色
line_styles = ['-', '--', '-.', ':']  # 不同线型
labels = ['Treatment Pre', 'Treatment Post', 'Control Pre', 'Control Post']

# 1. 学习参与度子图
for i, col in enumerate(['learning_engagement_exp_pre', 'learning_engagement_exp_post',
                         'learning_engagement_ctrl_pre', 'learning_engagement_ctrl_post']):
    data = df[col].dropna()
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 500)

    y = kde(x)
    axes[0].plot(x, y, color=colors[i], linestyle=line_styles[i], linewidth=2, label=labels[i])

    # 为实验组后测和对照组后测添加峰值垂直线
    if i == 1 or i == 3:  # 实验组后测(i=1)和对照组后测(i=3)
        # 找到峰值位置
        peaks, _ = find_peaks(y, height=0.03)
        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(y[peaks])]  # 找到最高峰值
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            # 绘制垂直线
            axes[0].axvline(x=peak_x, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
            # 添加峰值标记
            axes[0].plot(peak_x, peak_y, marker='o', markersize=6, color=colors[i],
                         markeredgecolor='w', markeredgewidth=1)

axes[0].set_title('Learning Engagement', fontweight='bold', fontname='Times New Roman')
axes[0].set_xlabel('Score', fontname='Times New Roman')
axes[0].set_ylabel('Proportion', fontname='Times New Roman')
axes[0].grid(True, linestyle='--', alpha=0.7)

# 2. 计算思维能力子图
for i, col in enumerate(['ct_skills_exp_pre', 'ct_skills_exp_post',
                         'ct_skills_ctrl_pre', 'ct_skills_ctrl_post']):
    data = df[col].dropna()
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = kde(x)
    axes[1].plot(x, y, color=colors[i], linestyle=line_styles[i], linewidth=2, label=labels[i])

    # 为实验组后测和对照组后测添加峰值垂直线
    if i == 1 or i == 3:  # 实验组后测(i=1)和对照组后测(i=3)
        # 找到峰值位置
        peaks, _ = find_peaks(y, height=0.02)
        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(y[peaks])]  # 找到最高峰值
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            # 绘制垂直线
            axes[1].axvline(x=peak_x, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
            # 添加峰值标记
            axes[1].plot(peak_x, peak_y, marker='o', markersize=6, color=colors[i],
                         markeredgecolor='w', markeredgewidth=1)

axes[1].set_title('Computational Thinking Skills', fontweight='bold', fontname='Times New Roman')
axes[1].set_xlabel('Score', fontname='Times New Roman')
# axes[1].set_ylabel('Proportion', fontname='Times New Roman')
axes[1].grid(True, linestyle='--', alpha=0.7)

# 3. 学业成绩子图
for i, col in enumerate(['academic_perf_exp_pre', 'academic_perf_exp_post',
                         'academic_perf_ctrl_pre', 'academic_perf_ctrl_post']):
    data = df[col].dropna()
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = kde(x)
    axes[2].plot(x, y, color=colors[i], linestyle=line_styles[i], linewidth=2, label=labels[i])

    # 为实验组后测和对照组后测添加峰值垂直线
    if i == 1 or i == 3:  # 实验组后测(i=1)和对照组后测(i=3)
        # 找到峰值位置
        peaks, _ = find_peaks(y, height=0.03)
        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(y[peaks])]  # 找到最高峰值
            peak_x = x[peak_idx]
            peak_y = y[peak_idx]
            # 绘制垂直线
            axes[2].axvline(x=peak_x, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
            # 添加峰值标记
            axes[2].plot(peak_x, peak_y, marker='o', markersize=6, color=colors[i],
                         markeredgecolor='w', markeredgewidth=1)

axes[2].set_title('Academic Achievement', fontweight='bold', fontname='Times New Roman')
axes[2].set_xlabel('Score', fontname='Times New Roman')
# axes[2].set_ylabel('Proportion', fontname='Times New Roman')
axes[2].grid(True, linestyle='--', alpha=0.7)

# 添加图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.05), frameon=True, shadow=True,
           prop={'family': 'Times New Roman', 'size': 13})

# 调整布局并保存
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # 为底部图例留出空间
# plt.savefig('score_distributions.pdf', dpi=300, bbox_inches='tight', format='pdf')  # 保存为PDF确保字体嵌入
plt.savefig('score_distributions4.png', dpi=300, bbox_inches='tight')
# plt.show()
