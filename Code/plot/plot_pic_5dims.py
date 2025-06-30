import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# 设置Times New Roman字体（期刊标准）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# 全局图表设置
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 300

# 读取Excel数据
df = pd.read_excel('ct_subskills_data.xlsx')

# 定义5个子维度
subskills = [
    'Problem Decomposition',
    'Pattern Recognition',
    'Abstraction',
    'Algorithm Design',
    'Algorithm Evaluation'
]

# 每个子维度对应的列名（按顺序：实验组前测、后测；对照组前测、后测）
col_groups = {
    'Problem Decomposition': ['prob_decomp_exp_pre', 'prob_decomp_exp_post', 'prob_decomp_ctrl_pre',
                              'prob_decomp_ctrl_post'],
    'Pattern Recognition': ['pattern_recog_exp_pre', 'pattern_recog_exp_post', 'pattern_recog_ctrl_pre',
                            'pattern_recog_ctrl_post'],
    'Abstraction': ['abstraction_exp_pre', 'abstraction_exp_post', 'abstraction_ctrl_pre', 'abstraction_ctrl_post'],
    'Algorithm Design': ['algo_design_exp_pre', 'algo_design_exp_post', 'algo_design_ctrl_pre',
                         'algo_design_ctrl_post'],
    'Algorithm Evaluation': ['algo_eval_exp_pre', 'algo_eval_exp_post', 'algo_eval_ctrl_pre', 'algo_eval_ctrl_post']
}

# 创建图表和子图
fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
# fig.suptitle('Computational Thinking Subskills: Pre-test vs. Post-test Distributions', fontsize=14, fontweight='bold', y=1.02, fontname='Times New Roman')

# 定义颜色和线型
colors = ['magenta', 'b', '#2ca02c', '#d62728']  # 专业学术配色
line_styles = ['-', '--', '-.', ':']  # 不同线型
labels = ['Treatment Pre', 'Treatment Post', 'Control Pre', 'Control Post']

# 遍历每个子维度
for idx, skill in enumerate(subskills):
    ax = axes[idx]
    cols = col_groups[skill]

    # 绘制当前子维度的4条曲线
    for i, col in enumerate(cols):
        data = df[col].dropna()
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 500)
        y = kde(x)
        ax.plot(x, y, color=colors[i], linestyle=line_styles[i], linewidth=1.8, label=labels[i])

        # 为实验组后测和对照组后测添加峰值垂直线
        if i == 1 or i == 3:  # 实验组后测(i=1)和对照组后测(i=3)
            # 找到峰值位置
            peaks, _ = find_peaks(y, height=0.1)
            if len(peaks) > 0:
                peak_idx = peaks[np.argmax(y[peaks])]  # 找到最高峰值
                peak_x = x[peak_idx]
                peak_y = y[peak_idx]
                # 绘制垂直线
                ax.axvline(x=peak_x, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.2)
                # 添加峰值标记
                ax.plot(peak_x, peak_y, marker='o', markersize=5, color=colors[i],
                        markeredgecolor='w', markeredgewidth=0.8)

    # 设置子图标题和标签
    ax.set_title(skill, fontweight='bold', fontname='Times New Roman', pad=10)
    ax.set_xlabel('Score', fontname='Times New Roman')
    if idx == 0:  # 仅最左侧子图显示y轴标签
        ax.set_ylabel('Proportion', fontname='Times New Roman')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 添加子图图例（仅第一个子图）
    # if idx == 0:
    #     ax.legend(loc='lower center', frameon=True, shadow=True,
    #               prop={'family': 'Times New Roman', 'size': 13})
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.05), frameon=True, shadow=True,
           prop={'family': 'Times New Roman', 'size': 13})
# 调整子图间距
plt.subplots_adjust(wspace=0.25, bottom=0.15)

# 添加全局标题说明
# fig.text(0.5, 0.05, 'Vertical dashed lines indicate peak values for post-test groups', ha='center', fontsize=10, fontname='Times New Roman', style='italic')

# 保存图像
# plt.savefig('ct_subskills_distributions.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('ct_subskills_distributions9.png', dpi=300, bbox_inches='tight')
# plt.show()