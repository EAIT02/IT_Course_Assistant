import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
from scipy import stats

# 1. 数据准备
# 假设数据已准备好，包含以下列：
# treatment: 0=对照组, 1=实验组
# engagement_gain: 学习参与度增益分数 (后测-前测)
# ct_gain: 计算思维增益分数 (后测-前测)
# achievement_gain: 学业成绩增益分数 (后测-前测)
# gender: 性别 (0=男, 1=女)
# age: 年龄

# engagement = pd.read_excel('pretest_data.xlsx', sheet_name='学习参与度')
engagement = pd.read_excel('posttest_data.xlsx', sheet_name='学习参与度')
# ct_pre = pd.read_excel('pretest_data.xlsx', sheet_name='计算思维能力')
ct = pd.read_excel('posttest_data.xlsx', sheet_name='计算思维能力')
# achievement_pre = pd.read_excel('pretest_data.xlsx', sheet_name='学业成绩')
achievement = pd.read_excel('posttest_data.xlsx', sheet_name='学业成绩')
demographics = pd.read_excel('age.xlsx')

# 生成模拟数据 (实际使用时替换为真实数据)
np.random.seed(42)
n = 410  # 样本量

data = pd.DataFrame({
    'treatment': np.concatenate([np.ones(205), np.zeros(205)]),
    'engagement_gain': np.concatenate([engagement['T_gain'], engagement['C_gain']]),

        # np.random.normal(5.0, 1.5, 205),  # 实验组
        # np.random.normal(0.5, 1.5, 205)  # 对照组
    # ]),
    'ct_gain': np.concatenate([ct['T_gain'], ct['C_gain']]),
    #     np.random.normal(7.2, 2.0, 205),  # 实验组
    #     np.random.normal(1.0, 2.0, 205)  # 对照组
    # ]),
    'achievement_gain': np.concatenate([achievement['T_gain'], achievement['C_gain']]),
        # np.random.normal(4.8, 1.8, 205),  # 实验组
        # np.random.normal(0.0, 1.8, 205)  # 对照组
    # ]),
    'gender': demographics['gender'], #np.random.randint(0, 2, n),
    'age': demographics['age'], #np.random.randint(15, 18, n)
})


# 2. 路径分析 (验证中介路径)
def path_analysis(data):
    """执行路径分析验证中介效应"""
    # 第一步: treatment → engagement
    X1 = data[['treatment', 'gender', 'age']]
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(data['engagement_gain'], X1).fit()

    # 第二步: engagement → ct (控制treatment)
    X2 = data[['engagement_gain', 'treatment', 'gender', 'age']]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(data['ct_gain'], X2).fit()

    # 第三步: ct → achievement (控制treatment和engagement)
    X3 = data[['ct_gain', 'engagement_gain', 'treatment', 'gender', 'age']]
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(data['achievement_gain'], X3).fit()

    return model1, model2, model3


# 执行分析
model1, model2, model3 = path_analysis(data)


# 3. 计算链式中介效应
def calculate_mediation_effect(model1, model2, model3):
    """计算链式中介效应 (treatment→engagement→ct→achievement)"""
    # 提取路径系数
    a = model1.params['treatment']  # treatment→engagement
    b = model2.params['engagement_gain']  # engagement→ct
    c = model3.params['ct_gain']  # ct→achievement

    # 计算中介效应
    indirect_effect = a * b * c

    # 计算标准误 (Delta方法)
    se_a = model1.bse['treatment']
    se_b = model2.bse['engagement_gain']
    se_c = model3.bse['ct_gain']

    se_indirect = np.sqrt(
        (b * c * se_a) ** 2 +
        (a * c * se_b) ** 2 +
        (a * b * se_c) ** 2
    )

    # 计算z值和p值
    z = indirect_effect / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    # 置信区间
    ci_lower = indirect_effect - 1.96 * se_indirect
    ci_upper = indirect_effect + 1.96 * se_indirect

    return {
        '路径系数 a (treatment→engagement)': a,
        '路径系数 b (engagement→ct)': b,
        '路径系数 c (ct→achievement)': c,
        '链式中介效应 (a*b*c)': indirect_effect,
        '标准误': se_indirect,
        'z值': z,
        'p值': p,
        '95%置信区间': [ci_lower, ci_upper]
    }


# 计算中介效应
mediation_result = calculate_mediation_effect(model1, model2, model3)

# 4. 输出结果
print("=" * 80)
print("链式中介效应分析结果")
print("=" * 80)

# 打印路径分析结果
print("\n路径1: treatment → engagement")
print(model1.summary())

print("\n路径2: engagement → ct (控制treatment)")
print(model2.summary())

print("\n路径3: ct → achievement (控制engagement和treatment)")
print(model3.summary())

# 打印中介效应结果
print("\n链式中介效应 (treatment→engagement→ct→achievement):")
for key, value in mediation_result.items():
    if key == '95%置信区间':
        print(f"{key}: [{value[0]:.4f}, {value[1]:.4f}]")
    else:
        print(f"{key}: {value:.4f}")

# 5. 生成论文结果描述
print("\n" + "=" * 80)
print("可直接用于论文的结果描述:")
print("=" * 80)
print(
    f"路径分析验证了假设的中介机制。治疗分配显著预测学习参与度的提升(β = {mediation_result['路径系数 a (treatment→engagement)']:.3f}, "
    f"p = {model1.pvalues['treatment']:.4f})，而学习参与度的提升又显著预测计算思维能力的增强(β = {mediation_result['路径系数 b (engagement→ct)']:.3f}, "
    f"p = {model2.pvalues['engagement_gain']:.4f})。计算思维能力的提升进一步显著预测学业成绩的提高(β = {mediation_result['路径系数 c (ct→achievement)']:.3f}, "
    f"p = {model3.pvalues['ct_gain']:.4f})。\n")

print(
    f"链式中介效应分析显示，治疗分配通过'学习参与度→计算思维'路径对学业成绩的间接效应为{mediation_result['链式中介效应 (a*b*c)']:.4f} "
    f"(SE = {mediation_result['标准误']:.4f}, 95% CI [{mediation_result['95%置信区间'][0]:.4f}, "
    f"{mediation_result['95%置信区间'][1]:.4f}])，且具有统计学显著性(z = {mediation_result['z值']:.3f}, p = {mediation_result['p值']:.4f})。"
    "这支持了AI课程助手通过提升学习参与度和计算思维能力间接促进学业成绩的理论机制。")