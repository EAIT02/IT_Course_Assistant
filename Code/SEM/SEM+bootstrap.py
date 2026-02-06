import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SEM结构方程模型 + Bootstrap法中介路径分析（使用后测值）")
print("="*80)

# 1. 读取数据
print("\n正在读取数据...")
engagement_df = pd.read_excel(r'posttest_data.xlsx', sheet_name='学习参与度')
ct_df = pd.read_excel(r'posttest_data.xlsx', sheet_name='计算思维能力')
achievement_df = pd.read_excel(r'posttest_data.xlsx', sheet_name='学业成绩')
demographics_df = pd.read_excel(r'age.xlsx')

print(f"学习参与度数据: {len(engagement_df)} 行")
print(f"计算思维能力数据: {len(ct_df)} 行")
print(f"学业成绩数据: {len(achievement_df)} 行")
print(f"人口统计学数据: {len(demographics_df)} 行")

# 2. 数据预处理
n = len(engagement_df)
print(f"\n总样本量: {n}")

# 创建处理变量T (1=实验组, 0=对照组)
T = np.concatenate([np.ones(n), np.zeros(n)])

# 创建后测分数向量
# 对于学习参与度和计算思维能力，使用'实验组'和'对照组'列作为后测值
# 对于学业成绩，使用'Post_EE'和'Post_C'列作为后测值
E_post = np.concatenate([engagement_df['Post_EE'], engagement_df['Post_C']])
CT_post = np.concatenate([ct_df['Post_EE'], ct_df['Post_C']])
A_post = np.concatenate([achievement_df['Post_EE'], achievement_df['Post_C']])

# 创建前测分数向量
Pre_E = np.concatenate([engagement_df['Pre_EE'], engagement_df['Pre_C']])
Pre_CT = np.concatenate([ct_df['Pre_EE'], ct_df['Pre_C']])
Pre_A = np.concatenate([achievement_df['Pre_EE'], achievement_df['Pre_C']])

# 读取人口统计学数据（前205行是对照组，后205行是实验组）
# 需要重新排列：前n个是实验组，后n个是对照组
gender = np.concatenate([demographics_df['gender'].values[205:], demographics_df['gender'].values[:205]])
age = np.concatenate([demographics_df['age'].values[205:], demographics_df['age'].values[:205]])

# 创建完整的数据框
df = pd.DataFrame({
    'T': T,
    'E': E_post,
    'CT': CT_post,
    'A': A_post,
    'gender': gender,
    'age': age,
    'Pre_E': Pre_E,
    'Pre_CT': Pre_CT,
    'Pre_A': Pre_A
})

# 添加常数项
df['intercept'] = 1

print("数据准备完成")

# 3. Bootstrap中介效应分析函数
def bootstrap_mediation_analysis(df, outcome_variable, n_bootstraps=1000, random_seed=42):
    """
    使用Bootstrap方法检验中介效应 T -> E -> Outcome
    参数:
        df: 数据框
        outcome_variable: 结果变量名称 ('CT' 或 'A')
        n_bootstraps: Bootstrap抽样次数
        random_seed: 随机种子
    返回:
        分析结果字典
    """
    print(f"\n{'='*60}")
    print(f"中介路径分析: T -> E -> {outcome_variable}")
    print(f"{'='*60}")
    
    # 初始化存储Bootstrap结果的列表
    a_boot = []  # T -> E 的路径系数
    b_boot = []  # E -> Outcome 的路径系数
    c_prime_boot = []  # T -> Outcome (控制E) 的直接效应
    indirect_effects = []  # 间接效应 a * b
    
    print(f"开始Bootstrap抽样，共{n_bootstraps}次...")
    
    # Bootstrap循环
    for i in range(n_bootstraps):
        # 有放回抽样
        df_boot = resample(df, n_samples=len(df), random_state=i*random_seed)
        
        # 路径1: E = α0 + α1*T + α2*gender + α3*age + α4*Pre_E + ε
        # 解释：学习参与度受实验处理、性别、年龄和前测成绩的影响
        X_a = df_boot[['intercept', 'T', 'gender', 'age', 'Pre_E']]
        model_a = sm.OLS(df_boot['E'], X_a).fit()
        a = model_a.params['T']  # 提取T的系数（路径a）
        
        # 路径2: Outcome = β0 + β1*E + β2*T + β3*gender + β4*age + β5*Pre_Outcome + ξ
        # 解释：结果变量受学习参与度、实验处理、性别、年龄和前测成绩的影响
        # 注意：这里控制了T，以计算直接效应c'
        X_b = df_boot[['intercept', 'E', 'T', 'gender', 'age', f'Pre_{outcome_variable}']]
        model_b = sm.OLS(df_boot[outcome_variable], X_b).fit()
        b = model_b.params['E']  # 提取E的系数（路径b）
        c_prime = model_b.params['T']  # 提取T的系数（直接效应c'）
        
        # 保存Bootstrap结果
        a_boot.append(a)
        b_boot.append(b)
        c_prime_boot.append(c_prime)
        indirect_effects.append(a * b)  # 间接效应 = a * b
    
    print("Bootstrap抽样完成")
    
    # 转换为numpy数组
    a_boot = np.array(a_boot)
    b_boot = np.array(b_boot)
    c_prime_boot = np.array(c_prime_boot)
    indirect_effects = np.array(indirect_effects)
    
    # 计算原始模型的系数（不使用Bootstrap）
    # 路径1: E = α0 + α1*T + α2*gender + α3*age + α4*Pre_E + ε
    X_a_orig = df[['intercept', 'T', 'gender', 'age', 'Pre_E']]
    model_a_orig = sm.OLS(df['E'], X_a_orig).fit()
    
    # 路径2: Outcome = β0 + β1*E + β2*T + β3*gender + β4*age + β5*Pre_Outcome + ξ
    X_b_orig = df[['intercept', 'E', 'T', 'gender', 'age', f'Pre_{outcome_variable}']]
    model_b_orig = sm.OLS(df[outcome_variable], X_b_orig).fit()
    
    # 路径3: Outcome = γ0 + γ1*T + γ2*gender + γ3*age + γ4*Pre_Outcome + η
    # 解释：总效应模型，不控制中介变量E
    X_c_orig = df[['intercept', 'T', 'gender', 'age', f'Pre_{outcome_variable}']]
    model_c_orig = sm.OLS(df[outcome_variable], X_c_orig).fit()
    
    # 提取点估计值
    a_point = model_a_orig.params['T']  # 路径a: T -> E
    b_point = model_b_orig.params['E']  # 路径b: E -> Outcome
    c_prime_point = model_b_orig.params['T']  # 直接效应c': T -> Outcome (控制E)
    c_point = model_c_orig.params['T']  # 总效应c: T -> Outcome (不控制E)
    
    # 计算效应
    indirect_effect_point = a_point * b_point  # 间接效应 = a * b
    direct_effect_point = c_prime_point  # 直接效应 = c'
    total_effect_point = c_point  # 总效应 = c
    
    # 计算Bootstrap标准误
    indirect_effect_se = np.std(indirect_effects, ddof=1)
    
    # 计算百分位数置信区间（95% CI）
    ci_lower = np.percentile(indirect_effects, 2.5)
    ci_upper = np.percentile(indirect_effects, 97.5)
    
    # 计算z值和p值
    z_value = indirect_effect_point / indirect_effect_se
    p_value = 2 * (1 - norm.cdf(abs(z_value)))
    
    # 判断显著性：如果95% CI不包含0，则中介效应显著
    is_significant = ci_lower > 0 or ci_upper < 0
    
    # 输出结果
    print(f"\n路径系数:")
    print(f"  a (T -> E): {a_point:.4f} (SE={model_a_orig.bse['T']:.4f}, p={model_a_orig.pvalues['T']:.4f})")
    print(f"  b (E -> {outcome_variable}): {b_point:.4f} (SE={model_b_orig.bse['E']:.4f}, p={model_b_orig.pvalues['E']:.4f})")
    print(f"  c' (T -> {outcome_variable} | E): {c_prime_point:.4f} (SE={model_b_orig.bse['T']:.4f}, p={model_b_orig.pvalues['T']:.4f})")
    print(f"  c (T -> {outcome_variable}): {c_point:.4f} (SE={model_c_orig.bse['T']:.4f}, p={model_c_orig.pvalues['T']:.4f})")
    
    print(f"\n中介效应:")
    print(f"  间接效应 (a × b): {indirect_effect_point:.4f}")
    print(f"  直接效应 (c'): {direct_effect_point:.4f}")
    print(f"  总效应 (c): {total_effect_point:.4f}")
    
    print(f"\nBootstrap统计量:")
    print(f"  标准误: {indirect_effect_se:.4f}")
    print(f"  z值: {z_value:.4f}")
    print(f"  p值: {p_value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if is_significant:
        print(f"\n结论: 中介效应显著 (95% CI不包含0)")
    else:
        print(f"\n结论: 中介效应不显著 (95% CI包含0)")
    
    # 返回结果字典
    return {
        'outcome_variable': outcome_variable,
        'model_a': model_a_orig,
        'model_b': model_b_orig,
        'model_c': model_c_orig,
        'a_point': a_point,
        'b_point': b_point,
        'c_prime_point': c_prime_point,
        'c_point': c_point,
        'indirect_effect_point': indirect_effect_point,
        'direct_effect_point': direct_effect_point,
        'total_effect_point': total_effect_point,
        'indirect_effect_se': indirect_effect_se,
        'z_value': z_value,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': is_significant,
        'a_se': model_a_orig.bse['T'],
        'b_se': model_b_orig.bse['E'],
        'c_prime_se': model_b_orig.bse['T'],
        'c_se': model_c_orig.bse['T'],
        'a_p': model_a_orig.pvalues['T'],
        'b_p': model_b_orig.pvalues['E'],
        'c_prime_p': model_b_orig.pvalues['T'],
        'c_p': model_c_orig.pvalues['T']
    }

# 4. 执行两个中介路径分析
n_bootstraps = 1000

# 路径1: T -> E -> CT（实验处理通过学习参与度影响计算思维能力）
results_CT = bootstrap_mediation_analysis(df, 'CT', n_bootstraps)

# 路径2: T -> E -> A（实验处理通过学习参与度影响学业成绩）
results_A = bootstrap_mediation_analysis(df, 'A', n_bootstraps)

# 5. 比较两个中介路径
print(f"\n{'='*80}")
print("中介路径比较")
print(f"{'='*80}")

print(f"\n路径1 (T -> E -> CT):")
print(f"  间接效应: {results_CT['indirect_effect_point']:.4f}")
print(f"  95% CI: [{results_CT['ci_lower']:.4f}, {results_CT['ci_upper']:.4f}]")
print(f"  显著性: {'显著' if results_CT['is_significant'] else '不显著'}")

print(f"\n路径2 (T -> E -> A):")
print(f"  间接效应: {results_A['indirect_effect_point']:.4f}")
print(f"  95% CI: [{results_A['ci_lower']:.4f}, {results_A['ci_upper']:.4f}]")
print(f"  显著性: {'显著' if results_A['is_significant'] else '不显著'}")

# 6. 创建详细的结果表格
print(f"\n{'='*80}")
print("创建详细结果表格")
print(f"{'='*80}")

def create_detailed_results_sheet(results, outcome_variable):
    """
    创建详细的结果表格
    """
    model_a = results['model_a']
    model_b = results['model_b']
    model_c = results['model_c']
    
    # 定义回归方程
    equations = {
        'T->E': f"E = {model_a.params['intercept']:.4f} + {model_a.params['T']:.4f}*T + {model_a.params['gender']:.4f}*gender + {model_a.params['age']:.4f}*age + {model_a.params['Pre_E']:.4f}*Pre_E",
        f'T->{outcome_variable}': f"{outcome_variable} = {model_c.params['intercept']:.4f} + {model_c.params['T']:.4f}*T + {model_c.params['gender']:.4f}*gender + {model_c.params['age']:.4f}*age + {model_c.params[f'Pre_{outcome_variable}']:.4f}*Pre_{outcome_variable}",
        f'E->{outcome_variable}': f"{outcome_variable} = {model_b.params['intercept']:.4f} + {model_b.params['E']:.4f}*E + {model_b.params['T']:.4f}*T + {model_b.params['gender']:.4f}*gender + {model_b.params['age']:.4f}*age + {model_b.params[f'Pre_{outcome_variable}']:.4f}*Pre_{outcome_variable}"
    }
    
    # 创建表格
    data = []
    
    # 添加中介效应汇总
    data.append(['中介效应汇总', '', '', ''])
    data.append(['中介路径', f'T -> E -> {outcome_variable}', '', ''])
    data.append(['间接效应 (a×b)', f"{results['indirect_effect_point']:.4f}", f"({results['indirect_effect_se']:.4f})", f"{results['p_value']:.4f}"])
    data.append(['直接效应 (c\')', f"{results['c_prime_point']:.4f}", f"({results['c_prime_se']:.4f})", f"{results['c_prime_p']:.4f}"])
    data.append(['总效应 (c)', f"{results['c_point']:.4f}", f"({results['c_se']:.4f})", f"{results['c_p']:.4f}"])
    data.append(['95% CI', f"[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]", '', ''])
    data.append(['显著性判断', '显著' if results['is_significant'] else '不显著', '', ''])
    data.append(['', '', '', ''])
    
    # 添加T->E的回归结果
    data.append(['回归方程: T -> E', '', '', ''])
    data.append(['方程', equations['T->E'], '', ''])
    data.append(['R²', f"{model_a.rsquared:.4f}", '', ''])
    data.append(['调整R²', f"{model_a.rsquared_adj:.4f}", '', ''])
    data.append(['自变量', '系数(标准误)', 'p值', '显著性'])
    for var in ['intercept', 'T', 'gender', 'age', 'Pre_E']:
        var_name = {'intercept': '常数', 'T': 'T', 'gender': 'gender', 'age': 'age', 'Pre_E': '前测成绩'}.get(var, var)
        coef = model_a.params[var]
        se = model_a.bse[var]
        p = model_a.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        data.append([var_name, f"{coef:.4f}({se:.4f})", f"{p:.4f}", sig])
    data.append(['', '', '', ''])
    
    # 添加T->Outcome的回归结果
    data.append([f'回归方程: T -> {outcome_variable}', '', '', ''])
    data.append(['方程', equations[f'T->{outcome_variable}'], '', ''])
    data.append(['R²', f"{model_c.rsquared:.4f}", '', ''])
    data.append(['调整R²', f"{model_c.rsquared_adj:.4f}", '', ''])
    data.append(['自变量', '系数(标准误)', 'p值', '显著性'])
    for var in ['intercept', 'T', 'gender', 'age', f'Pre_{outcome_variable}']:
        var_name = {'intercept': '常数', 'T': 'T', 'gender': 'gender', 'age': 'age', f'Pre_{outcome_variable}': '前测成绩'}.get(var, var)
        coef = model_c.params[var]
        se = model_c.bse[var]
        p = model_c.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        data.append([var_name, f"{coef:.4f}({se:.4f})", f"{p:.4f}", sig])
    data.append(['', '', '', ''])
    
    # 添加E->Outcome的回归结果
    data.append([f'回归方程: E -> {outcome_variable}', '', '', ''])
    data.append(['方程', equations[f'E->{outcome_variable}'], '', ''])
    data.append(['R²', f"{model_b.rsquared:.4f}", '', ''])
    data.append(['调整R²', f"{model_b.rsquared_adj:.4f}", '', ''])
    data.append(['自变量', '系数(标准误)', 'p值', '显著性'])
    for var in ['intercept', 'E', 'T', 'gender', 'age', f'Pre_{outcome_variable}']:
        var_name = {'intercept': '常数', 'E': 'E', 'T': 'T', 'gender': 'gender', 'age': 'age', f'Pre_{outcome_variable}': '前测成绩'}.get(var, var)
        coef = model_b.params[var]
        se = model_b.bse[var]
        p = model_b.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        data.append([var_name, f"{coef:.4f}({se:.4f})", f"{p:.4f}", sig])
    
    df = pd.DataFrame(data, columns=['指标', '系数(标准误)', 'p值', '显著性'])
    return df

# 7. 保存结果到Excel
print(f"\n{'='*80}")
print("保存结果到Excel文件")
print(f"{'='*80}")

output_file = r'results2.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet1: T -> E -> CT
    df_CT = create_detailed_results_sheet(results_CT, 'CT')
    df_CT.to_excel(writer, sheet_name='T->E->CT', index=False)
    
    # Sheet2: T -> E -> A
    df_A = create_detailed_results_sheet(results_A, 'A')
    df_A.to_excel(writer, sheet_name='T->E->A', index=False)

print(f"\n结果已保存到: {output_file}")
print(f"  - Sheet1: T->E->CT (实验处理通过学习参与度影响计算思维能力)")
print(f"  - Sheet2: T->E->A (实验处理通过学习参与度影响学业成绩)")

print(f"\n{'='*80}")
print("分析完成！")
print(f"{'='*80}")
