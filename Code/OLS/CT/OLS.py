# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from linearmodels import PanelOLS
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置中文显示（如果需要）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 1. 读取数据
# pretest = pd.read_excel('pretest_data.xlsx')
# posttest = pd.read_excel('posttest_data.xlsx')

# 2. 创建模拟的学生信息（实际应用中应使用真实数据）
# 假设有10个班级，每个班级20-25名学生
# np.random.seed(42)
# n_students = len(pretest) + len(posttest)

# 创建学生ID
# student_ids = [f"S{i+1:03d}" for i in range(n_students)]
# student_ids = [i for i in range(n_students)]


# 随机分配性别 (1=男, 0=女)
# gender = np.random.choice([0, 1], size=n_students, p=[0.45, 0.55])

# gender = data['gender']
# # 随机分配年龄 (15-18岁)
# age = data['age']#np.random.randint(15, 19, size=n_students)
#
# # 随机分配班级 (10个班级)
# classes = data['class'] # np.random.choice([f"C{i+1}" for i in range(10)], size=n_students)


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
pretest = pd.read_excel('pretest_data.xlsx')
posttest = pd.read_excel('posttest_data.xlsx')

# 2. 创建分析数据集
np.random.seed(42)
n_students = len(pretest) + len(posttest)

# 创建学生信息
student_ids = [f"S{i+1:03d}" for i in range(n_students)]
data = pd.read_excel("age.xlsx")
gender = data['gender']#np.random.choice([0, 1], size=n_students, p=[0.45, 0.55])
age = data['age']
classes = data['class']

# 合并数据
pretest_scores = pd.concat([pretest['实验组'], pretest['对照组']]).reset_index(drop=True)
posttest_scores = pd.concat([posttest['实验组'], posttest['对照组']]).reset_index(drop=True)
treatment = np.concatenate([np.ones(len(pretest['实验组'])), np.zeros(len(pretest['对照组']))])

df = pd.DataFrame({
    'student_id': student_ids,
    # 'class': classes,
    'treatment': treatment,
    'pretest': pretest_scores,
    'posttest': posttest_scores,
    'gender': gender,
    'age': age
})

# 3. 添加常数项并创建性别虚拟变量
df['const'] = 1
df['male'] = df['gender'].apply(lambda x: 1 if x == 1 else 0)

# 4. 定义自变量和因变量
X = df[['const', 'treatment', 'pretest', 'male', 'age']]
y = df['posttest']

# 5. 最小二乘法回归分析
model = sm.OLS(y, X)
results = model.fit()

# 6. 打印回归结果
print(results.summary())

# 7. 提取关键结果
treatment_coef = results.params['treatment']
treatment_pvalue = results.pvalues['treatment']
treatment_ci = results.conf_int().loc['treatment']
pretest_coef = results.params['pretest']
pretest_pvalue = results.pvalues['pretest']
rsquared = results.rsquared
adj_rsquared = results.rsquared_adj

# 8. 生成结果报告
report = f"""
### 最小二乘法回归分析结果

**模型设定：**
因变量：后测分数 (posttest)
自变量：
- 处理组 (treatment)：1=实验组, 0=对照组
- 前测分数 (pretest)
- 性别 (male)：1=男, 0=女
- 年龄 (age)

**回归结果：**
1. AI课程助手效应 (treatment):
   - 系数 = {treatment_coef:.4f}
   - 95%置信区间: [{treatment_ci[0]:.4f}, {treatment_ci[1]:.4f}]
   - p值 = {treatment_pvalue:.4f}
   - 显著性: {'显著' if treatment_pvalue < 0.05 else '不显著'}

2. 前测分数效应 (pretest):
   - 系数 = {pretest_coef:.4f}
   - p值 = {pretest_pvalue:.4f}
   - 显著性: {'显著' if pretest_pvalue < 0.05 else '不显著'}

3. 性别效应 (male):
   - 系数 = {results.params['male']:.4f}
   - p值 = {results.pvalues['male']:.4f}

4. 年龄效应 (age):
   - 系数 = {results.params['age']:.4f}
   - p值 = {results.pvalues['age']:.4f}

**模型统计量：**
- R² = {rsquared:.4f}
- 调整R² = {adj_rsquared:.4f}
- F统计量 = {results.fvalue:.2f}
- 观测数 = {results.nobs}
"""

print(report)