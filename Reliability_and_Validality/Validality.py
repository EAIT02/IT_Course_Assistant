import pandas as pd
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import os
import numpy as np

try:
    from semopy import Model, semplot, stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    print("警告：未安装semopy库，无法进行CFA分析。请运行: pip install semopy")

def calculate_kmovalue(file_path): # 读取15维度学习参与度数据，计算并返回KMO值
    """ 参数: file_path: Excel文件路径
    返回:
        kmo_model: 整体KMO值
        kmo_per_item: 每个维度的KMO """

    try: # 读取Excel数据
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 '{file_path}' 不存在，请检查路径是否正确")

    # 数据校验
    # 检查列数是否为15
    if df.shape[1] != 15:
        raise ValueError(f"数据包含 {df.shape[1]} 列，预期15列（15个维度）")

    # 检查数据范围是否在1~5之间
    if not ((df >= 1).all().all() and (df <= 5).all().all()):
        print("警告：部分数据超出1~5的计分范围，可能影响KMO计算结果")

    # 计算KMO值（返回每个题项的KMO和整体KMO）
    kmo_per_item, kmo_model = calculate_kmo(df)

    # 输出每个题项的KMO值（辅助判断是否有异常题项）
    print("各维度的KMO值：")
    for i, kmo in enumerate(kmo_per_item, 1):
        print(f"维度{i}: {kmo:.4f}")

    return kmo_model, kmo_per_item

def calculate_bartlett(file_path):
    """
    读取15维度学习参与度数据，计算并返回Bartlett球形度检验结果
    参数:
        file_path: Excel文件路径
    返回:
        chi_square: 卡方值
        p_value: p值
        dof: 自由度
    """
    # 读取Excel数据
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 '{file_path}' 不存在，请检查路径是否正确")

    # 数据校验
    # 检查列数是否为15
    if df.shape[1] != 15:
        raise ValueError(f"数据包含 {df.shape[1]} 列，预期15列（15个维度）")

    # 计算Bartlett球形度检验
    chi_square, p_value = calculate_bartlett_sphericity(df)

    # 计算自由度：自由度 = (k * (k - 1)) / 2，其中k为变量数量
    k = df.shape[1]
    dof = (k * (k - 1)) // 2

    print(f"\nBartlett球形度检验结果：")
    print(f"卡方值: {chi_square:.4f}")
    print(f"自由度: {dof}")
    print(f"p值: {p_value:.4e}")

    # 结果解释
    if p_value < 0.001:
        print("结论: p < 0.001，变量间存在显著相关性，适合进行因子分析")
    elif p_value < 0.05:
        print("结论: p < 0.05，变量间存在显著相关性，适合进行因子分析")
    else:
        print("结论: p ≥ 0.05，变量间相关性不显著，不适合进行因子分析")

    return chi_square, p_value, dof

def calculate_cfa(file_path):
    """
    进行验证性因子分析（CFA）
    参数:
        file_path: Excel文件路径
   返回:
        cfa_results: 包含CFA拟合指标的字典
    """
    if not SEMOPY_AVAILABLE:
        raise ImportError("semopy库未安装，无法进行CFA分析")

    # 读取Excel数据
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 '{file_path}' 不存在，请检查路径是否正确")

    # 数据校验
    if df.shape[1] != 15:
        raise ValueError(f"数据包含 {df.shape[1]} 列，预期15列（15个维度）")

    # 重命名列为X1到X15
    df_cfa = df.copy()
    df_cfa.columns = [f'X{i}' for i in range(1, 16)]

    # 定义CFA模型
    # 假设15个维度分为3个因子：行为参与（1-5）、认知参与（6-10）、情感参与（11-15）
    model_desc = """
    # 定义三个潜变量
    行为参与 =~ X1 + X2 + X3 + X4 + X5
    认知参与 =~ X6 + X7 + X8 + X9 + X10
    情感参与 =~ X11 + X12 + X13 + X14 + X15
    """

    print(f"\n{'='*60}")
    print("验证性因子分析（CFA）")
    print(f"{'='*60}")
    print("模型结构：")
    print("  - 行为参与: 维度1-5")
    print("  - 认知参与: 维度6-10")
    print("  - 情感参与: 维度11-15")

    # 创建模型并拟合
    model = Model(model_desc)
    results = model.fit(df_cfa)

    # 获取拟合指标
    fit_measures = stats.calc_stats(model)

    # 将Series转换为标量值
    chi2 = float(fit_measures['chi2'].iloc[0] if hasattr(fit_measures['chi2'], 'iloc') else fit_measures['chi2'])
    dof = int(fit_measures['DoF'].iloc[0] if hasattr(fit_measures['DoF'], 'iloc') else fit_measures['DoF'])
    pvalue = float(fit_measures['chi2 p-value'].iloc[0] if hasattr(fit_measures['chi2 p-value'], 'iloc') else fit_measures['chi2 p-value'])
    cfi = float(fit_measures['CFI'].iloc[0] if hasattr(fit_measures['CFI'], 'iloc') else fit_measures['CFI'])
    tli = float(fit_measures['TLI'].iloc[0] if hasattr(fit_measures['TLI'], 'iloc') else fit_measures['TLI'])
    rmsea = float(fit_measures['RMSEA'].iloc[0] if hasattr(fit_measures['RMSEA'], 'iloc') else fit_measures['RMSEA'])
    gfi = float(fit_measures['GFI'].iloc[0] if hasattr(fit_measures['GFI'], 'iloc') else fit_measures['GFI'])
    agfi = float(fit_measures['AGFI'].iloc[0] if hasattr(fit_measures['AGFI'], 'iloc') else fit_measures['AGFI'])
    
    srmr = None
    if 'SRMR' in fit_measures:
        srmr = float(fit_measures['SRMR'].iloc[0] if hasattr(fit_measures['SRMR'], 'iloc') else fit_measures['SRMR'])

    print(f"\nCFA拟合指标：")
    print(f"卡方值: {chi2:.4f}")
    print(f"自由度: {dof:.0f}")
    print(f"卡方/自由度: {chi2/dof:.4f}")
    print(f"p值: {pvalue:.4e}")
    print(f"CFI: {cfi:.4f}")
    print(f"TLI: {tli:.4f}")
    print(f"RMSEA: {rmsea:.4f}")
    if srmr is not None:
        print(f"SRMR: {srmr:.4f}")
    else:
        print(f"SRMR: N/A")
    print(f"GFI: {gfi:.4f}")
    print(f"AGFI: {agfi:.4f}")

    # 模型拟合评价
    print(f"\n模型拟合评价：")
    chi2_df = chi2 / dof
    if chi2_df < 2:
        print(f"  - 卡方/自由度 ({chi2_df:.4f}): 优秀")
    elif chi2_df < 3:
        print(f"  - 卡方/自由度 ({chi2_df:.4f}): 可接受")
    else:
        print(f"  - 卡方/自由度 ({chi2_df:.4f}): 不理想")

    if cfi >= 0.95:
        print(f"  - CFI ({cfi:.4f}): 优秀")
    elif cfi >= 0.90:
        print(f"  - CFI ({cfi:.4f}): 可接受")
    else:
        print(f"  - CFI ({cfi:.4f}): 不理想")

    if tli >= 0.95:
        print(f"  - TLI ({tli:.4f}): 优秀")
    elif tli >= 0.90:
        print(f"  - TLI ({tli:.4f}): 可接受")
    else:
        print(f"  - TLI ({tli:.4f}): 不理想")

    if rmsea <= 0.05:
        print(f"  - RMSEA ({rmsea:.4f}): 优秀")
    elif rmsea <= 0.08:
        print(f"  - RMSEA ({rmsea:.4f}): 可接受")
    else:
        print(f"  - RMSEA ({rmsea:.4f}): 不理想")

    # 获取因子载荷和参数
    estimates = model.inspect()

    # 计算SRMR（标准化均方根残差）
    s_matrix = model.calc_sigma()
    s_obs = df_cfa.cov()
    
    # 处理s_matrix（可能是tuple或list）
    if isinstance(s_matrix, tuple):
        s_matrix = s_matrix[0]  # 取第一个元素
    if isinstance(s_matrix, (list, tuple)):
        s_matrix = np.array(s_matrix)
    elif hasattr(s_matrix, 'values'):
        s_matrix = s_matrix.values
    
    # 确保s_obs是numpy数组
    if hasattr(s_obs, 'values'):
        s_obs_array = s_obs.values
    else:
        s_obs_array = np.array(s_obs)
    
    residual = s_obs_array - s_matrix
    srmr = np.sqrt(np.mean((residual / np.sqrt(np.outer(np.diag(s_obs_array), np.diag(s_obs_array))))**2))

    print(f"SRMR: {srmr:.4f}")
    if srmr <= 0.08:
        print(f"  - SRMR ({srmr:.4f}): 优秀")
    elif srmr <= 0.10:
        print(f"  - SRMR ({srmr:.4f}): 可接受")
    else:
        print(f"  - SRMR ({srmr:.4f}): 不理想")

    print(f"\n{'='*60}")
    print("因子载荷：")
    print(f"{'='*60}")
    print(estimates.to_string())

    # 计算标准化载荷、CR和AVE
    print(f"\n{'='*60}")
    print("信度和效度指标：")
    print(f"{'='*60}")

    # 提取因子载荷和误差方差
    factor_loadings = {}
    error_variances = {}
    standardized_loadings = {}
    
    for idx, row in estimates.iterrows():
        if row['op'] == '~' and row['rval'] in ['行为参与', '认知参与', '情感参与']:
            var_name = row['lval']
            loading = row['Estimate']
            factor_name = row['rval']
            
            if factor_name not in factor_loadings:
                factor_loadings[factor_name] = []
            factor_loadings[factor_name].append(loading)
            
            # 保存标准化载荷
            standardized_loadings[f"{var_name}_标准化载荷"] = loading
            
        elif row['op'] == '~~' and row['lval'].startswith('X'):
            var_name = row['lval']
            error_var = row['Estimate']
            error_variances[var_name] = error_var

    # 计算每个因子的CR和AVE
    cr_ave_results = {}
    for factor_name, loadings_list in factor_loadings.items():
        loadings = np.array(loadings_list)
        squared_loadings = loadings ** 2
        
        # AVE = sum(λ²) / number of indicators
        ave = np.sum(squared_loadings) / len(loadings)
        
        # CR = (sum(λ))² / [(sum(λ))² + sum(θ)]
        sum_loading = np.sum(loadings)
        sum_error = 0
        
        # 找到该因子对应的题项的误差方差
        for idx, row in estimates.iterrows():
            if row['op'] == '~' and row['rval'] == factor_name:
                var_name = row['lval']
                if var_name in error_variances:
                    sum_error += error_variances[var_name]
        
        cr = (sum_loading ** 2) / ((sum_loading ** 2) + sum_error)
        
        cr_ave_results[factor_name] = {
            'AVE': ave,
            'CR': cr
        }
        
        print(f"\n{factor_name}：")
        print(f"  AVE (平均方差提取量): {ave:.4f}")
        print(f"  CR (组合信度): {cr:.4f}")
        print(f"  标准化载荷: {', '.join([f'{l:.4f}' for l in loadings])}")
        
        # AVE和CR的评价
        if ave >= 0.5:
            print(f"  AVE评价: 优秀 (≥0.5)")
        else:
            print(f"  AVE评价: 不理想 (<0.5)")
            
        if cr >= 0.7:
            print(f"  CR评价: 优秀 (≥0.7)")
        elif cr >= 0.6:
            print(f"  CR评价: 可接受 (≥0.6)")
        else:
            print(f"  CR评价: 不理想 (<0.6)")

    # 返回结果
    cfa_results = {
        "卡方值": chi2,
        "自由度": dof,
        "卡方/自由度": chi2 / dof,
        "p值": pvalue,
        "CFI": cfi,
        "TLI": tli,
        "RMSEA": rmsea,
        "SRMR": srmr,
        "GFI": gfi,
        "AGFI": agfi
    }

    # 添加标准化载荷
    cfa_results.update(standardized_loadings)

    # 添加CR和AVE
    for factor_name in ['行为参与', '认知参与', '情感参与']:
        if factor_name in cr_ave_results:
            cfa_results[f"{factor_name}_AVE"] = cr_ave_results[factor_name]['AVE']
            cfa_results[f"{factor_name}_CR"] = cr_ave_results[factor_name]['CR']

    # 模型拟合评价
    cfa_results["模型拟合评价"] = "优秀" if (chi2_df < 2 and cfi >= 0.95 and tli >= 0.95 and rmsea <= 0.05 and srmr <= 0.08) else "可接受" if (chi2_df < 3 and cfi >= 0.90 and tli >= 0.90 and rmsea <= 0.08 and srmr <= 0.10) else "不理想"

    return cfa_results, estimates

if __name__ == "__main__":
    # 输入文件路径
    files = ["信效度检验/LE_pre.xlsx", "信效度检验/LE_exp.xlsx"]
    output_file = "信效度检验/results1.xlsx"

    # 检查文件是否存在
    results = []
    cfa_results = []

    for file_name in files:
        if not os.path.exists(file_name):
            print(f"错误：未找到文件 '{file_name}'")
            continue

        print(f"\n{'='*60}")
        print(f"正在处理文件: {file_name}")
        print(f"{'='*60}")

        try:
            # 计算KMO值
            kmo_model, kmo_per_item = calculate_kmovalue(file_name)
            print(f"\n整体KMO值: {kmo_model:.4f}")

            # 结果解释
            print("\nKMO值解释参考：")
            print(" - ≥0.9：极佳，非常适合因子分析")
            print(" - 0.8~0.9：良好，适合因子分析")
            print(" - 0.7~0.8：一般，基本适合因子分析")
            print(" - 0.6~0.7：较差，勉强适合因子分析")
            print(" - <0.6：差，不适合因子分析")

            # 针对学习参与度数据的建议
            if kmo_model < 0.7:
                print("\n建议：KMO值低于0.7，可检查是否有低KMO值的题项（<0.5），考虑删除后重新计算")

            # 计算Bartlett球形度检验
            chi_square, p_value, dof = calculate_bartlett(file_name)

            # 保存KMO和Bartlett结果
            result = {
                "文件名": file_name,
                "整体KMO值": kmo_model,
                "KMO评价": "极佳" if kmo_model >= 0.9 else "良好" if kmo_model >= 0.8 else "一般" if kmo_model >= 0.7 else "较差" if kmo_model >= 0.6 else "差",
                "Bartlett卡方值": chi_square,
                "Bartlett自由度": dof,
                "Bartlett p值": p_value,
                "Bartlett结论": "适合因子分析" if p_value < 0.05 else "不适合因子分析"
            }

            # 添加各维度的KMO值
            for i, kmo in enumerate(kmo_per_item, 1):
                result[f"维度{i}_KMO"] = kmo

            results.append(result)

            # 进行CFA分析
            if SEMOPY_AVAILABLE:
                try:
                    cfa_result, loadings = calculate_cfa(file_name)
                    cfa_result["文件名"] = file_name
                    cfa_results.append(cfa_result)
                except Exception as e:
                    print(f"CFA分析失败：{str(e)}")
            else:
                print("\n跳过CFA分析（semopy库未安装）")

        except Exception as e:
            print(f"计算失败：{str(e)}")

    # 输出结果到Excel文件
    if results:
        print(f"\n{'='*60}")
        print(f"正在将结果保存到: {output_file}")
        print(f"{'='*60}")

        df_results = pd.DataFrame(results)

        # 使用ExcelWriter保存多个sheet
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='KMO和Bartlett检验', index=False)

            if cfa_results:
                try:
                    df_cfa = pd.DataFrame(cfa_results)
                    df_cfa.to_excel(writer, sheet_name='CFA验证性因子分析', index=False)
                except Exception as e:
                    print(f"创建CFA DataFrame失败：{str(e)}")
                    import traceback
                    traceback.print_exc()

        print(f"结果已成功保存到 '{output_file}'")
        print("\nSheet1 - KMO和Bartlett检验结果预览：")
        print(df_results.to_string())

        if cfa_results:
            print(f"\nSheet2 - CFA验证性因子分析结果预览：")
            print(pd.DataFrame(cfa_results).to_string())
    else:
        print("\n没有成功计算任何结果")
