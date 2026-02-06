import pandas as pd

def cronbach_alpha(df):
    """
    计算克隆巴赫Alpha系数
    参数:
        df: 包含题项数据的DataFrame，每行是一个样本，每列是一个题项
    返回:
        alpha: 克隆巴赫Alpha系数值
    """
    # 计算题项数量
    k = df.shape[1]

    # 计算每个题项的方差（样本方差，自由度为n-1）
    item_variances = df.var(axis=0, ddof=1)
    sum_item_var = item_variances.sum()

    # 计算所有题项的总分
    total_scores = df.sum(axis=1)

    # 计算总分的方差（样本方差）
    total_var = total_scores.var(ddof=1)

    # 应用克隆巴赫Alpha公式
    alpha = (k / (k - 1)) * (1 - sum_item_var / total_var)
    return alpha


if __name__ == "__main__":
    # 读取Excel文件（请替换为你的文件路径）
    file_path = "LE_pre.xlsx"
    try:
        # 读取所有数据（假设第一行是列名，数据从第二行开始）
        df = pd.read_excel(file_path)

        # 检查数据维度是否符合要求
        if df.shape[1] != 15:
            print(f"警告：数据包含{df.shape[1]}列，预期15列（15个维度）")

        # 检查数据范围是否在1~5之间
        if not ((df >= 1).all().all() and (df <= 5).all().all()):
            print("警告：部分数据超出1~5的计分范围")

        # 将数据分为三个维度
        df_dim1 = df.iloc[:, 0:5]  # 第1-5题
        df_dim2 = df.iloc[:, 5:10]  # 第6-10题
        df_dim3 = df.iloc[:, 10:15]  # 第11-15题

        # 计算三个维度的克隆巴赫Alpha系数
        alpha_dim1 = cronbach_alpha(df_dim1)
        alpha_dim2 = cronbach_alpha(df_dim2)
        alpha_dim3 = cronbach_alpha(df_dim3)
        alpha_total = cronbach_alpha(df)  # 整体15个题

        # 输出结果，保留4位小数
        print("克隆巴赫Alpha系数结果:")
        print(f"维度1（第1-5题）: {alpha_dim1:.4f}")
        print(f"维度2（第6-10题）: {alpha_dim2:.4f}")
        print(f"维度3（第11-15题）: {alpha_dim3:.4f}")
        print(f"整体（15个题）: {alpha_total:.4f}")

        # 解释结果参考标准
        print("\n结果解释参考:")
        print(" - α ≥ 0.9: 内部一致性极好")
        print(" - 0.8 ≤ α < 0.9: 内部一致性良好")
        print(" - 0.7 ≤ α < 0.8: 内部一致性可接受")
        print(" - 0.6 ≤ α < 0.7: 内部一致性较差（探索性研究可接受）")
        print(" - α < 0.6: 内部一致性差")

    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'，请检查文件路径是否正确")
    except Exception as e:
        print(f"计算过程中发生错误: {str(e)}")
