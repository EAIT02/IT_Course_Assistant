import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math
# 计算习题ej的难度: 第一次做错本题的学习者的人数/做过本题的人数
# 计算学习者si的认知难度: 第一次做对的题目的难度求和/第一次做对的题目的数量
# 求习题的难度和学习者的认知难度
def d(S_num, C_num, E_num, X):
    de = [0 for i in range(E_num)]
    ds = [0 for i in range(S_num)]
    
    SF = [0 for i in range(E_num)] # 第一次做错第i道题目的人数
    num = [0 for i in range(E_num)] # 做过第j道题目的人数
    ET = [set() for i in range(S_num)] # 学习者i第一次做对的题目的编号
    
    for i in range(S_num):
        st = [0 for i in range(E_num)]
        for (j, p) in X[i]:
            num[j] += 1
            if p == 1: # 做对本题
                ET[i].add(j)
            else:
                if st[j] == 0:
                    SF[j] += 1
                    st[j] = 1
    for j in range(E_num):
        de[j] = SF[j] / num[j] if num[j]!= 0 else 0
        
    for i in range(S_num):
        tmp = 0        # 学习者i第一次做对的题目的难度的求和
        for ej in ET[i]:
            tmp += de[ej]
        ds[i] = tmp / len(ET[i]) if len(ET[i])!= 0 else 0
        
    return de, ds
# 计算每个学习者的WKC(si, Epsilon), 长度为 C_num 的one-hot编码向量
def get_WKC(S_num, C_num, E_num, CS, Epsilon):
    
    WKC = [[0 for i in range(C_num)] for i in range(S_num)]
    
    for si in range(S_num):
        for ck in range(C_num):
            if CS[si][ck] < Epsilon:
                WKC[si][ck] = 1
    
    return WKC
# 获得每个学习者的候选习题集合CE, CE[si]={习题编号, 不是01矩阵}
def CESFA(S_num, C_num, E_num, Q, WKC, de):
    # CE = [[0 for j in range(E_num)] for i in range(S_num)]
    CE = [[] for i in range(S_num)]
    Euclidean_norm = np.zeros((S_num, E_num)) # 初始化一个全0numpy矩阵
    # if E_num > 1000:
    #     T_window = (int)(E_num * 0.05) # 100
    #     N = 50
    # elif E_num > 100:  # Statics2011(154)
    #     N = 25
    # else:
    #     T_window = (int)(E_num * 0.5) # 10
    #     N = 0.7
    N = 3
    for si in range(S_num):
        delta = sum(de) / E_num
        for ej in range(E_num):
            a = 1 - Sim(Q[ej], WKC[si])
            b = abs(de[ej] - delta)
            Euclidean_norm[si][ej] = math.sqrt(a * a + b * b)
        if N > 1:
            CE[si] = np.argsort(Euclidean_norm[si])[:N]
        else:
            CE[si] = list(np.where(Euclidean_norm[si] > N)[0])

    # ls = [[] for ej in range(E_num)]
    # for ej in range(E_num):
    #     for ck in range(C_num):
    #         if Q[ej][ck] == 1:
    #             ls[ej].append(ck)
    # for si in range(S_num):
    #     for ej in range(E_num):
    #         cnt = 0 # 记录习题ej与WKC(s_i, epsilon)的重合的知识点的数量
    #         for ck in ls[ej]:
    #         # for ck in range(C_num):  # 知识点ck有重合, 则将习题ej纳入si的候选习题集合
    #         #     if Q[ej][ck] == 1 and WKC[si][ck] == 1:
    #             if WKC[si][ck] == 1:
    #                 cnt += 1
    #             if cnt >= (int)(sum(WKC[si]) * 0.04): # 重合的知识点数目大于弱知识点集合的 1/2
    #             # if cnt == 4:
    #                 CE[si].append(ej)
    #                 break
    return CE
# 计算Rel(si, epsilon)的平均习题难度
def Avd(S_num, C_num, E_num, Rel, de):
    # 注意: 此处的参数Rel是针对学习者si的习题推荐列表
    sum_d = 0
    num_Rel = 0
    for j in range(E_num):
        if Rel[j] == 1: # 第j道习题将要被推荐
            num_Rel += 1
            sum_d += de[j]
    ans = sum_d / num_Rel if num_Rel != 0 else 0
    return ans
# 习题推荐难度匹配
def Rdm(S_num, C_num, E_num, Rel, si, de, ds):
    # 注意: 此处的参数Rel是针对学习者si的习题推荐列表
    ans = Avd(S_num, C_num, E_num, Rel, de)
    ans = abs(ans - ds[si])
    return ans
# Rel中未被覆盖的知识点比例
def Rkc(S_num, C_num, E_num, Q, WKC, Rel, si):
    # 注意: 此处的参数Rel是针对学习者si的习题推荐列表
    
    num_WKC = 0
    num_kc = 0
    KC_Rel = [0 for ck in range(C_num)] # Rel中的习题所包含的知识点 one-hot向量
    
    for ck in range(C_num):
        if WKC[si][ck] == 1:
            num_WKC += 1
    
    for ej in range(E_num):
        if Rel[ej] == 1:
            for ck in range(C_num):
                if Q[ej][ck] == 1:
                    KC_Rel[ck] = 1
    num_kc = sum(KC_Rel)
    
    ans = num_WKC - num_kc
    if ans < 0:
        print(si, '###推荐太好了！偏差为0？？###')
    ans = ans / num_WKC if num_WKC != 0 else 0
    return ans
# 习题推荐数量比例(从CE中选择Rel)
def Prc(S_num, C_num, E_num, Rel, CE, si):
    return sum(Rel[si]) / sum(CE[si]) if sum(CE[si]) != 0 else 0
# 计算两个向量的余弦相似度cos值
def Sim(qi, qj):
    a = norm(qi)
    b = norm(qj)
    return dot(qi, qj) / (a * b) if a * b != 0 else 0
# Rel(si, Epsilon)的多样性
def Div(S_num, C_num, E_num, Q, Rel, si, Epsilon):    
    # 注意: 此处的参数Rel是针对学习者si的习题推荐列表
    num_Rel = sum(Rel[si])
    ans = 0
    for ei in range(E_num):
        for ej in range(E_num):
            if ei == ej:
                continue
            ans += 1 - Sim(Q[ei], Q[ej])
    return ans / (num_Rel * (num_Rel - 1)) if num_Rel > 1 else 0
    
# Rel(si, Epsilon)的平滑程度
def Smt(S_num, C_num, E_num, Rel, si, de):
    # 注意: 此处的参数Rel是针对学习者si的习题推荐列表
    num_Rel = sum(Rel[si])
    ans = 0
    for ej in range(E_num - 1):
        ans += abs(de[ej] - de[ej + 1])
    return ans / num_Rel if num_Rel != 0 else 0
    
# 计算Rel(si, Epsilon)的推荐花费
def Rct(S_num, C_num, E_num, Q, WKC, de, ds, CE, Rel, si, Epsilon):
    
    f1 = Rdm(S_num, C_num, E_num, Rel, si, de, ds)
    f2 = Rkc(S_num, C_num, E_num, Q, WKC, Rel, si)
    f3 = Prc(S_num, C_num, E_num, Rel, CE=CE, si=si)
    f4 = Div(S_num, C_num, E_num, Q, Rel, si, Epsilon)
    f5 = Smt(S_num, C_num, E_num, Rel, si, de)
# 假如有项函数指标为0?那么需要将其重置为1吗？
    return f1 * f2 * f3 * f4 * f5
    
# 输出推荐给每个学习者si的习题推荐列表Rel(si, Epsilon)
def Print_Rel(S_num, C_num, E_num, Rel):
    for si in range(S_num):
        print(si, end=':')
        ls = []
        for ej in range(E_num):
            if Rel[si][ej] == 1:
                ls.append(ej)
        print(ls)
    
    