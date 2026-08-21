import random
import numpy as np
import time
import pandas as pd

import my_functions
# import Genetic_Algorithm

# 1. 模拟生成学生的做题记录(针对真实的数据，只需要再编写一个数据预处理的代码即可！)
# 2. cs认知状态由认知诊断模型获得！
# 3. 假如有项函数指标为0?那么需要将其重置为1吗？
# 4. 如何求候选习题集CE呢？ 利用弱知识点覆盖的思想(再具体一点)
# 5. 思考如何去实现popular算法？
# 6. 针对不同的数据集进行数据预处理，得到输入到GA里的参数


class MyDataset():
    def __init__(self, Epsilon, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001, Rec_num=5):
        self.Epsilon = Epsilon # 弱知识点覆盖因子
        self.delta = delta     # 难度系数
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.new_offsprings_num = new_offsprings_num  # 交叉 变异需新增的个体比例
        self.Rec_num = Rec_num # 最后给每一个学习者推荐习题列表时有Rec_num个选择列表
    # def forward(self, name):
    #     if name == "Frcsub":
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.Frcsub()
    #     elif name == 'Math1':
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.Math1()
    #     elif name == 'Math2':
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.Math2()
    #     elif name == 'ASSISTments2009':
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.ASSISTments2009()
    #     elif name == 'ASSISTments2017':
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.ASSISTments2017()
    #     elif name == 'Statics2011':
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.Statics2011()
    #     elif name == 'Algebra2005':
    #         self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = self.Algebra2005()
    #     GA = Genetic_Algorithm.Genetic_Algorithm(CE=self.CE, S_num=self.S_num, C_num=self.C_num, X=self.X, Q=self.Q, de=self.de, ds=self.ds, WKC=self.WKC,
    #                                                  delta=self.delta, population_size=self.population_size,
    #                                                  generation_num=self.generation_num,
    #                                                  new_offsprings_num=self.new_offsprings_num, pc=self.pc, pm=self.pm,
    #                                                  Rec_num=self.Rec_num)
    #     GA.evaluate()
    def ASSISTments2017(self):
        data = pd.read_csv('../../data/ASSISTments2017/ASSISTments2017.csv')
        S_num, E_num, C_num = data['user_id'].nunique(), data['problem_id'].nunique(), data['skill_id'].nunique()
        user_id, problem_id, correct, skill_id = list(data['user_id']), list(data['problem_id']), list(
            data['correct']), list(data['skill_id'])
        X = [[] for i in range(S_num)]
        Q = [[0 for j in range(C_num)] for i in range(E_num)]
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for i in range(len(user_id)):  # 枚举所有的做题记录
            si = user_id[i]
            ej = problem_id[i]
            answer = correct[i]
            X[si].append((ej, answer))
            if answer == 1:
                Q[ej][skill_id[i]] = 1
                CS[si][skill_id[i]] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de)
        for i in range(len(CE)):
            print(len(CE[i]))
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE
    def ASSISTments2009(self):
        data = pd.read_csv('../../data/ASSISTments0910/ASSISTments0910.csv')
        S_num, E_num, C_num = data['user_id'].nunique(), data['problem_id'].nunique(), data['skill_id'].nunique()
        user_id, problem_id, correct, skill_id = list(data['user_id']), list(data['problem_id']), list(data['correct']), list(data['skill_id'])

        X = [[] for i in range(S_num)]
        Q = [[0 for j in range(C_num)] for i in range(E_num)]
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for i in range(len(user_id)): # 枚举所有的做题记录
            si = user_id[i]
            ej = problem_id[i]
            answer = correct[i]
            X[si].append((ej,answer))
            if answer == 1:
                Q[ej][skill_id[i]] = 1
                CS[si][skill_id[i]] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de)
        # for i in range(len(CE)):
        #     print(len(CE[i]))
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE
        # print(user_id, type(user_id), len(user_id))
    def Math1(self):
        path = '../../data/Math2015/Math1/q.txt'
        Q = np.loadtxt(path).astype(int)  # Q矩阵 习题-知识点
        # print(Q, type(Q), Q.shape, len(Q), len(Q[0]))
        path = '../../data/Math2015/Math1/data.txt'
        raw_X = np.loadtxt(path)  # 原始做题序列 学习者-习题
        for i in range(len(raw_X)):
            for j in range(len(raw_X[0])):
                if raw_X[i][j] >= 0.5:
                    raw_X[i][j] = 1
                else:
                    raw_X[i][j] = 0
        S_num, E_num, C_num = raw_X.shape[0], raw_X.shape[1], Q.shape[1]
        X = [[] for i in range(raw_X.shape[0])]  # 改成元组形式 学习者-(习题, correct)
        for si in range(S_num):
            for ej in range(E_num):
                if raw_X[si][ej] == 0:
                    X[si].append((ej, 0))
                else:
                    X[si].append((ej, 1))
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for si in range(S_num):
            for (ej, correct) in X[si]:
                if correct == 0:
                    continue
                for ck in range(C_num):
                    if Q[ej][ck] == 1:
                        CS[si][ck] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de)
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE
    def Math2(self):
        path = '../../data/Math2015/Math2/q.txt'
        Q = np.loadtxt(path).astype(int)  # Q矩阵 习题-知识点
        # print(Q, type(Q), Q.shape, len(Q), len(Q[0]))
        path = '../../data/Math2015/Math2/data.txt'
        raw_X = np.loadtxt(path)  # 原始做题序列 学习者-习题
        for i in range(len(raw_X)):
            for j in range(len(raw_X[0])):
                if raw_X[i][j] >= 0.5:
                    raw_X[i][j] = 1
                else:
                    raw_X[i][j] = 0
        S_num, E_num, C_num = raw_X.shape[0], raw_X.shape[1], Q.shape[1]
        X = [[] for i in range(raw_X.shape[0])]  # 改成元组形式 学习者-(习题, correct)
        for si in range(S_num):
            for ej in range(E_num):
                if raw_X[si][ej] == 0:
                    X[si].append((ej, 0))
                else:
                    X[si].append((ej, 1))
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for si in range(S_num):
            for (ej, correct) in X[si]:
                if correct == 0:
                    continue
                for ck in range(C_num):
                    if Q[ej][ck] == 1:
                        CS[si][ck] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de)
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE
    def Frcsub(self):
        path = '../../data/Math2015/Frcsub/q.txt'
        Q = np.loadtxt(path).astype(int) # Q矩阵 习题-知识点
        # print(Q, type(Q), Q.shape, len(Q), len(Q[0]))
        path = '../../data/Math2015/Frcsub/data.txt'
        raw_X = np.loadtxt(path).astype(int) # 原始做题序列 学习者-习题
        S_num, E_num, C_num = raw_X.shape[0], raw_X.shape[1], Q.shape[1]
        X = [[] for i in range(raw_X.shape[0])] # 改成元组形式 学习者-(习题, correct)
        for si in range(S_num):
            for ej in range(E_num):
                if raw_X[si][ej] == 0:
                    X[si].append((ej, 0))
                else:
                    X[si].append((ej, 1))
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for si in range(S_num):
            for (ej, correct) in X[si]:
                if correct == 0:
                    continue
                for ck in range(C_num):
                    if Q[ej][ck] == 1:
                        CS[si][ck] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de)
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE
    def Statics2011(self):
        data = pd.read_csv('../../data/Statics2011/Statics2011.csv')
        problem_id = data['problem_id']
        skill_id = data['skill_id']
        correct = data['correct']
        user_id = data['user_id']
        skill_id = skill_id.str.split(',').apply(list)
        all_skill = [item for sublist in skill_id for item in sublist]
        skill_encoded, _ = pd.factorize(all_skill)
        encoded_skill = [[] for i in range(len(skill_id))]
        start_pos = 0
        for i, sublist in enumerate(skill_id):
            encoded_skill[i] = list(skill_encoded[start_pos: start_pos + len(sublist)])
            start_pos += len(sublist)
        # print(encoded_skill, len(encoded_skill))
        # encoded_skill_id = [list(skill_encoded[i:i+len(sublist)]) for i, sublist in enumerate(skill_id)]
        S_num, C_num, E_num = data['user_id'].nunique(), max(skill_encoded) + 1, data['problem_id'].nunique()
        Q = [[0 for j in range(C_num)] for i in range(E_num)]
        X = [[] for i in range(S_num)]
        for k in range(len(encoded_skill)):
            ej = problem_id[k]
            answer = correct[k]
            si = user_id[k]
            X[si].append((ej, answer))
            for ck in encoded_skill[k]:
                # print(E_num, C_num, ej, ck)
                Q[ej][ck] = 1
        print(S_num, E_num, C_num)
        # print(Q, '\n', len(Q), len(Q[0]))
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for si in range(S_num):
            for (ej, correct) in X[si]:
                if correct == 0:
                    continue
                for ck in range(C_num):
                    if Q[ej][ck] == 1:
                        CS[si][ck] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de)
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE # 331, 154, 130
    def Algebra2005(self):
        path = '../../data/Algebra2005/Algebra2005.csv'
        data = pd.read_csv(path)
        # df['user_id'], _ = pd.factorize(df['user_id'], sort=True) # 离散化 排序从小到大
        # df['problem_id'], _ = pd.factorize(df['problem_id'], sort=True)
        # df['skill_id'], _ = pd.factorize(df['skill_id'], sort=True)
        S_num, E_num, C_num = data['user_id'].nunique(), data['problem_id'].nunique(), data['skill_id'].nunique()
        user_id, problem_id, correct, skill_id = list(data['user_id']), list(data['problem_id']), list(
                                                 data['correct']), list(data['skill_id'])
        X = [[] for i in range(S_num)]
        Q = [[0 for j in range(C_num)] for i in range(E_num)]
        CS = [[0 for i in range(C_num)] for j in range(S_num)]
        for i in range(len(user_id)):  # 枚举所有的做题记录
            si = user_id[i]
            ej = problem_id[i]
            answer = correct[i]
            X[si].append((ej, answer))
            if answer == 1:
                Q[ej][skill_id[i]] = 1
                CS[si][skill_id[i]] = 1
        de, ds = my_functions.d(S_num, C_num, E_num, X)
        WKC = my_functions.get_WKC(S_num, C_num, E_num, CS, self.Epsilon)
        CE = my_functions.CESFA(S_num, C_num, E_num, Q, WKC, de) # 到此运行时间约1min
        # print('运行时间为', time.time() - stime)
        # print(S_num, E_num, C_num)
        # for i in range(S_num):
        #     print(i, len(CE[i]))
        return S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE # 574 1084 436


stime = time.time()
mydata = MyDataset(Epsilon=0.5, delta=0.5, population_size=50,
                   generation_num=50, new_offsprings_num=0.3, pc=0.6, pm=0.001, Rec_num=5)
mydata.Frcsub()
# mydata.forward('ASSISTments2009')
# print('运行时间为',time.time() - stime)
