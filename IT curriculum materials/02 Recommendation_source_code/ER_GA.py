import random
from numpy import dot
from numpy.linalg import norm
import statistics

import sys
sys.path.append("../")


# 种群中染色体的长度是固定的==E_num[si], 但染色体的数目是population_size
class Genetic_Algorithm():
    def __init__(self, CE, S_num, C_num, X, Q, de, ds, WKC, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001,
                 Rec_num=5):
        self.E_num = [len(CE[i]) for i in range(S_num)] # 为学习者si推荐的候选习题集的数目
        self.CE = CE
        self.S_num = S_num
        self.C_num = C_num
        self.Q = Q
        self.de = de
        self.ds = ds
        self.WKC = WKC
        self.delta = delta   #难度系数
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.new_offsprings_num = new_offsprings_num # 交叉 变异需新增的个体比例
        self.Rec_num = Rec_num
        self.X = X
        self.all_correct_knowledge = [set() for i in range(self.S_num)]#学习者si正确回答的习题包含的知识点
        for si in range(self.S_num):
            for (ej, correct) in self.X[si]:
                if correct == 0:
                    continue
                for ck in range(self.C_num):
                    if self.Q[ej][ck] == 1:
                        self.all_correct_knowledge[si].add(ck)

    def evaluate(self): # 评估指标，直接输出
        Rel = self.forward() # S_num * self.Rec_num * 变化长度的推荐列表
        Acc = [0 for i in range(self.S_num)]
        Nov = [0 for i in range(self.S_num)]
        Div = [0 for i in range(self.S_num)]
        Proximity = [0 for i in range(self.S_num)]
        Coverage = [0 for i in range(self.S_num)]
        Quantity = [0 for i in range(self.S_num)]
        Volatility = [0 for i in range(self.S_num)]

        for si in range(self.S_num):
            if self.E_num[si] <= 0:
                continue
            if si >= self.S_num // 10: # 对于学习者较多的数据，可以提前终止
                break
            for j in range(self.Rec_num): # 分别计算推荐列表的三个指标
                Rel_list = Rel[si][j]
                acc, nov, div, Pro, Cov, Qua, Vol = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, Rel_list)
                Acc[si] = max(Acc[si], acc)
                Nov[si] = max(Nov[si], nov)
                Div[si] = max(Div[si], div)
                Proximity[si] = max(Proximity[si], Pro)
                Coverage[si] = max(Coverage[si], Cov)
                Quantity[si] = max(Quantity[si], Qua)
                Volatility[si] = max(Volatility[si], Vol)

        print('删除评价指标列表中的0元素之前：')
        print('Acc_max=',max(Acc),'Acc_mean=', statistics.mean(Acc), 'Acc_std=', statistics.stdev(Acc), 'Acc_min=',min(Acc))
        print('Nov_max=',max(Nov),'Nov_mean=', statistics.mean(Nov), 'Nov_std=', statistics.stdev(Nov), 'Nov_min=',min(Nov))
        print('Div_max=',max(Div),'Div_mean=', statistics.mean(Div), 'Div_std=', statistics.stdev(Div), 'Div_min=',min(Div))
        print('Proximity_max=', max(Proximity), 'Proximity_mean=', statistics.mean(Proximity), 'Proximity_std=', statistics.stdev(Proximity), 'Proximity_min=',min(Proximity))
        print('Coverage_max=', max(Coverage), 'Coverage_mean=', statistics.mean(Coverage), 'Coverage_std=',statistics.stdev(Coverage), 'Coverage_min=', min(Coverage))
        print('Quantity_max=', max(Quantity), 'Quantity_mean=', statistics.mean(Quantity), 'Quantity_std=',statistics.stdev(Quantity), 'Quantity_min=', min(Quantity))
        print('Volatility_max=', max(Volatility), 'Volatility_mean=', statistics.mean(Volatility), 'Volatility_std=',statistics.stdev(Volatility), 'Volatility_min=', min(Volatility))
        Acc, Nov, Div = [x for x in Acc if x > 0], [x for x in Nov if x > 0], [x for x in Div if x > 0]
        Proximity, Coverage, Quantity, Volatility = [x for x in Proximity if x > 0], [x for x in Coverage if x > 0], [x for x in Quantity if x > 0], [x for x in Volatility if x > 0]
        print('删除评价指标列表中的0元素之后：')
        print('Acc_max=',max(Acc),'Acc_mean=', statistics.mean(Acc), 'Acc_std=', statistics.stdev(Acc), 'Acc_min=',min(Acc))
        print('Nov_max=',max(Nov),'Nov_mean=', statistics.mean(Nov), 'Nov_std=', statistics.stdev(Nov), 'Nov_min=',min(Nov))
        print('Div_max=',max(Div),'Div_mean=', statistics.mean(Div), 'Div_std=', statistics.stdev(Div), 'Div_min=',min(Div))
        print('Proximity_max=', max(Proximity), 'Proximity_mean=', statistics.mean(Proximity), 'Proximity_std=',statistics.stdev(Proximity), 'Proximity_min=', min(Proximity))
        print('Coverage_max=', max(Coverage), 'Coverage_mean=', statistics.mean(Coverage), 'Coverage_std=',statistics.stdev(Coverage), 'Coverage_min=', min(Coverage))
        print('Quantity_max=', max(Quantity), 'Quantity_mean=', statistics.mean(Quantity), 'Quantity_std=',statistics.stdev(Quantity), 'Quantity_min=', min(Quantity))
        print('Volatility_max=', max(Volatility), 'Volatility_mean=', statistics.mean(Volatility), 'Volatility_std=',statistics.stdev(Volatility), 'Volatility_min=', min(Volatility))
    def forward(self): # 针对每一个学习者, 进行遗传算法迭代, 返回Rel[si]
        Rel = [[] for i in range(self.S_num)]
        for si in range(self.S_num):
            if self.E_num[si] <= 0: # CE候选习题列表为0，可以把该学生忽略
                continue
            if si >= self.S_num // 10: # 对于学习者较多的数据，可以提前终止
                break
            population = self.Init_Population(si) # 生成初始种群
            for age in range(self.generation_num):
                print(age, end='  ')
                population = self.Crossover(si, population)
                # population = self.Mutation(si, population)
                population = self.Selection(si, population)
            print('针对第',si,'个学习者推荐的习题列表为:')
            for c in range(self.Rec_num):
                ls = []
                chromosome = population[c]
                for k in range(len(chromosome)): # 枚举ont-hot向量
                    if chromosome[k] == 0:       # 第k个习题不选
                        continue
                    ls.append(self.CE[si][k])
                print(ls)
                Rel[si].append(ls)
        return Rel # 针对每个学习者的推荐列表 Rel[si] 不是01编码
    def Init_Population(self, si):
        # population = list(np.random.randint(0, 1, (self.population_size, self.E_num[si])))  # 生成popsize×chromosome_length的二维0、1序列
        population = [[random.randint(0, 1) for i in range(self.E_num[si])] for j in range(self.population_size)]
        return population

    def Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(self, si, Rel_list):
# 给定一个习题推荐列表，计算si的准确性Accuracy指标、新颖性、多样性
        Acc, Nov, Div, Proximity, Coverage, Quantity, Volatility = 0, 0, 0, 0, 0, 0, 0

        WKC_tmp = self.WKC
        for ej in Rel_list:
            Acc += abs(self.ds[si] - self.de[ej])  # 1 - abs(self.delta - self.de[ej])

            Rel_knowledge = set()

            for ck in range(self.C_num):
                if self.Q[ej][ck] == 0:
                    continue
                Rel_knowledge.add(ck)
                WKC_tmp[si][ck] = 0
            intersection = len(Rel_knowledge.intersection(self.all_correct_knowledge[si]))
            union = len(Rel_knowledge.union(self.all_correct_knowledge[si]))
            Nov += ((intersection / float(union)) if union != 0 else 0)  # 计算Jaccard相似度

            for ei in Rel_list:
                if ei == ej:
                    continue
                Div += self.Sim(self.Q[ei], self.Q[ej])

            Proximity += self.de[ej]

        for j in range(len(Rel_list) - 1):
            Volatility += abs(self.de[Rel_list[j]] - self.de[Rel_list[j + 1]])

        Acc = Acc / len(Rel_list) if len(Rel_list) > 0 else 0
        Nov = Nov / len(Rel_list) if len(Rel_list) > 0 else 0
        Div = Div / (len(Rel_list) * (len(Rel_list) - 1)) if len(Rel_list) > 1 else 0
        Proximity = abs((Proximity / len(Rel_list) if len(Rel_list) > 0 else 0) - self.ds[si])
        Coverage = sum(WKC_tmp[si]) / sum(self.WKC[si]) if sum(self.WKC[si]) > 0 else 0
        Quantity = len(Rel_list) / sum(self.CE[si]) if sum(self.CE[si]) > 0 else 0
        Volatility = Volatility / len(Rel_list) if len(Rel_list) > 0 else 0

        return (1-Acc), (1-Nov), (1-Div), Proximity, Coverage, Quantity, Volatility

    def Fitness(self, si, population):
        ans = [[0, 0, 0] for i in range(len(population))]   # Acc, Nov, Div
# 将染色体chromosome转化为习题推荐列表，然后计算三个指标，用sum来表示适应度函数，越大越好
        for c in range(self.population_size):
            chromosome = population[c]
            Rel_list = []
            for i in range(len(chromosome)):
                if chromosome[i] == 0:
                    continue
                ei = self.CE[si][i]
                Rel_list.append(ei)

            ans[c][0], ans[c][1], ans[c][2], ans[c][3], ans[c][4], ans[c][5], ans[c][6] = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, Rel_list) # Acc, Nov, Div
        fitness = [sum(ans[c]) for c in range(len(population))]
        # fitness = [ans[c][2] for c in range(len(population))]
        return fitness
    def Selection(self, si, population): # 精英保留策略
        # 根据fitness 进行倒序排序, 优先选择数值较小的, 最终数目self.population_size
        fitness = self.Fitness(si, population)
        print([round(x, 3) for x in fitness])
        sorted_pairs = sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)
        sorted_fitness, sorted_population = zip(*sorted_pairs) # 解压排序后的列表，得到按fitness降序排列的fitness和population
        sorted_population = list(sorted_population)
        offsprings = sorted_population[:self.population_size]
        return offsprings
    def Crossover(self, si, population):
        # 随机选择两个chromosome,
        offsprings = population
        cnt = 0
        num = len(population)
        while cnt <= (int)(self.E_num[si] * self.new_offsprings_num):
            # t = random.random()
            # if t > self.pc:
            #     continue
            c1, c2 = random.randint(0, num - 1), random.randint(0, num - 1) # c1,c2枚举的是选择那两个染色体
            partial = random.randint(0, self.E_num[si]-1)
            offsprings.append(population[c1][:partial] + population[c2][partial:])
            cnt += 1
        return offsprings
    def Mutation(self, si, population):
        offsprings = population
        cnt = 0
        num = len(population)
        while cnt <= (int)(self.E_num[si] * self.new_offsprings_num):
            # t = random.random()
            # if t > self.pm:
            #     continue
            c = random.randint(0, num - 1)
            partial = random.randint(0, self.E_num[si] - 1)
            chromosome = population[c]
            chromosome[partial] ^= 1
            offsprings.append(chromosome)
            cnt += 1
        return offsprings
    def Sim(self, qi, qj):
        a = norm(qi)
        b = norm(qj)
        return dot(qi, qj) / (a * b) if a * b != 0 else 0