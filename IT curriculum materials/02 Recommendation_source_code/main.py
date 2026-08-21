import random
import numpy as np
import time
import pandas as pd
import ER_GA, ER_TGA

'''
可能是习题集的问题，变异反而可能把效果变差
再怎么交叉，候选习题集已经弄好了，也白搭了
'''

import sys
# sys.path.append("../")
import Process_Data

class My_model():
    def __init__(self, Epsilon, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001, pa=0.3, Rec_num=5, N=5):
        self.Epsilon = Epsilon # 弱知识点覆盖因子
        self.delta = delta     # 难度系数
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.pa = pa    # 联姻概率
        self.new_offsprings_num = new_offsprings_num  # 交叉 变异需新增的个体比例
        self.Rec_num = Rec_num # 最后给每一个学习者推荐习题列表时有Rec_num个选择列表
        self.N = N # 种群的个数
    def forward(self, name):

        my_DataSet = Process_Data.MyDataset(self.Epsilon, self.delta, self.population_size, self.generation_num, self.new_offsprings_num,
                                               self.pc, self.pm, self.Rec_num)
        if name == "Frcsub":
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.Frcsub()
        elif name == 'Math1':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.Math1()
        elif name == 'Math2':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.Math2()
        elif name == 'ASSISTments2009':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.ASSISTments2009()
        elif name == 'ASSISTments2017':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.ASSISTments2017()
        elif name == 'Statics2011':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.Statics2011()
        elif name == 'Algebra2005':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE = my_DataSet.Algebra2005()
        GA = ER_TGA.Genetic_Algorithm(CE=self.CE, S_num=self.S_num, C_num=self.C_num, X=self.X, Q=self.Q,
                                                 de=self.de, ds=self.ds, WKC=self.WKC,
                                                 delta=self.delta, population_size=self.population_size,
                                                 generation_num=self.generation_num,
                                                 new_offsprings_num=self.new_offsprings_num, pc=self.pc, pm=self.pm, pa = self.pa,
                                                 Rec_num=self.Rec_num, N=self.N)
        # GA = GA4ER.Genetic_Algorithm(CE=self.CE, S_num=self.S_num, C_num=self.C_num, X=self.X, Q=self.Q,
        #                                de=self.de, ds=self.ds, WKC=self.WKC,
        #                                delta=self.delta, population_size=self.population_size,
        #                                generation_num=self.generation_num,
        #                                new_offsprings_num=self.new_offsprings_num, pc=self.pc, pm=self.pm,
        #                                Rec_num=self.Rec_num)

        GA.evaluate()

stime = time.time()
name = 'Math1' #sys.argv[1]
# generation_num = int(sys.argv[2])
# N = int(sys.argv[3]) # 种群的个数，必为偶数
# print('name=', name, 'Age=',generation_num, 'N=',N)
model = My_model(Epsilon=0.35, delta=0.5, population_size=30,
                   generation_num=30, new_offsprings_num=0.3, pc=0.6, pm=0.0001, pa=0.3, Rec_num=2, N=6)

model.forward(name)
print('The running time is',time.time() - stime)