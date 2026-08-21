import queue
# python实现线性分配问题 or 二分图最大匹配的模板算法, 但是不适用于一般图
from scipy.optimize import linear_sum_assignment

'''
优先队列选择部落 算法
具体为定义一个Alliance类，插入到优先队列PriorityQueue()中。
在每一次执行联姻过程时，需调用联姻部落选择算法生成C(C(n,2), n / 2)个组合，且需要更新Edge_Weight，重新插入一轮。
'''
class Alliance(object):
    def __init__(self, Id, u, v, Historical_Marriage_Iterations, Distance_Last_Marriage_Iterations):
    # Id是边的编号, u v是节点的编号, Historical_Marriage_Iterations是(u,v)历史联姻的总次数, Distance_Last_Marriage_Iterations是(u,v)距离上次联姻间隔的代数
        self.Id = Id
        self.u = u; self.v = v
        self.Historical_Marriage_Iterations = Historical_Marriage_Iterations
        self.Distance_Last_Marriage_Iterations = Distance_Last_Marriage_Iterations

    def __lt__(self, other): # 自定义比较关系, 先比较历史联姻次数, 越小的越先出队列先去联姻, 然后再去比较距离上次联姻间隔的代数, 越大的先联姻(需要初始化为无穷小, 且要及时记录更新)
        if self.Historical_Marriage_Iterations == other.Historical_Marriage_Iterations:
            return self.Distance_Last_Marriage_Iterations > other.Distance_Last_Marriage_Iterations
        return self.Historical_Marriage_Iterations < other.Historical_Marriage_Iterations

n = 5
Edge_Weight = [[-1, -1, 0, 0] for i in range((int)(n*(n-1)/2))] # 初始化
cnt = 0
vis = [False for i in range(n+1)]
for i in range(1, n + 1):
    for j in range(i + 1, n + 1):
        Edge_Weight[cnt] = [i, j, i, j]
        cnt += 1
que = queue.PriorityQueue()
for num in range((int)(n*(n-1)/2)):
    [x, y, u, v] = Edge_Weight[num]
    que.put(Alliance(Id = num, u = u, v = v, Historical_Marriage_Iterations=x, Distance_Last_Marriage_Iterations=y))

while que.empty() == False:
    Top = que.get() # 提取后就会出队列
    u = Top.u; v = Top.v; num = Top.Id
    if vis[u] == True or vis[v] == True:
        continue
    print("联姻(%d, %d)"%(u, v))
    vis[u] = vis[v] = True
    Edge_Weight[num][2] += 1 # 为下一次联姻选择做准备
    Edge_Weight[num][3] = 0
    # print(Top.Id, Top.u, Top.v, Top.Historical_Marriage_Iterations, Top.Distance_Last_Marriage_Iterations)
    # print(que.qsize())
