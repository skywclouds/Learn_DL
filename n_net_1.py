#author: Henry Wang
#date:   2024-05-16
#A Simple NeuralNetwork
import numpy as np

# 激活函数
def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

# 神经网络类
class NeuralNetwork:
    # 构造函数
    def __init__(self, x, y):  
        self.input = x
        self.hiddenLayerNumber = 0
        self.weights = []
        self.bs = []
        self.zs = []
        self.y = y
        self.output = np.zeros(self.y.shape)
    # 损失函数
    def loss(self):
        l = self.y * np.log(self.output) + (1-self.y) * np.log(1 - self.output)
        m = self.y.shape[1]
        l = -np.sum(l,axis=1) / m
        return l
    # 前向传播
    def feedForward(self):
        tempLayer = self.input
        # 迭代隐藏层
        for i in range(0,self.hiddenLayerNumber):
            tempLayer = np.matmul(self.weights[i],tempLayer) + self.bs[i]
            self.zs.append(tempLayer)
            tempLayer = sigmoid(tempLayer)
        # 计算输出层
        tempLayer = np.matmul(self.weights[-1],tempLayer) + self.bs[-1]
        self.zs.append(tempLayer)
        self.output = sigmoid(tempLayer)
    # 反向传播    
    def backProp(self,alpha):
        A = sigmoid(self.zs[-1])
        m = self.y.shape[1]
        dl_dz = -(self.y - A) / m
        for i in range(self.hiddenLayerNumber,0,-1):
            A_1 = sigmoid(self.zs[i-1])# i-1层的A
            dl_dw = np.matmul(dl_dz,np.transpose(A_1))# 这里是矩阵乘法，不是对应元素相乘)
            dl_db = np.sum(dl_dz,axis=1).reshape(-1,1)
            dz_da_1 = np.transpose(self.weights[i])
            dl_da_1 = np.matmul(dz_da_1,dl_dz)# 这里是矩阵乘法，不是对应元素相乘
            da_1_dz_1 = A_1 * (1-A_1)# 这里是对应元素相乘，不是矩阵乘法
            dl_dz = dl_da_1 * da_1_dz_1# 这里是对应元素相乘，不是矩阵乘法
            self.weights[i] -= alpha * dl_dw
            self.bs[i] -= alpha * dl_db
        dl_dw = np.matmul(dl_dz,np.transpose(self.input))
        dl_db = np.sum(dl_dz,axis=1).reshape(-1,1)
        self.weights[0] -= alpha *  dl_dw
        self.bs[0] -= alpha * dl_db
    # 训练
    def train(self,alpha,epoch):
        for i in range(0,epoch):
            self.feedForward()
            self.backProp(alpha)
            print("epoch "+ str(i+1) + " loss " + str(self.loss()))
    #添加隐藏层
    def addHiddenLayer(self,num):
        col_num = 0
        # 若没有隐藏层，则建立与输入层的连接
        if self.hiddenLayerNumber == 0:
            col_num = self.input.shape[0]
        # 若有隐藏层，则建立与前一个隐藏层的连接
        else:
            col_num = self.weights[-1].shape[0]
        self.weights.append(np.random.rand(num,col_num))
        self.bs += (np.random.rand(num,1),)
        self.hiddenLayerNumber += 1
    def addOutputLayer(self):
        self.weights.append(np.random.rand(self.y.shape[0],self.weights[-1].shape[0]))
        self.bs.append(np.random.rand(self.y.shape[0],1))

X = np.array([[0,2,0,1,2,1,2],
              [0,0,2,1,1,2,2],])
Y = np.array([0,0,0,0,1,1,1]).reshape(1,-1)

nn = NeuralNetwork(X,Y)
nn.addHiddenLayer(10)
nn.addHiddenLayer(10)
nn.addOutputLayer()

nn.train(0.01,100)
  
