# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# plt 设置正常显示中文与负号
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示中文标签时的负号

trainNum = 501
xTrain = np.linspace(100, 200, trainNum).reshape([1, -1]) #1*501个数据
np.random.shuffle(xTrain)
yTrain = 500 * np.sin(xTrain) + 20 * np.random.normal(0, 2, xTrain.shape)

xMaxNum = np.max(xTrain)
xMinNum = np.min(xTrain)
yMaxNum = np.max(yTrain)
yMinNum = np.min(yTrain)

# 归一化
xTrain = ((xTrain - xMinNum) / (xMaxNum - xMinNum))
yTrain = ((yTrain - yMinNum) / (yMaxNum - yMinNum))

#plt.clf() #清空plt画板
#plt.plot(xTrain[0], yTrain[0], 'ro', label = u'训练数据') #将训练数据用圆点的形式显示
#plt.legend() #显示图片中的图例说明，如点代表数据，蓝线代表拟合曲线
#plt.savefig('curve_data.png', dpi=200) #保存图片

 ##### 激活函数的定义 #####           
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            
##### 神经网络结构的定义 #####
learning_rate = 1
hiddenDim = 4
W1 = np.random.randn(hiddenDim, 1) * 0.01
b1 = np.zeros((hiddenDim, 1))
W2 = np.random.randn(1, hiddenDim) * 0.01
b2 = np.zeros((1, 1))

def compute_cost(y, yTrain, choice='CrossEntropy', m = trainNum):
#    if 'SquareError':
#        return 0
    if 'CrossEntropy':
        logprobs = np.multiply(np.log(y),yTrain) + np.log(1-y)*(1-yTrain)
        cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)                 
    return cost

def sigmoid_ds(x):
    return x * (1-x)

def backward_propagation(a1, y, m = trainNum):
    dZ2 = y - yTrain
    dW2 = 1/m * np.dot(dZ2, a1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(a1, 2))
    dW1 = 1/m * np.dot(dZ1, xTrain.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, db1, dW2, db2

def forward_propagation(W1, b1, W2, b2):
    a0 = xTrain
    a1 = sigmoid( np.dot(W1, a0) + b1)
    a2 = sigmoid( np.dot(W2, a1) + b2 )
    y = a2
    return a1, y

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

costLast = 0
##### 训练过程 #####
for i in range(100001):
    a1, y = forward_propagation(W1, b1, W2, b2)
    cost = compute_cost(y, yTrain)
    dW1, db1, dW2, db2 = backward_propagation(a1, y)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2)
    
    if i % 1000 == 0:
        print("第" + str(i) + "次训练，cost=" + str(cost) )   
        plt.clf()
        plt.plot(xTrain[0], yTrain[0], 'mo', label=u'测试数据') #画点
        plt.plot(xTrain[0], y[0], label = u'拟合曲线') #画线
        plt.legend()
        plt.savefig('curve_fitting_test.png', dpi=200)
        plt.show() #将曲线学习结果用图片直观的显示出来
#        if(abs(cost - costLast) < 0.000001):
#            break
#        costLast = cost
 
plt.clf()
plt.plot(xTrain[0], yTrain[0], 'mo', label=u'测试数据') #画点
plt.plot(xTrain[0], y[0], label = u'拟合曲线') #画线
plt.legend()
plt.savefig('curve_fitting_test.png', dpi=200)
plt.show() #将曲线学习结果用图片直观的显示出来