#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
1：程序的主要内容是训练一个60000张图片的手写数字，测试集有10000张，一次性训练60000张图片，速度慢，所以分成batch来训练，每个batch有128个数据，
batch 的个数为60000/mini_batch_size,向下取整，剩下的数据单独为一个batch，其中进行了洗牌操作。

2：网络的结构是：输入->LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> softmax.softmax输出函数用于多分类，输出每个类别的概率值，我们选
取概率值最高的为相应的分类。因为输出用了softmax，所以代价函数我们选用交叉熵，即cost=-（ylog（A3）），其中A3为输出值

3：dropout的思想就是：在隐藏层的每个激活函数使用dropout，生成一个维度和每层激活函数输出（A1,A2）相同的随机数组，将随机数组与keep_pro比较，小于
keep_pro的为1，大于的为0，为1的被保留，为0的被关闭，最后还要除以keep_pro,保持关闭神经元也没有太大影响。在正向传播使用了dropout，反向传播也
使用dropout，在dA1和dA2之后使用，所用的数组在正向传播中已被存入cache。

4：在测试集上测试时不需要使用dropout，测试一万张的准确率，输出训练集上60000张图片和测试集上10000张图片的准确率

"""

import numpy as np
import matplotlib.pyplot as plt
import load_mnist
#以上是导入所需要的库

def sigmoid(x):
    '定义sigmoid激活函数'
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    '定义relu激活函数'
    s = np.maximum(0, x)
    return s

def softmax(x):
    '定义softmax激活函数'
    A=np.exp(x) / np.sum(np.exp(x),axis=0, keepdims=True)
    return A

def initialize_parameters(layer_dims):
    """初始化参数W1,b1；W2,b2；W3，b3"""
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):#对权值和偏置随机初始化，w使用高斯分布初始化，b使用零初始化
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def forward_propagation(X, parameters):
    '不加dropout的前向传播，用于测试集'
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> softmax
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

def forward_propagation_with_dropout(X, parameters, keep_prob):
    '带有dropout的前向传播用于训练集'
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    #网络结构 LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> softmax
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    #对A1使用dropout,
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) #生成一个和激活函数输出A1维数一样的0-1之间的随机数
    D1 = D1 < keep_prob       #对于D1中得每个数，如果小于keep_prob，则该数为1，否则为零，为1则保留，为零则关闭神经元
    A1 = np.multiply(A1, D1)
    A1 = A1 / keep_prob        #A1/keep_prob的目的是保持关闭一些神经元对结果没有太大影响

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    #对A2使用dropout
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = np.multiply(A2, D2)
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    'dropout用于反向传播'
    '''反向传播公式：
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    '''

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    #对dA2用dropout,因为正向传播用了dropout，所以反向传播也用dropout，对dA2和dA1使用
    dA2 = np.multiply(dA2, D2)
    dA2 = dA2 / keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    #对dA1用dropout
    dA1 = np.multiply(dA1, D1)
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def update_parameters(parameters, grads, learning_rate):
    '更新权值'

    n = len(parameters) // 2

    for k in range(n):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters

def compute_cost(A3, Y):
    '计算cost'
    m = Y.shape[1]

    #对cost使用交叉熵，即cost=-（y*log（输出值））
    cost = -(np.multiply(Y,np.log(A3))).sum()/ m

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

def predict(X, Y, parameters):
    '预测准确率'
    m = X.shape[1]

    probas, caches =forward_propagation(X, parameters)
    m=X.shape[1]
    probas_reverse=probas.T
    Y_reverse=Y.T
    result=0
    for i in range(m):
        temp1=np.argmax(probas_reverse[i])
        temp2=np.argmax(Y_reverse[i])
        if temp1==temp2:
            result=result+1
    accuracy=result/m
    print("正确率：%f" % accuracy)

def model(X, Y, learning_rate, num_iterations, print_cost, lambd, keep_prob):
    '模型'
    grads = {}
    costs = []
    layers_dims = [X.shape[0], 1024,500,10]

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        if keep_prob == 1:      #keep_pro=1相当于没有加dropout
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        cost = compute_cost(a3, Y)
        grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

if __name__ == "__main__":
    from time import clock
    tic = clock()
    
    X_train, Y_train, X_test, Y_test = load_mnist.load_dataset_tf()
    #X_train, Y_train, X_test, Y_test = load_mnist.load_dataset()
    parameters = model(X_train, Y_train, learning_rate = 0.01, num_iterations=5, print_cost=True, lambd=0, keep_prob=0.5)
    predict(X_train,Y_train,parameters)   #训练集准确率
    predict(X_test,Y_test,parameters)      #测试集准确率
    
    toc = clock()
    print("Time:" + str(toc-tic) + "ms")    