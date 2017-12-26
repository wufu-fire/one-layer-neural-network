import numpy as np

class Perceptron(object):
    '''
    eta: 学习率
    n_iter: 权重向量的训练次数
    w_: 神经分叉权重向量
    errors: 用于记录神经元判断出错次数
    '''
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, Y):
        '''
        :param X: 输入样本向量
        :param Y: 对应的样本分类
        :return:
        '''

        # 初始化权重向量，增加一个w0
        self.w_ = np.zeros(1+X.shape[1])
        # errors记录错误情况
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                '''
                更新公式
                '''
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        pass
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass
    def predict(self, X):
        return np.where(self.net_input(X) >=0, 1, -1)
        pass
