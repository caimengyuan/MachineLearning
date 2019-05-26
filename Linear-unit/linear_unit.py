from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights:\t:%s\nbias:\t:%f\n' %(self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,..]和weights[w1,w2...]打包在一起
        # 变成[(x1,w1),(x2,w2)...]
        # 然后利用map函数计算[x1*w1, x2*w2,...]
        # 最后利用reduce求和
        return self.activator(reduce(lambda a, b: a+b,
                                     map(lambda x_w: x_w[0] * x_w[1], zip(input_vec, self.weights))
                                     , 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, lables, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label),...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, lables)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1, w2, w3...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3)...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = list(map(lambda x_w: x_w[1] + rate * delta * x_w[0], zip(input_vec, self.weights)))
        # 更新bias
        self.bias += rate * delta

# 定义激活函数f
f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''
        初始化线性单元，设置输入参数的个数
        '''
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练，迭代10轮，学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu

if __name__ == '__main__':
    '''
    训练线性单元
    '''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))