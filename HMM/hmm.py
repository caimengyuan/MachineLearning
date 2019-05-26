import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box 3"]        #状态
n_states = len(states)

observations = ["red", "white"]     #观测值
n_observation = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])       #初始状态概率向量

transition_probability = np.array([         #状态转移概率矩阵A
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

emission_probability = np.array([           #观测概率矩阵B
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)    #n_components是假设的隐含状态数量
model.startprob_ = start_probability    #startprob_参数对应我们的隐藏状态初始分布
model.transmat_ = transition_probability        #该参数对应状态转移矩阵A
model.emissionprob_ = emission_probability      #该参数对应观测状态概率矩阵B

# if __name__ == '__main__':
#     seen = np.array([[0,1,0]]).T
#     logprob, box = model.decode(seen, algorithm="viterbi")      #利用维特比算法
#     print("The ball picked:", ", ".join(map(lambda x: str(np.array(observations)[x]), seen)))
#     print("The hidden box", ", ".join(map(lambda x: str(np.array(states)[x]), box)))

# if __name__ == '__main__':
#     seen = np.array([[0, 1, 0]]).T
#     box2 = model.predict(seen)
#     print("The ball picked:", ", ".join(map(lambda x: str(np.array(observations)[x]), seen)))
#     print("The hidden box", ", ".join(map(lambda x: str(np.array(states)[x]), box2)))

#问题一 Evaluation
# if __name__ == '__main__':
#     seen = np.array([[0,1,0]]).T
#     print(model.score(seen))        #score函数返回的是以自然对数为底的对数概率值

#问题二 Learning
# model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
# X2 = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]])
# model2.fit(X2)
# print(model2.startprob_)
# print(model2.transmat_)
# print(model2.emissionprob_)
# print(model2.score(X2))
# model2.fit(X2)
# print(model2.startprob_)
# print(model2.transmat_)
# print(model2.emissionprob_)
# print(model2.score(X2))
# model2.fit(X2)
# print(model2.startprob_)
# print(model2.transmat_)
# print(model2.emissionprob_)
# print(model2.score(X2))

'''
实例2
'''
startprob = np.array([0.6, 0.3, 0.1, 0.0])
#The transition matrix, note that there are no transitions possible
#between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
#The means of each component
means = np.array([[0.0, 0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])
#The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1))

#Build an HMM instance and set parameters
model3 = hmm.GaussianHMM(n_components=4, covariance_type="full")    #covariance_type 取值为"full"意味着所有的μ,Σ都需要指定
#取值为“spherical”则Σ的非对角线元素为0，对角线元素相同。取值为“diag”则的非对角线元素为0，对角线元素可以不同，"tied"指所有的隐藏状态对应的观测状态分布使用相同的协方差矩阵Σ

#Instead of fitting it from the data, we directly set the estimated
#parameters, the means and covariance of components
model3.startprob_ = startprob
model3.transmat_ = transmat
model3.means_ = means
model3.covars_ = covars

if __name__ == '__main__':
    seen = np.array([[1.1, 2.0], [-1, 2.0], [3, 7]])
    logprob, state = model3.decode(seen, algorithm="viterbi")
    print(state)