import numpy as np
from hmmlearn import hmm
import torch.nn as nn
import time
from hmmlearn.base import _BaseHMM as myhmm
from torch.autograd import Variable
import torch

total_epochs = 20

epsilon = 1e-3

blocks = []
with open('/Users/Salar/Downloads/Train_Arabic_Digit.txt') as f:
    current_block = []
    for line in f:
        if line.strip():
            z = line.split(' ')
            mfcc = [ float(x) for x in z]
            mfcc = np.array(mfcc, dtype=np.float32)
            current_block.append(mfcc)
        else:
            current_block = np.array(current_block, dtype=np.float32)
            blocks.append(current_block)
            current_block = []
    blocks.append(current_block)

def compute_delta(input):
    delta = np.zeros(13, dtype=np.float32)
    for i in range(13):
        if i + 1 < 13:
            c_t_1 = input[i + 1]
        else:
            c_t_1 = 0
        if i + 2 < 13:
            c_t_2 = input[i + 2]
        else:
            c_t_2 = 0
        if i - 1 > -1:
            c_t_m_1 = input[i -1]
        else:
            c_t_m_1 = 0
        if i - 2 > -1:
            c_t_m_2 = input[i - 2]
        else:
            c_t_m_2 = 0
        delta[i] = (c_t_1 + 2 * c_t_2 - c_t_m_1 - 2 * c_t_m_2) / 10 + epsilon




    return delta




del blocks[0]

for i, block in enumerate(blocks):
    delta = np.apply_along_axis(compute_delta, 1, block)
    delta_delta = np.apply_along_axis(compute_delta, 1 , delta)
    blocks[i] = np.concatenate((blocks[i], delta), axis=1)
    blocks[i] = np.concatenate((blocks[i], delta_delta), axis=1)


labeled_blocks = []


for i in range(0, 6600, 660):
    labeled_blocks.append(blocks[i:i + 660])


start = time.time()


models = []
inputable = []
inputable_lengths = []
for i in range(10):
    input_X = labeled_blocks[i][0]
    lengths = [len(labeled_blocks[i][0])]
    for k, item in enumerate(labeled_blocks[i]):
        if k == 0:
            continue
        input_X = np.concatenate((input_X, item))
        lengths.append(len(item))
    print('model {} started'.format(i))
    model = hmm.GMMHMM(n_components= 12, n_mix= 4 , covariance_type='diag', n_iter=1000)
    print('model {} created'.format(i))
    model.fit(input_X, lengths)
    print('model {} trained'.format(i))
    models.append(model)
    inputable.append(input_X)
    inputable_lengths.append(lengths)

classes_states = [[] for _ in range(10)]
for i in range(10):
    for j, data in enumerate(labeled_blocks[i]):
        _, a = models[i].decode(data)
        classes_states[i].append(a)

mlp = nn.Sequential(
    nn.Linear(13, 5),
    nn.Tanh(),
    nn.Linear(5,7),
    nn.Tanh(),
    nn.Linear(7,12),
    nn.Softmax()
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr = 1e-4)

hmm_model = [[] for i in range(10)]
fit_model = [[] for i in range(10)]

for i in range(10):
    for epoch in range(total_epochs):
        for k, item in enumerate(labeled_blocks[i]):
            for j,data in enumerate(item):
                state_hot = np.zeros(12)
                state_hot[classes_states[i][k][j]] = 1
                t_data = Variable(torch.FloatTensor(data))
                t_state_hot = Variable(torch.FloatTensor(state_hot))
                optimizer.zero_grad()
                out = mlp(t_data)
                loss = criterion(out, t_state_hot)
                loss.backward()
                optimizer.step()
    train_data = torch.FloatTensor(inputable[i])
    train_data_t = Variable(train_data)
    out = mlp(train_data_t)
    hmm_model[i] = hmm.GaussianHMM(n_components=12)
    train_size = np.sum(inputable_lengths[i])
    a = torch.sum(out, dim=0) / train_size
    hmm_model[i].emissionprob_ = a
    fit_model[i] = hmm_model[i].fit(inputable[i], inputable_lengths[i])


train_time = time.time() - start

print('training took {}'.format(train_time))

blocks = []
with open('/Users/ali/Downloads/Test_Arabic_Digit.txt') as f:
    current_block = []
    for line in f:
        if line.strip():
            z = line.split(' ')
            mfcc = [ float(x) for x in z]
            mfcc = np.array(mfcc, dtype=np.float32)
            current_block.append(mfcc)
        else:
            current_block = np.array(current_block, dtype=np.float32)
            blocks.append(current_block)
            current_block = []
    blocks.append(current_block)



del blocks[0]

for i, block in enumerate(blocks):
    delta = np.apply_along_axis(compute_delta, 1, block)
    delta_delta = np.apply_along_axis(compute_delta, 1 , delta)
    blocks[i] = np.concatenate((blocks[i], delta), axis=1)
    blocks[i] = np.concatenate((blocks[i], delta_delta), axis=1)


labeled_blocks = []


for i in range(0, 2200, 220):
    labeled_blocks.append(blocks[i:i + 220])


start = time.time()


scores = np.zeros(10)
pred_label = np.zeros((10, 220))
for i in range(10):
    for j, data in enumerate(labeled_blocks[i]):
        for q in range(10):
            scores[q] = (fit_model[q]).score(data)

        label = np.argmax(scores)
        pred_label[i,j] = label

correct_test = 0
for i in range(10):
    for j, data in enumerate(labeled_blocks[i]):
        if(pred_label[i,j] == i):
            correct_test += 1

precision = correct_test / 220

print("test took {}".format(time.time()- start))
print('test accuracy is {}'.format(precision))










