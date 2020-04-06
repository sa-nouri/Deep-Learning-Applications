import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle

def load_dataset():
    train_data = np.loadtxt("/Users/ali/Downloads/mnist_train.csv", delimiter=',')
    test_data = np.loadtxt("/Users/ali/Downloads/mnist_test.csv", delimiter=',')
    tr_data = np.asarray(train_data[:,1:])
    tr_label = np.asarray(train_data[:,:1]).astype(int)

    ts_data = np.asarray(test_data[:, 1:])
    ts_label = np.asarray(test_data[:, :1]).astype(int)

    return tr_data, tr_label, ts_data, ts_label
def ReLU(x):
    return 0.5 * (x + np.abs(x))



def ReLU_derivative(x):
    y = copy.deepcopy(x)
    y[y <= 0] = 0
    y[y > 0] = 1
    return y

def gausian(x):
    return np.exp(-(x ** 2))

def gausian_derivative(x):
    return -2 * x * gausian(x)



def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return 1 / (1 + np.exp(-x))

def leaky_relu(x):
    return np.maximum(0.2 * x, x)

def leaky_relu_derivative(x):
    y = copy.deepcopy(x)
    y[y<=0] = 0.2
    y[y>0] = 1
    return y

class Neural_Network:
    def __init__(self, no_input, no_hidden_layer, hidden_layer_size, no_output, batch_size):
        self.no_input = no_input
        self.no_hidden_layer = no_hidden_layer
        self.hidden_layer_size = hidden_layer_size
        self.no_output = no_output
        self.std = 0.1
        self.batch_size = batch_size

        self.weights, self.bias = self.initialize()



    def initialize(self):
        weights = list()
        bias = list()
        if (self.no_hidden_layer == 0):
            weights.append(np.random.randn(self.no_input, self.no_output) /np.sqrt(self.no_input/2))
            bias.append(self.std * np.random.randn(self.no_output))
            return weights, bias

        weights.append(np.random.randn(self.no_input, self.hidden_layer_size[0]) / np.sqrt(self.no_input / 2))
        for i in range(0, len(self.hidden_layer_size) - 1):
            weights.append(np.random.randn(self.hidden_layer_size[i], self.hidden_layer_size[i + 1]) / np.sqrt(
                self.hidden_layer_size[i] / 2))
        weights.append(
            np.random.randn(self.hidden_layer_size[-1], self.no_output) / np.sqrt(self.hidden_layer_size[-1] / 2))

        for i in self.hidden_layer_size:
            bias.append(self.std * np.random.randn(i))
        bias.append(self.std * np.random.randn(self.no_output))

        return weights, bias


    def generate_mini_batch(self, data, label, shuffle = True):
        if shuffle:
            indicies = np.arange(data.shape[0])
            np.random.shuffle(indicies)
        for i in range(0, data.shape[0] - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indicies[i: i + self.batch_size]
            else:
                excerpt = slice(i, i + self.batch_size)
            yield data[excerpt], label[excerpt]

    def correct_index(self, scores):
        index = np.zeros([scores.shape[0], scores.shape[1]])
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                temp = copy.deepcopy(scores[i,:])
                temp[j] -= 1
                index[i,j] = np.linalg.norm(temp)

        return index



    def train(self, data, label, epoch,  learning_rate_add, test_data, test_label):
        learning_rate = learning_rate_add
        losses = list()
        percisions = list()
        test_percisions = list()
        for l in range(epoch):
            each_loss = list()
            each_correction = list()
            for batch in self.generate_mini_batch(data, label):
                batch_data, batch_label = batch

                modified_label = list()
                for k in range(self.batch_size):
                    modified_label.append(batch_label[k][0])

                z = list()
                s = list()
                s.append(np.dot(batch_data,self.weights[0]) + self.bias[0])
                z.append(ReLU(np.dot(batch_data,self.weights[0]) + self.bias[0]))


                for j in range(1 , self.no_hidden_layer):
                    s.append(np.dot(z[j - 1], self.weights[j]) + self.bias[j])
                    z.append(ReLU(np.dot(z[j - 1], self.weights[j]) + self.bias[j]))
                if(self.no_hidden_layer > 0):
                    s.append(np.dot(z[-1], self.weights[-1]) + self.bias[-1])
                    z.append(gausian(np.dot(z[-1], self.weights[-1]) + self.bias[-1]))




                # forward finished
                #compute loss

                correct_score = np.zeros(z[-1].shape)
                correct_score[np.arange(self.batch_size), modified_label[:]] = 1
                index = np.argmin(self.correct_index(z[-1]), axis=1)
                each_correction.append(np.sum(1*(modified_label[:] == index)))
                margins = np.multiply( z[-1] - correct_score,z[-1] - correct_score)

                each_loss.append(np.sum(margins) / self.batch_size)



                #loss computed

                #calculate gradient

                delta_output = -2 * (correct_score -  z[-1])

                delta_output = np.multiply(delta_output , gausian_derivative(s[-1]))




                delta_hiddens = list()
                if(self.no_hidden_layer > 0):
                    delta_hiddens.append(np.multiply(np.dot(delta_output, self.weights[-1].T) , ReLU_derivative(s[-2])))
                    for m in range(2 , self.no_hidden_layer + 1):
                        delta_hiddens.append(np.multiply(np.dot(delta_hiddens[-1], self.weights[len(self.weights) - m].T) , ReLU_derivative(s[len(self.weights) - m -1])))
                    dw = np.dot(z[-2].T, delta_output)/self.batch_size

                    db = np.sum(delta_output, axis=0) / self.batch_size
                else:
                    dw = np.dot(batch_data.T, delta_output) / self.batch_size

                    db = np.sum(delta_output, axis=0) / self.batch_size

                self.weights[-1] -= learning_rate * dw
                self.bias[-1] -= learning_rate * db
                for i in range(len(delta_hiddens) - 1):
                    dw = np.dot(z[-i - 3].T, delta_hiddens[i]) / self.batch_size
                    db = np.sum(delta_hiddens[i], axis=0) / self.batch_size
                    self.weights[-i - 2] -= learning_rate * dw
                    self.bias[ -i - 2] -= learning_rate * db
                if(self.no_hidden_layer > 0):
                    dw = np.dot(batch_data.T, delta_hiddens[-1]) / self.batch_size


                    db = np.sum(delta_hiddens[-1], axis=0) / self.batch_size
                    self.weights[0] -= learning_rate * dw
                    self.bias[0] -= learning_rate * db


            print("epoch {} average loss is: {} and percision is : {} and rate is: {}".format(l + 1, np.mean(each_loss), np.mean(each_correction) / self.batch_size, learning_rate))
            losses.append(np.mean(each_loss))
            percisions.append(np.mean(each_correction) / self.batch_size)

            if (l % 20 == 19):
                learning_rate = learning_rate / 10

            test_percisions.append(self.test(test_data, test_label))

        plt.plot(range(1, epoch + 1), losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss during training in epochs')

        plt.show()

        plt.plot(range(1, epoch + 1), percisions)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy during training in epochs')
        plt.show()

        plt.plot(range(1, epoch + 1), test_percisions)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy of test during training in epochs')
        plt.show()


    def test(self, input_data, input_label):
        each_correction = list()
        for batch in self.generate_mini_batch(input_data, input_label):
            batch_data, batch_label = batch
            modified_label = list()
            for k in range(self.batch_size):
                modified_label.append(batch_label[k][0])

            z = list()
            s = list()
            s.append(np.dot(batch_data, self.weights[0]) + self.bias[0])
            z.append(ReLU(np.dot(batch_data, self.weights[0]) + self.bias[0]))

            for j in range(1, self.no_hidden_layer):
                s.append(np.dot(z[j - 1], self.weights[j]) + self.bias[j])
                z.append(ReLU(np.dot(z[j - 1], self.weights[j]) + self.bias[j]))
            if(self.no_hidden_layer > 0):
                s.append(np.dot(z[-1], self.weights[-1]) + self.bias[-1])
                z.append(gausian(np.dot(z[-1], self.weights[-1]) + self.bias[-1]))
            index = np.argmin(self.correct_index(z[-1]), axis=1)
            each_correction.append(np.sum(1 * (modified_label[:] == index)))

        print("in last epoch test percision was:{}".format(np.mean(each_correction)/self.batch_size))
        return np.mean(each_correction)/self.batch_size






x,y,z,t = load_dataset()
print("data set loaded")



fac = 255 * 0.99 + 0.01
x = x/fac
z = z/fac





a = Neural_Network(x.shape[1],1,[200],10,128)
a.train(x,y,100,0.01, z, t)
