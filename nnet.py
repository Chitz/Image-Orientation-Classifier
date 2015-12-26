from __future__ import division

__author__ = 'sagabhan, ctewani'
import random
import math
from data_cache import Cache


class Neuron:
    def __init__(self, input, output):
        self.input = input  # current input to neuron
        self.output = output  # current input to neuron

    def fire(self):
        self.output = self.sigmoid(self.input)

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    def der_sigmoid(self, value):
        return self.sigmoid(value) * (1 - self.sigmoid(value))

    def get_der_sigmoid(self):
        return self.sigmoid(self.input) * (1 - self.sigmoid(self.input))

    def get_derivative(self):
        return self.get_der_sigmoid()


class Nnet:
    def __init__(self, hidden_nodes_count):
        print "Classifying with Neural Network classifier"
        self.classes = ['0', '90', '180', '270']
        self.expected = [0 for i in range(4)]
        self.learning_rate = 0.1
        self.input_nodes = 192
        self.output_nodes = 4
        self.hidden_nodes_count = int(hidden_nodes_count)
        # total 192 input nodes one for each feature
        self.input = [Neuron(0, 0) for i in range(self.input_nodes)]
        # given number of hidden layer nodes, initialize with random weights
        self.hidden = [Neuron(0, 0) for i in range(self.hidden_nodes_count)]
        # 3 output layer nodes, one per possible standard rotation; initialize with random weights
        self.output = [Neuron(0, 0) for i in range(self.output_nodes)]

        self.w1 = [[random.uniform(-0.01, 0.01) for i in range(self.hidden_nodes_count)] for j in range(
            self.input_nodes)]
        self.w2 = [[random.uniform(-0.01, 0.01) for i in range(self.output_nodes)] for j in
                   range(self.hidden_nodes_count)]

    def forward_propogate(self, image_input):
        # initialize input neurons
        for i in range(self.input_nodes):
            self.input[i].input = self.input[i].output = image_input[i] / 255

        # update hidden layer values
        for i in range(self.hidden_nodes_count):
            total = 0.0
            for j in range(self.input_nodes):
                total += self.input[j].output * self.w1[j][i]

            self.hidden[i].input = total
            self.hidden[i].fire()

        # move to output layer
        for i in range(self.output_nodes):
            total = 0.0
            for j in range(self.hidden_nodes_count):
                total += self.hidden[j].output * self.w2[j][i]

            self.output[i].input = total
            self.output[i].fire()
            if self.output[i].output >= 0.7:
                self.output[i].output = 1.0
            elif self.output[i].output <= 0.4:
                self.output[i].output = 0.0

        return self.output

    def back_propogate_error(self, target_label):
        # back propagate error
        # on_node = self.classes.index(target_label)

        error_delta = [0.0 for i in range(self.output_nodes)]

        # calculate delta for output layer
        expected_output = target_label

        for i in range(self.output_nodes):
            error_delta[i] = (expected_output[i] - self.output[i].output) * self.output[i].get_derivative()

        # error delta for hidden layer
        error_delta_hidden = [0.0 for i in range(self.hidden_nodes_count)]

        for i in range(self.hidden_nodes_count):
            error = 0.0
            for j in range(self.output_nodes):
                error += error_delta[j] * self.w2[i][j]

            error_delta_hidden[i] = error * self.hidden[i].output * self.hidden[i].get_derivative()

        # update weights for hidden -> output layer
        for i in range(self.hidden_nodes_count):
            for j in range(self.output_nodes):
                self.w2[i][j] += error_delta[j] * self.hidden[i].output * self.learning_rate

        # update input -> hidden layer weights
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes_count):
                self.w1[i][j] += error_delta_hidden[j] * self.input[i].output * self.learning_rate

        error = 0.0
        for i in range(self.output_nodes):
            error += math.pow(expected_output[i] - self.output[i].output, 2)

        return math.sqrt(error)

    def test(self):
        prediction = []
        predicted_label = 0
        # test now
        # for image in Cache.test:
        for image in Cache.test:
            # print "Testing for image: " + image[0]
            result = []
            nodes = self.forward_propogate(image[2])  # (image[2])
            self.expected = [0.0 for k in range(4)]
            self.expected[self.classes.index(image[1])] = 1.0
            for i in range(len(self.classes)):
                result.append(nodes[i].output)

            predicted_label = self.classes[result.index(max(result))]
            prediction.append(predicted_label)

        return prediction

    def classify(self):
        # complete the training first
        # for image in Cache.train:
        converge = False
        cnt = 3
        while not converge and cnt > 0:
            print "Training iteration : " + str(cnt)
            cnt -= 1
            random.shuffle(Cache.train)
            error = 0.0
            for i in range(len(Cache.train)):
                # forward propagation
                self.forward_propogate(Cache.train[i][2])
                self.expected = [0.0 for k in range(4)]
                self.expected[self.classes.index(Cache.train[i][1])] = 1.0
                error += self.back_propogate_error(self.expected)
            if error <= 1000:
                print "converged: " + str(cnt)
                converge = True

        print "Training complete, let's test now"

        return self.test()
