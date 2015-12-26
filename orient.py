from __future__ import division

__author__ = 'sagabhan, ctewani'

from sys import argv, exit
from time import time
import knn
from nnet import Nnet
from best import Best
from data_cache import Cache
import k_means
from collections import Counter, defaultdict


class Score:
    def printConfusionMatrix(self):
        for labelX in self.labels:
            for labelY in self.labels:
                print self.confusionMatrix[labelX][labelY],
            print

    def score(self, algo, ground_truth, algo_outputs):
        correct = 0
        # print algo_outputs
        op = open(algo + '_ouput.txt', 'w')
        self.labels = ['0', '90', '180', '270']
        self.confusionMatrix = defaultdict(Counter)  # confusionMatrix[x][y] => ground truth x, predicted y => count
        for i in range(len(algo_outputs)):
            if int(algo_outputs[i]) == int(ground_truth[i][1]):
                correct += 1
            self.confusionMatrix[ground_truth[i][1]][algo_outputs[i]] += 1
            op.write(Cache.test[i][0] + ' ' + algo_outputs[i] + '\n')

        op.close()
        print round((correct / len(ground_truth) * 100), 2)
        self.printConfusionMatrix()


class Solver:
    def solve(self, algo, algo_param):
        if algo == "knn":
            #return knn.knn(int(algo_param))
            return k_means.kmeans(8, int(algo_param), False)
        elif algo == "nnet":
            nnet = Nnet(int(algo_param));
            return nnet.classify()
        elif algo == "best":
            best = Best(algo_param)
            return best.classify()


def main():
    if len(argv) != 5:
        print "Usage: python orient.py training_file test_file algo algo_param"
        exit()

    (train_file, test_file, algo, algo_param) = argv[1:]
    # load train and test data
    Cache.read_data(train_file, False)
    Cache.read_data(test_file, True)

    # start classifying
    startTime = time()
    solver = Solver()
    algo_output = solver.solve(algo, algo_param)

    # to read direct from output file
    # algo_output = custom_result()

    endTime = time()

    score = Score()
    score.score(algo, Cache.test, algo_output)
    print "Time taken", endTime - startTime


if __name__ == '__main__':
    main()
