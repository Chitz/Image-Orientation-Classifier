from __future__ import division

__author__ = 'sagabhan, ctewani'

from collections import Counter, defaultdict
from Queue import PriorityQueue
from data_cache import Cache
from random import sample
import knn_kmeans
from numpy import array, linalg

class Node:
    def __init__(self, dist, centeriodIndex):
        self.dist = dist
        self.centeriodIndex = centeriodIndex

    def __lt__(self, other):
        return self.dist < other.dist

def kmeans(l, k, isNumpy):

    # randomly initialize clusters
    cluster_centeriods = [x[2] for x in sample(Cache.train, l)]
    iterations = 0
    while iterations < 10:
        cluster_nodes_acc = defaultdict(list)
        cluster_nodes = defaultdict(list)
        cluster_node_count = Counter()

        if isNumpy:
            cluster_centeriods_array = [array(x) for x in cluster_centeriods]

        # initialize cluster_nodes to zeros
        for i in range(len(cluster_centeriods)):
            cluster_nodes[i] = [0] * len(cluster_centeriods[0])

        # assign each sample in training set
        for train in Cache.train:
            NodePQ = PriorityQueue()

            if isNumpy:
                trainArray = array(train[2])

            for centeriodIndex in range(len(cluster_centeriods)):

                # manhatten - Numpy
                if isNumpy:
                    totalDist = sum(abs(trainArray - cluster_centeriods_array[centeriodIndex]))
                else:
                    # manhatten - Tradition
                    totalDist = 0
                    for index in range(len(train[2])):
                        diff = train[2][index] - cluster_centeriods[centeriodIndex][index]
                        if diff >= 0:
                            totalDist += diff
                        else:
                            totalDist -= diff

                '''
                # euclidean - Numpy
                if isNumpy:
                    totalDist = linalg.norm(trainArray - cluster_centeriods_array[centeriodIndex])
                else:
                    # euclidean - Traditional
                    totalDist = 0
                    for index in range(len(train[2])):
                        totalDist += pow((train[2][index] - cluster_centeriods[centeriodIndex][index]), 2)
                    totalDist = pow(totalDist, 1 / 2)
                '''

                node = Node(totalDist, centeriodIndex)
                NodePQ.put(node)

            node = NodePQ.get()
            for vectorIndex in range(len(cluster_nodes[node.centeriodIndex])):
                cluster_nodes[node.centeriodIndex][vectorIndex] += train[2][vectorIndex]

            cluster_nodes_acc[node.centeriodIndex].append(train)
            cluster_node_count[node.centeriodIndex] += 1

        # mean of cluster
        cluster_centeriods = []
        for i in range(len(cluster_nodes)):
            cluster_nodes[i] = [cluster_nodes[i][j]/cluster_node_count[i] for j in range(len(cluster_nodes[i])) if cluster_node_count[i] != 0]
            cluster_centeriods.append(cluster_nodes[i])

        #print cluster_node_count
        print "iterations", iterations

        iterations += 1

    '''
    with open('knn_features.txt', 'w') as f:
        f.write(str(cluster_node_count))
        f.write('\n\n')
        f.write(str(cluster_node_count))
        f.write('\n\n')
        f.write(str(cluster_nodes_acc))
    '''

    return knn_kmeans.knn_means(k, cluster_centeriods, cluster_nodes_acc, isNumpy)