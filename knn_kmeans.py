from __future__ import division

__author__ = 'sagabhan, ctewani'

from collections import Counter
from Queue import PriorityQueue
from data_cache import Cache
from numpy import array, linalg, sum, abs

class Node:
    def __init__(self, dist, orientation):
        self.dist = dist
        self.orientation = orientation

    def __lt__(self, other):
        return self.dist < other.dist


def find_majority(k, results):
    resultDict = Counter()
    for i in range(k):
        resultDict[results[i]] += 1

    #print results,
    most_common = []
    best = 0
    for orient in resultDict:
        if best < resultDict[orient]:
            most_common = []
            best = resultDict[orient]
            most_common.append(orient)
        elif best == resultDict[orient]:
            most_common.append(orient)

    return most_common


def knn_means(k, cluster_centeriods, cluster_nodes_acc, isNumpy):
    k_param = k
    results = []
    if isNumpy:
        cluster_centeriods_array = [array(x) for x in cluster_centeriods]
    count = 0
    for test in Cache.test:
        k = k_param
        NodePQR = PriorityQueue()
        result = []

        if isNumpy:
            testArray = array(test[2])

        for centeriodIndex in range(len(cluster_centeriods)):


            # Manhatten - Numpy
            if isNumpy:
                totalDist = sum(abs(testArray - cluster_centeriods_array[centeriodIndex]))
            else:
                # manhatten - Tradition
                totalDist = 0
                for index in range(len(test[2])):
                    diff = test[2][index] - cluster_centeriods[centeriodIndex][index]
                    if diff >= 0:
                        totalDist += diff
                    else:
                        totalDist -= diff

            '''
            # Euclidean - Numpy
            if isNumpy:
                totalDist = linalg.norm(testArray - cluster_centeriods_array[centeriodIndex])
            else:
                # Euclidean - Traditional
                totalDist = 0
                for index in range(len(test[2])):
                    totalDist += pow((abs(cluster_centeriods[centeriodIndex][index] - test[2][index])), 2)

                totalDist = pow(totalDist, 1 / 2)
            '''

            node = Node(totalDist, centeriodIndex)
            NodePQR.put(node)

        NodePQ = PriorityQueue()

        for i in range(3):
            node = NodePQR.get()

            for train in cluster_nodes_acc[node.orientation]:

                if isNumpy:
                    trainArray = array(train[2])


                # Manhatten - Numpy
                if isNumpy:
                    totalDist = sum(abs(testArray - testArray))
                else:
                    # manhatten - Tradition
                    totalDist = 0
                    for index in range(len(test[2])):
                        diff = train[2][index] - test[2][index]
                        if diff >= 0:
                            totalDist += diff
                        else:
                            totalDist -= diff

                '''
                # Euclidean - Numpy
                if isNumpy:
                    totalDist = linalg.norm(testArray - testArray)
                else:
                     # Euclidean - Traditional
                    totalDist = 0
                    for index in range(len(test[2])):
                        totalDist += pow((test[2][index] - train[2][index]), 2)

                    totalDist = pow(totalDist, 1 / 2)
                '''

                node = Node(totalDist, train[1])
                NodePQ.put(node)

        for i in range(k):
            node = NodePQ.get()
            result.append(node.orientation)

        resultClass = find_majority(k, result)

        while len(resultClass) != 1:
            # decrease k by 1
            print k,
            k -= 1
            resultClass = find_majority(k, result)

        #print resultClass
        results.append(resultClass[0])
        count += 1
        

    return results
