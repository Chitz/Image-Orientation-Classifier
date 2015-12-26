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


def knn(k, isNumpy):
    print "### K-Nearest Neighbour"
    results = []
    for test in Cache.test:
        NodePQ = PriorityQueue()
        result = []

        if isNumpy:
            testArray = array(test[2])

        for train in Cache.train:

            if isNumpy:
                trainArray = array(train[2])

            # Manhatten - Numpy
            if isNumpy:
                totalDist = sum(abs(testArray - testArray))
            else:
                # manhatten - Traditional
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
            k -= 1
            resultClass = find_majority(k, result)

        results.append(resultClass[0])
        #del NodePQ

    return results
