#!/usr/bin/python

#########################################################
# CSE 5523 starter code (HW#5)
# Alan Ritter
#########################################################

import random
import math
import sys
import re
import matplotlib.pyplot as plt

# GLOBALS/Constants
VAR_INIT = 1

def logExpSum(x):
    return max(x) + math.log(sum(math.exp(xi - max(x)) for xi in x))

def readTrue(filename='wine-true.data'):
    f = open(filename)
    labels = []
    splitRe = re.compile(r"\s")
    for line in f:
        labels.append(int(splitRe.split(line)[0]))
    return labels

#########################################################################
# Reads and manages data in appropriate format
#########################################################################
class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows, self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    # Computers the range of each column (returns a list of min-max tuples)
    def Range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1, self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min, max))
        return ranges

    def __getitem__(self, row):
        return self.data[row]

#########################################################################
# Computes EM on a given data set, using the specified number of clusters
# self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################
class EM:
    def __init__(self, data, nClusters):
        # Initialize parameters randomly...
        random.seed()
        self.parameters = []
        self.priors = []        # Cluster priors
        self.nClusters = nClusters
        self.data = data
        ranges = data.Range()
        for i in range(nClusters):
            p = []
            initRow = random.randint(0, data.nRows-1)
            for j in range(data.nCols):
                # Randomly initialize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p)

        # Initialize priors uniformly
        for c in range(nClusters):
            self.priors.append(1/float(nClusters))

    def LogLikelihood(self, data):
        return sum(logExpSum([math.log(self.priors[c]) + self.LogProb(i, c, data) for c in range(self.nClusters)])
                   for i in range(data.nRows))

    # Compute marginal distributions of hidden variables
    def Estep(self):
        p = []
        for row in range(self.data.nRows):
            pi = [self.priors[c] * math.exp(self.LogProb(row, c, self.data)) for c in range(self.nClusters)]
            tmp = sum(pi)
            p.append([x / tmp for x in pi])

        return p

    # Update the parameter estimates
    def Mstep(self, p):
        for c in range(self.nClusters):
            pr = [p[i][c] for i in range(self.data.nRows)]
            self.priors[c] = sum(pr) / self.data.nRows
            for col in range(self.data.nCols):
                x = [self.data[i][col] for i in range(self.data.nRows)]
                mean = self.parameters[c][col][0]
                newMean = sum(pr[i] * x[i] for i in range(self.data.nRows)) / sum(pr)
                newVar = sum(pr[i] * (x[i] - mean) ** 2 for i in range(self.data.nRows)) / sum(pr)
                self.parameters[c][col] = (newMean, newVar)

    # Computes the probability that row was generated by cluster
    def LogProb(self, row, cluster, data):
        param = self.parameters[cluster]
        x = data[row]
        return -0.5 * sum(math.log(param[i][1]) + (x[i] - param[i][0]) ** 2 / param[i][1] for i in range(data.nCols))

    def Run(self, maxsteps=100, testData=None):
        trainLikelihood = [self.LogLikelihood(self.data)]
        testLikelihood = [self.LogLikelihood(testData)]if testData is not None else None

        diff = 1000
        while math.fabs(diff / trainLikelihood[-1]) > 0.001:
            self.Mstep(self.Estep())
            diff = trainLikelihood[-1]
            trainLikelihood.append(self.LogLikelihood(self.data))
            if testData != None:
                testLikelihood.append(self.LogLikelihood(testData))
            diff = trainLikelihood[-1] - diff

        return trainLikelihood, testLikelihood

    def getPrediction(self):
        pr = self.Estep()
        labels = []
        for p in pr:
            maxP = 0
            for i in range(len(p)):
                if p[i] > p[maxP]:
                    maxP = i
            labels.append(maxP + 1)
        return labels

def identifyClusters(labels, pred):
    info = dict((k, dict((k, 0) for k in set(pred))) for k in set(labels))
    for i in range(len(labels)):
        info[labels[i]][pred[i]] += 1
    return info

def accuracy(info):
    correct = 0.0
    total = 0.0
    for i in info:
        for j in info[i]:
            total += info[i][j]
        correct += max(info[i][k] for k in info[i])

    return correct / total

if __name__ == "__main__":
    d = Data('wine.train')
    td = Data('wine.test')
    labels = readTrue('wine-true.data')

    # (a)
    while True:
        try:
            em = EM(d, 3)
            trainLikelihood, testLikelihood = em.Run(100, td)
            break
        except:
            pass
    pred = em.getPrediction()
    plt.title('Iterations vs log likelihood for training and test set')
    plt.xlabel('No. of iterations')
    plt.ylabel('Log Likelihood')
    plt.xticks(range(len(trainLikelihood)))
    plt.plot(range(len(trainLikelihood)), trainLikelihood, 'bo', ms = 3, ls = '-', lw = 1, label = 'Training set')
    plt.plot(range(len(testLikelihood)), testLikelihood, 'gs', ms = 3, ls = '-', lw = 1, label = 'Test set')
    plt.legend()
    plt.show()

    # (b)
    details = []
    for i in range(10):
        while True:
            try:
                em = EM(d, 3)
                trainLikelihood, testLikelihood = em.Run(100, td)
                details.append([trainLikelihood[0], trainLikelihood[-1], len(trainLikelihood),
                                testLikelihood[0], testLikelihood[-1], len(testLikelihood)])
                break
            except:
                pass

    for row in details:
        print row

    # (c)
    em = EM(d, 3)
    trainLikelihood, testLikelihood = em.Run(100)
    pred = identifyClusters(labels, em.getPrediction())
    print pred

    # (d)
    ll = [[], []]
    for nc in range(1, 11):
        while True:
            try:
                em = EM(d, nc)
                trainLikelihood, testLikelihood = em.Run(100, td)
                ll[0].append(trainLikelihood[-1])
                ll[1].append(testLikelihood[-1])
                break
            except:
                pass
    plt.title('Final log likelihood vs no. of clusters: ')
    plt.xlabel('No. of clusters')
    plt.ylabel('Final log Likelihood')
    plt.xticks(range(1 + len(ll[0])))
    plt.plot(range(1, 1 + len(ll[0])), ll[0], 'bo', ms = 3, ls = '-', lw = 1, label = 'Training set')
    plt.plot(range(1, 1 + len(ll[1])), ll[1], 'gs', ms = 3, ls = '-', lw = 1, label = 'Test set')
    plt.legend()
    plt.show()
