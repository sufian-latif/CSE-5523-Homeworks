from collections import namedtuple
import sys
import math
from Data import *

DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'

def Entropy(data):
    if len(data) == 0: return 0.0
    nPos = sum(1 for d in data if d[0] == POS_CLASS)
    nNeg = len(data) - nPos
    f = lambda a, b: 0.0 if a == 0 else -1.0 * a / b * (math.log(a, 2) - math.log(b, 2))
    return f(nPos, len(data)) + f(nNeg, len(data))

def Split(data, f):
    return [d for d in data if d[f.feature] == f.value], [d for d in data if d[f.feature] != f.value]

def InformationGain(data, f):
    if len(data) == 0: return 0.0
    tData, fData = Split(data, f)
    return Entropy(data) - (len(tData) * Entropy(tData) + len(fData) * Entropy(fData)) / len(data)

def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" % (prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')

def ID3(data, features, MIN_GAIN=0.1):
    maxGain = -1
    splitFeature = None
    for f in features:
        gain = InformationGain(data, f)
        if gain > maxGain:
            maxGain = gain
            splitFeature = f

    if not maxGain > MIN_GAIN:
        nPos = [d[0] for d in data].count(POS_CLASS)
        return DtNode(splitFeature, (nPos, len(data) - nPos), maxGain, None, None)

    lData, rData = Split(data, splitFeature)
    newFeatures = features - set([splitFeature])
    return DtNode(splitFeature, (len(lData), len(rData)), maxGain,
                  ID3(lData, newFeatures, MIN_GAIN), ID3(rData, newFeatures, MIN_GAIN))

if __name__ == "__main__":
    train = MushroomData(sys.argv[1])
    dev = MushroomData(sys.argv[2])

    dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))

    PrintTree(dTree)

    print Accuracy(dTree, dev.data)
