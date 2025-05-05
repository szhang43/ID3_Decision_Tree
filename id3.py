#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
import math 
from collections import Counter
# Node class for the decision tree
import node

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p

def entropy(p):
    if p == 0 or p == 1: 
        return 0 
    return -p * math.log2(p) - (1 - p) * math.log2(1 -p)

print(entropy(0.5))  # Should return 1.0
print(entropy(1.0))  # Should return 0.0

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data

def infogain(py_pxi, pxi, py, total):
    p_y = py / total 
    p_x = pxi / total 
    p_not_x = 1 - p_x

    if pxi == 0:
        p_y_given_x = 0
    else:
        p_y_given_x = py_pxi / pxi

    if total - pxi == 0:
        p_y_given_not_x = 0
    else:
        p_y_given_not_x = (py - py_pxi) / (total - pxi)
    
    h_y = entropy(p_y)
    h_y_given_x = entropy(p_y_given_x)
    h_y_given_not_x = entropy(p_y_given_not_x)

    h_y_given_xi = p_x * h_y_given_x + p_not_x * h_y_given_not_x

    return h_y - h_y_given_xi

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

def isPure(data): 
    labelIndex = -1 
    labels = [row[labelIndex] for row in data]
    return all(label == labels[0] for label in labels)

def majorityClass(data): 
    labelIndex = -1 
    labels = [row[labelIndex] for row in data]
    return Counter(labels).most_common(1)[0][0]

def splitData(data, attributeIndex):
    left = [row for row in data if row[attributeIndex] == 0]
    right = [row for row in data if row[attributeIndex] == 1]
    return left, right 

def bestAttribute(data): 
    total = len(data)
    py = sum(row[-1] for row in data)

    bestIG = -1 
    bestAttribute = None 

    for i in range(len(data[0]) - 1):
        pxi = sum(row[i] for row in data)
        py_pxi = sum(1 for row in data if row[i] == 1 and row[-1] == 1)

        ig = infogain(py_pxi, pxi, py, total)

        if ig > bestIG: 
            bestIG = ig
            bestAttribute = i 
    return bestAttribute

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    if isPure(data): 
        return node.Leaf(varnames, data[0][-1])
    
    if len(data[0]) <= 1: 
        return node.Lead(varnames, majorityClass(data))
    bestAttrSplit = bestAttribute(data)

    if bestAttrSplit is None: 
        return node.Leaf(varnames, majorityClass(data))
    
    leftData, rightData = splitData(data, bestAttrSplit)

    if not leftData or not rightData:
        return node.Leaf(varnames, majorityClass(data))
    
    leftSubTree = build_tree(leftData, varnames)
    rightSubTree = build_tree(rightData, varnames)

    return node.Split(varnames, bestAttrSplit, leftSubTree, rightSubTree)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
