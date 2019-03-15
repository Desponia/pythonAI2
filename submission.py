# -*- coding: utf-8 -*-

import random
import collections
import math
import time
from util import *

_X_ = None


############################################################
# Sentiment Classification
############################################################

# Problem B: extractWordFeatures

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE
    # raise NotImplementedError
    phi = collections.defaultdict(int)  # or "phi = collections.defaultdict(int)"

    for word in x.split():
        if word not in phi:
            phi[word] = 0
        phi[word] += 1
    # END_YOUR_CODE
    return phi


# Problem C: learnPredictor

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned, and error values lists for train and test datasets

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    # BEGIN_YOUR_CODE
    # raise NotImplementedError
    weights = {}
    trainErrorList = []
    testErrorList = []

    def predictor(x):
        phi = featureExtractor(x)
        if dotProduct(phi, weights) >= 0:
            y = 1
        else:
            y = -1
        return y

    for t in range(numIters):
        for trainExample in trainExamples:
            # print('trainExample : ', trainExample)
            x, y = trainExample
            # print(y)
            phi = featureExtractor(x)

            # calculate Loss value
            loss = max(0, 1 - dotProduct(weights, phi) * y)  # use 'max' and 'dotProduct'

            # print('loss :', loss)

            # update the weight vector
            if loss > 0:
                increment(weights, eta * y, phi)  # 'increment' is defined in util.py

        train_error = evaluatePredictor(trainExamples, predictor)
        test_error = evaluatePredictor(testExamples, predictor)
        print("%d-th iteration: train error = %.2f, test error = %.2f" % \
              (t, train_error, test_error))

        trainErrorList.append(train_error)
        testErrorList.append(test_error)

    # END_YOUR_CODE

    return weights, trainErrorList, testErrorList


# Problem F: extractBigramFeatures Problem
def extractBigramFeatures(x):
    """
    Extract unigram(word) and bigram features for a string x.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_CODE
    # raise NotImplementedError
    phi = extractWordFeatures(x)
    words = ('<s>',) + tuple(x.split()) + ('</s>',)

    for idx in range(len(words) - 1):
        feature = words[idx: idx + 2]
        phi[feature] = phi.get(feature, 0) + 1

    # END_YOUR_CODE
    return phi


############################################################
# k-means Clustering
############################################################

# Problem K: kmeans

def kmeans(examples, K, maxIters):
    '''
    example : dic
    examples: list of dic
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE
    # raise NotImplementedError
    centers = random.sample(examples, K)
    assignments = [0] * len(examples)

    def get_l2_loss(center_id, example): # 센터와 example feature(vectoor) 사이의 거리
        center = centers[center_id]
        features = set(list(center.keys()) + list(example.keys()))

        return sum((center.get(feature, 0) - example.get(feature, 0)) ** 2 for feature in features)  # use 'dict.get'

    for iter_cnt in range(maxIters):
        prev_assignments = assignments
        assignments = [0] * len(examples)

        # update assignments
        for example_id, example in enumerate(examples):
            assignments[example_id] = min((center_id for center_id in range(len(centers))),
                                          key=lambda x: get_l2_loss(x, example))  # use 'get_l2_loss'
        # early stopping
        if prev_assignments == assignments:
            print('early stopping k-means')
            break

        # update centers
        cluster_sizes = [0] * K
        cluster_sums = [{} for _ in range(K)]
        for example_id, example in enumerate(examples):
            cluster_id = assignments[example_id]  # use assignments
            cluster_sizes[cluster_id] += 1
            cluster_sum = cluster_sums[cluster_id]
            increment(cluster_sum, 1., example)

        centers = [{feature: value_sum / cluster_size for feature, value_sum in cluster_sum.items()} for cluster_sum, cluster_size in zip(cluster_sums, cluster_sizes)]

    else:
        print('max iteration')

    loss = 0
    for example_id, example in enumerate(examples):
        loss += get_l2_loss(assignments[example_id], example)

    # END_YOUR_CODE

    return centers, assignments, loss


# Problem M: kmeans_optimized

def kmeans_optimized(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE
    # raise NotImplementedError

    centers = random.sample(examples, K)
    assignments = [0] * len(examples)

    # 모든 뮤의 제곱을 저장해서 제곱한거 리턴... ||뮤||2
    def get_squared_norms():
        return [sum(value ** 2 for value in center.values()) for center in centers]

    center_squared_norms = get_squared_norms()

    def get_l2_loss(center_id, example):  # 센터와 example feature(vectoor) 사이의 거리
        l2_loss = center_squared_norms[center_id]
        center = centers[center_id]
        for key, value in example.items():
            center_value = center.get(key, 0)
            l2_loss += value ** 2 - 2* value * center_value

        return l2_loss

    for iter_cnt in range(maxIters):
        prev_assignments = assignments
        assignments = [0] * len(examples)

        # update assignments
        for example_id, example in enumerate(examples):
            assignments[example_id] = min((center_id for center_id in range(len(centers))),
                                          key=lambda x: get_l2_loss(x, example))  # use 'get_l2_loss'
        # early stopping
        if prev_assignments == assignments:
            print('early stopping k-means')
            break

        # update centers
        cluster_sizes = [0] * K
        cluster_sums = [{} for _ in range(K)]
        for example_id, example in enumerate(examples):
            cluster_id = assignments[example_id]  # use assignments
            cluster_sizes[cluster_id] += 1
            cluster_sum = cluster_sums[cluster_id]
            increment(cluster_sum, 1., example)

        centers = [{feature: value_sum / cluster_size for feature, value_sum in cluster_sum.items()} for
                   cluster_sum, cluster_size in zip(cluster_sums, cluster_sizes)]

        center_squared_norms = get_squared_norms()

    else:
        print('max iteration')

    loss = 0
    for example_id, example in enumerate(examples):
        loss += get_l2_loss(assignments[example_id], example)


    # END_YOUR_CODE
    return centers, assignments, loss
