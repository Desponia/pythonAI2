import collections
import os

if os.name == 'posix' and 'DISPLAY' not in os.environ:  # check display availability
    # https://stackoverflow.com/questions/8257385/automatic-detection-of-display-availability-with-matplotlib
    # https://stackoverflow.com/a/7174725
    import matplotlib

    matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
# import matplotlib.figure as figure
import numpy as np

from submission import *
from util import *


def main():
    polarity_train_path = os.path.join('data', 'polarity.train')
    polarity_dev_path = os.path.join('data', 'polarity.dev')

    try:
        print('\n========== Problem B ==========')
        print('\n---------- Test b0 ----------')
        output = extractWordFeatures("a b a")
        answer = {"a": 2, "b": 1}
        print("output : %s" % output)
        print("answer : %s" % answer)

        print('\n---------- Test b1 ----------')
        random.seed(42)
        for i in range(10):
            sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
        output = extractWordFeatures(sentence)
        print("output : %s" % output)
        print("answer : %s" % "{'aa': 14, 'a': 16, 'c': 28, 'b': 23, 'ab': 19}")

        print('\n========== Problem C ==========')
        print('\n---------- Test c0 ----------')
        trainExamples = (("pretty good", 1), ("bad plot", -1), ("not good", -1), ("pretty scenery", 1))
        testExamples = (("pretty", 1), ("bad", -1))
        weights, _, _ = learnPredictor(trainExamples, testExamples, extractWordFeatures, numIters=1, eta=1)
        print(weights)

        print('\n---------- Test c1 ----------')
        trainExamples = (("hello world", 1), ("goodnight moon", -1))
        testExamples = (("hello", 1), ("moon", -1))
        weights, _, _ = learnPredictor(trainExamples, testExamples, extractWordFeatures, numIters=20, eta=0.01)
        print("weight for `hello\' : %f (should > 0)" % weights['hello'])
        print("weight for `moon\' : %f (should < 0)" % weights['moon'])

        print('\n---------- Test c2 ----------')
        trainExamples = readExamples(polarity_train_path)
        testExamples = readExamples(polarity_dev_path)
        weights, trainErrorsUnigram, testErrorsUnigram = learnPredictor(trainExamples, testExamples,
                                                                        extractWordFeatures, numIters=20, eta=0.01)
        outputWeights(weights, os.path.join('result', 'weights'))
        outputErrorAnalysis(testExamples, extractWordFeatures, weights,
                            os.path.join('result', 'error-analysis'))  # Use this to debug
        trainError = evaluatePredictor(trainExamples,
                                       lambda x: (1 if dotProduct(extractWordFeatures(x), weights) >= 0 else -1))
        testError = evaluatePredictor(testExamples,
                                      lambda x: (1 if dotProduct(extractWordFeatures(x), weights) >= 0 else -1))
        print("Final train error = %s (should < 0.04)" % trainError)
        print("Final test error = %s (should < 0.30)" % testError)

        print('\n========== Problem D ==========')
        visualizePredictorError(trainErrorsUnigram, testErrorsUnigram)

        print('\n========== Problem F ==========')
        print('\n---------- Test f0 ----------')
        trainExamples = readExamples(polarity_train_path)
        testExamples = readExamples(polarity_dev_path)
        weightsBigram, trainErrorsBigram, testErrorsBigram = learnPredictor(trainExamples, testExamples,
                                                                            extractBigramFeatures, numIters=20,
                                                                            eta=0.01)
        outputWeights(weightsBigram, os.path.join('result', 'weights-bigram'))
        outputErrorAnalysis(testExamples, extractBigramFeatures, weightsBigram,
                            os.path.join('result', 'error-analysis-bigram'))  # Use this to debug
        trainErrorBigram = evaluatePredictor(trainExamples, lambda x: (
            1 if dotProduct(extractBigramFeatures(x), weightsBigram) >= 0 else -1))
        testErrorBigram = evaluatePredictor(testExamples, lambda x: (
            1 if dotProduct(extractBigramFeatures(x), weightsBigram) >= 0 else -1))
        print("Final train error = %s (should < 0.04)" % trainErrorBigram)
        print("Final test error = %s (should < 0.27)" % testErrorBigram)

        print('\n========== Problem G ==========')
        visualizeUnigramAndBigram(trainErrorsUnigram, testErrorsUnigram, trainErrorsBigram, testErrorsBigram)

        print('\n========== Problem K ==========')
        print('\n---------- Test k0 ----------')
        random.seed(42)
        x1 = {0: 0, 1: 0}  # x1 => (0, 0)
        x2 = {0: 0, 1: 1}  # x2 => (0, 1)
        x3 = {0: 0, 1: 2}  # x3 => (0, 2)
        x4 = {0: 0, 1: 3}  # x4 => (0, 3)
        x5 = {0: 0, 1: 4}  # x5 => (0, 4)
        x6 = {0: 0, 1: 5}  # x6 => (0, 5)
        examples = [x1, x2, x3, x4, x5, x6]
        centers, assignments, totalCost = kmeans(examples, 2, maxIters=10)
        # (there are two stable centroid locations)
        print(centers)
        print(assignments)
        print("output square loss : %.1f" % totalCost)
        print("answer square loss : 4.0 or 5.5")

        print('\n---------- Test k1 ----------')
        K = 6
        numExamples = 1000
        numFillerWords = 1000
        testClustering(kmeans, K, numExamples, numFillerWords)

        # print ('\n---------- Test k2 ----------')
        # K = 6
        # numExamples = 10000
        # numFillerWords = 10000
        # testClustering(kmeans, K, numExamples, numFillerWords)  # 200 seconds in my PC

        print('\n========== Problem L ==========')
        # x1 = {0:0, 1:0}         # x1 => (0, 0)
        # x2 = {0:0, 1:1}         # x2 => (0, 1)
        # x3 = {0:2, 1:0}         # x3 => (2, 0)
        # x4 = {0:2, 1:1}         # x4 => (2, 1)
        # simple_examples = [x1, x2, x3, x4]
        np.random.seed(10)
        example_array = np.append(np.random.multivariate_normal(mean=(2, 2), cov=((1, 0), (0, 1)), size=30),
                                  np.random.multivariate_normal(mean=(-2, -2), cov=((1, 0), (0, 1)), size=30),
                                  axis=0)
        examples = [{0: ex[0], 1: ex[1]} for ex in example_array]
        centers, assignments, totalCost = kmeans(examples, 2, maxIters=10)

        visualizeClusters(examples, assignments, centers)

        print('\n========== Problem M ==========')
        K = 6
        numExamples = 10000
        numFillerWords = 10000
        testClustering(kmeans_optimized, K, numExamples, numFillerWords)

    except NotImplementedError as err:
        # print err
        print("\nNotImplementedError: you didn't implement the function.")


def visualizePredictorError(trainErrorList, testErrorList):
    '''
    Plot a graph that visualizes changes for error.
    |trainErrorList| and |testErrorList| containts training and test error values for each iteration.
    '''
    plt.plot(list(range(len(trainErrorList))), trainErrorList,
             'bo-', label='Train')
    plt.plot(list(range(len(testErrorList))), testErrorList,
             'ro-', label='Test')
    plt.legend(loc="upper right")
    plt.ylabel('Error')
    plt.xlabel('Itearation')
    plt.title('Sentiment analysis')
    plt.xticks(list(range(1, 20, 2)))
    # plt.savefig('demo.png', bbox_inches='tight')
    plt.show()


def visualizeUnigramAndBigram(trainErrorsUnigram, testErrorsUnigram, trainErrorsBigram, testErrorsBigram):
    plt.plot(list(range(1, 21)), testErrorsUnigram, 'rx-', label='Test (unigram)')
    plt.plot(list(range(1, 21)), testErrorsBigram, 'ms-', label='Test (bigram)')
    plt.plot(list(range(1, 21)), trainErrorsUnigram, 'bo-', label='Train (unigram)')
    plt.plot(list(range(1, 21)), trainErrorsBigram, 'c*-', label='Train (bigram)')
    plt.legend()
    plt.ylabel('Error')
    plt.xticks([1, 5, 10, 15, 20])
    plt.title('Sentiment Analysis')
    plt.show()


def testClustering(kmeans, K, numExamples, numFillerWords):
    examples = generateClusteringExamples(numExamples=numExamples, numWordsPerTopic=3,
                                          numFillerWords=numFillerWords)
    print('clustering on {} examples...'.format(numExamples))
    start = time.time()
    centers, assignments, totalCost = kmeans(examples, K, maxIters=100)
    end = time.time()
    print('time elapsed for clustering: ', end - start, 'seconds')
    print('total cost: ', totalCost)
    outputClusters(os.path.join('result', 'output-cluster'), examples, centers, assignments)


def visualizeClusters(examples, assignments, centers):
    K = len(centers)
    clusters = [[] for _ in range(K)]
    for idx, example in enumerate(examples):
        clusters[assignments[idx]].append(example)

    example_colors = ('r', 'b', 'c', 'y', 'k')  # you can add other colors
    assert K <= len(example_colors)

    fig, ax = plt.subplots()

    for cluster, color in zip(clusters, example_colors):
        ax.scatter(
            x=[example[0] for example in cluster],
            y=[example[1] for example in cluster],
            color=color
        )

    for i in range(len(examples)):
        ax.annotate(
            s='x%d' % (i + 1),
            xy=(examples[i][0], examples[i][1])
        )

    ax.scatter(
        x=[_[0] for _ in centers],
        y=[_[1] for _ in centers],
        color='g'
    )

    for i in range(len(centers)):
        ax.annotate(
            s='$\mu$' + '%d' % (i + 1),
            xy=(centers[i][0], centers[i][1])
        )

    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.show()


if __name__ == '__main__':
    main()
