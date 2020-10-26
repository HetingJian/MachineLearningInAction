from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

from Code.mnist.img2Vector import *


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # Construct an array by repeating A the number of times given by reps.
    # 因此，若输入一个二元组，第一个数表示复制的行数，第二个数表示对inx的重复的次数
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


'''
对未知类别属性的数据集中的每个点依次执行以下操作:
(1)计算已知类别数据集中的点与当前点之间的距离；
(2)按照距离递增次序排序
(3)选取与当前点距离最小的k个点
(4)确定前k个点所在类别的出现概率
(5)返回前k个点出现频率最高的类别作为当前点的预测分类
'''


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split(' ')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


'''
下面的公式可以将任意取值范围的特征值转换为0到1区间的值
newValue = (oldValue - min)/(max - min)
'''


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(Ranges , (m,1))
    return normDataSet, ranges, minVals


'''
特征值矩阵有1000*3个值，而minVals和range的值都为1*3
使用Numpy库中tile()函数将变量内容复制成输入矩阵同样大小的矩阵
'''

if __name__ == '__main__':
    group, labels = createDataSet()
    # print(group, labels)
    # print(classify0([0.1, 0], group, labels, 1))

    returnMat, nb = img2Vector('1.txt')
    # print(returnMat)
    # print(nb)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:, 1], returnMat[:, 2])
    plt.show()
