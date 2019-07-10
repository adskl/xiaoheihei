# 实现相亲网数据的人员分类
## 代码如下
	import operator
	from numpy import *
## 数据初始化函数
	def knn():
	  group = array([[1,1],[1,0],[0,1],[0,0]])
      label = ["A","C","B","D"]
      return group,label

## k-临近算法函数
	def classify(inX,dataSet,label,k):
     dataSetSize = dataSet.shape[0]         #找到数据的个数
     diffMat = tile(inX,(dataSetSize,1))-dataSet    #做差记住tile函数的作用
     sqDiffMat = diffMat**2
     sqDsitance = sqDiffMat.sum(axis=1)
     distance =sqDsitance**0.5
     sortedDistance = distance.argsort()
     calssCount = {}
     for i in range(k):
        votellable = label[sortedDistance[i]]
        calssCount[votellable] = calssCount.get(votellable,0)+1
     sortedclassCount = sorted(calssCount.items(),
     key=operator.itemgetter(1),reverse=True)         #排序

     return sortedclassCount[0][0]

## 读取文件函数
	def file2matrix(filename):
    	fr = open(filename)
    	arrayOlines = fr.readlines()
    	numberOlines = len(arrayOlines)
    	returnMat = zeros((numberOlines,3))     #输出的特征的矩阵
    	classLabelVector = []                        #输出的是预测值
    	index = 0
    	for line in arrayOlines:
       	 	line = line.strip()      #删除空格
        	listFromLine = line.split('\t')
        	returnMat[index,:] = listFromLine[0:3]
        	if listFromLine[-1]=="largeDoses":
            	classLabelVector.append(3)
        	if listFromLine[-1]=="smallDoses":
            	classLabelVector.append(2) 
        	if listFromLine[-1]=="didntLike":
            	classLabelVector.append(1)
        	index+=1
    	return returnMat,classLabelVector

## 归一化函数
	def autoNorm(dataSet):
     minVals = dataSet.min(0)      #0代表从每列中选取最小值
     maxVals = dataSet.max(0)
     ranges = maxVals-minVals
     normDataSet = zeros(shape(dataSet))
     m = dataSet.shape[0]
     normDataSet = dataSet-tile(minVals,(m,1))
     normDataSet = normDataSet/tile(ranges,(m,1))
     return normDataSet,ranges,minVals

## 测试测试集的效果
	def datingClassTest():
     hoRatio = 0.1
     datingDataMat,datingLabels = file2matrix("E:/python 书籍/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/datingTestSet.txt")
     norMat,ranges,minVals = autoNorm(datingDataMat)
     m = norMat.shape[0]
     numTestVecs = int(m*hoRatio)
     errorCount = 0.0

     for i in range(numTestVecs):
         classfierResult = classify(norMat[i,:],norMat[numTestVecs:m,:],\
            datingLabels[numTestVecs:m],4)
         print("测试返回的是%d,真实的数据是%d"%(classfierResult,datingLabels[i]))
         if classfierResult!=datingLabels[i]:
             errorCount += 1.0
     print("错误率为%f"%(errorCount/float(numTestVecs)))