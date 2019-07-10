# 数字的识别
## 代码如下
	from numpy import *
	from os import listdir
	import kNN

	#读取文件的函数
	def img2vector(filename):
    	returnVect = zeros((1,1024))
    	fr = open(filename)
   	 	for i in range(32):
    	    lineStr = fr.readline()
    	    for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
            
    return returnVect

	def handwritingClassTest():
  	  hwLabels = []
  	  trainingFileList = listdir("E:/python 书籍/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits")
   	  m = len(trainingFileList)
   	  trainingMat = zeros((m,1024))
      for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]        #找到文件的名字
        classNumStr = int(fileStr.split("_")[0])    #这个是文件的数字
        hwLabels.append(classNumStr)                 #特征向量
        trainingMat[i,:] = img2vector("E:/python 书籍/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits/"+str(filenameStr))
     testFileList = listdir('E:/python 书籍/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits')
     errorCount = 0.0
     mTest = len(testFileList)
     for i in range(mTest):
        filenameStr2 = testFileList[i]
        fileStr2 = filenameStr2.split('.')[0]        #找到文件的名字
        classNumStr2 = int(fileStr2.split("_")[0])    #这个是文件的数字
        vectorUnderTest = img2vector("E:/python 书籍/机器学习实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits/"+str(filenameStr2))
        classifierResult = kNN.classify(vectorUnderTest,trainingMat,hwLabels,3)
        print("测试的输出结果是%d,真正的结果是%d"%(classifierResult,classNumStr2))
        if classifierResult!=classNumStr2: errorCount+=1
    print("错误率是%f"%(errorCount/float(mTest)))
    






