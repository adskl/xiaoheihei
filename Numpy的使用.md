# python的Numpy使用
	from numpy import *
	random.rand(4,4) 生成一个4*4的矩阵
	randMat=mat(random.rand(4,4))  生成一个矩阵
	randMat.I 生成一个矩阵的逆运算
	invRandMat=randMat.I
	randMat*invRantMat  矩阵相乘
	myEye=randMat*invRandMat
	myEye-eye(4)  得到误差值
## k临近算法
### 准备：导入python数据
	from numpy import *
	import operator  导入运算符模块，排序使用的
	def createrDateSet():
	   group = arrary{[[1.0,1.0],[1.0,1.1],[0,0],[0,0.1]]}
	   label = ["A","A","B","B"]
	   return group,labels
### k-近邻算法
	def classify(inX,dataSet,labels,k):
	  dataSetSize = dataSet.shape{0}
	  diffMat = tile(inX,(dataSetSize,1))-dataSet
函数argsort返回的是一个排序后原来的数据的位置  
函数strip()是删除字符串的特定的东西
### 归一化处理
	newvalue = oldvalue-min/max-min
### 总结：
其实这个测试集就是输入的，然后暂存在里面，然后输入一个测试的数据，就求得最小的距离，找最好的几个，然后找到最优的，最近的输出测试集的那个标签即可。并没有所谓的测试集，只是暂存在里面而已。