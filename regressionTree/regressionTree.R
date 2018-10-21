
# 这是一棵回归树，且变量均为连续型变量
# -*- coding:utf-8 -*- 

regLeaf = function(dataSet){
  # 计算因变量的均值
  mean(dataSet[,ncol(dataSet)])
}


regErr = function(dataSet){
  # 计算因变量的RSS
  # 样本方差*(样本个数-1) = RSS
  var(dataSet[,ncol(dataSet)])*(nrow(dataSet)-1)
}


# === 根据给定的特征以及特征值对数据进行切分 ====
binSplitDataSet = function(dataSet, feature, value){
  # 保持与《machine learning in action》一致的分裂法；
  # 大于某个值别时放在左边，小于时放在右边；
  # 这里目前只考虑数值型变量
  
  # dataSet: 数据集
  # feature: 所选取的特征
  # value: 所选取特征下的特征值
  leftData = dataSet[dataSet[, feature] > value, ]
  rightData = dataSet[dataSet[, feature] <= value, ]
  return(result = list(leftData = leftData, rightData = rightData))
}


chooseBestSplit = function(dataSet, tolS=1, tolN=4){
  # 此函数将遍历所有特征所有取值，从而得到最优特征最优取值
  # tolS: 容许的误差下降值
  # tolN: 切分的样本最小数
  
  # ------ 约束条件一 ------
  # 如果因变量y只有一个类别，则不需作任何分割
  # 因此预测值为因变量下数据的平均值即可,运行结束,退出
  if(length(unique(dataSet[, ncol(data)])) == 1){
    return(list(NULL, regLeaf(dataSet)))
  }
  
  # 得到全数据集的行列数
  m = dim(dataSet)[1]; n = dim(dataSet)[2]
  
  # 根据RSS的变化程度大小来选择最优的分割变量
  S = regErr(dataSet) # 计算未分割前因变量的RSS
  bestS = Inf; bestIndex = 0; bestValue = 0
  
  for (featIndex in 1:(n-1)){ # 遍历所有特征(除掉最后一列y因变量)
    for (splitVal in unique(dataSet[, featIndex])){ # 遍历每一个特征所有取值的唯一值
      # 根据所遍历的特征以及相应的特征值对数据进行切分
      result = binSplitDataSet(dataSet, featIndex, splitVal)
      leftData = result[[1]]; rightData = result[[2]]
      # 如果切分出来的子树上边的样本个数小于给定的阈值时，不满足分裂条件，跳过这次迭代！！！
      # 此处next的用处等价于python中的continue.
      if ((nrow(leftData) < tolN) || (nrow(rightData) < tolN))  next
      
      # 当两个子树的样本数都大于tolN时，分别计算分裂后两颗子树上边的样本点因变量的RSS，然后求和
      newS = regErr(leftData) + regErr(rightData)
      
      # 如果分裂后得到RSS的和比未分裂前的小，
      # 那么确认此次分裂是可行的.
      # 输出此次分裂所选取的分裂变量，分裂值，以及分裂后得到更新的RSS。
      # 如果分裂后得到的RSS的和比未分裂前的大，那么不更新分裂变量，分裂值及RSS,
      # 继续遍历其余特征及特征值。
      # 最终输出最优特征，最优特征值，最优RSS.
      if (newS < bestS){ 
        bestIndex = featIndex
        bestValue = splitVal
        bestS = newS
      }
    }
  }
  
  # ---- 约束条件二 ----
  # 如果最优分割下的RSS减少值小于给定的阈值，
  # 此次分割作废,运行结束，退出
  # 继续返回 预测值为所有值的平均值
  if ((S - bestS) < tolS) {
    return(list(NULL, regLeaf(dataSet)))
  }
  # 在最优分割变量下对数据进行重新分割
  result = binSplitDataSet(dataSet, bestIndex, bestValue)
  leftData = result[[1]]; rightData = result[[2]]
  
  # ---- 约束条件三 -----
  # 为稳妥起见，如果最优RSS分割下的子树个数小于阈值，退出
  # 感觉这个条件没有也没有太大关系
  if ((nrow(leftData) < tolN) || (nrow(rightData) < tolN)){
    return(list(NULL, regLeaf(dataSet)))
  }
  # 返回最终最优分割点，最优分割预测值
  return(list(bestIndex = bestIndex, bestValue = bestValue))
}


createTree = function(dataSet, tolS=1, tolN=4){
  retTree <- list() # 创建树的根结点
  # 得到分裂的最优分割值及分割变量
  bestSplit = chooseBestSplit(dataSet, tolS, tolN)
  feature = bestSplit[[1]]; value = bestSplit[[2]]
  # 判断分裂特征值是否存在，不存在的话返回全数据因变量均值
  if (is.null(feature)) return(value)
  retTree$spInd = feature # 将分裂特征赋予给根结点及往后的子树
  retTree$spVal = value # 将分裂特征选中的分裂特征值赋予给根结点及往后的子树
  # 在最优分割特征及最优分割特征值下对数据进行切分
  result = binSplitDataSet(dataSet, feature, value)
  lSet = result[[1]]; rSet = result[[2]]
  # 递归形成大树
  retTree$left = createTree(lSet, tolS, tolN)
  retTree$right = createTree(rSet, tolS, tolN)
  return(retTree)
}


# ===== 对'ex00.txt'数据进行切分 ======
myData = read.table("/Users/chenshicong/Downloads/book/MLinAction/machinelearninginaction3x-master/Ch09/ex00.txt")
# myData = read.table(".../ex00.txt")
# 树结果解读.
# 一开始选择第一列变量作为分裂变量，分裂值为0.50794;
# 分裂之后，左子树或者右子树击中了"chooseBestSplit"中三个中断条件，
# 分离结束，分别输出左子树以及右子树下的目标值的均值.
createTree(myData)

# 将tolN改为200 
createTree(myData, tolS = 0, tolN = 200)



# ====== 对'ex0.txt'数据进行切分 =====

# myData1 = read.table(".../ex0.txt")
myData1 = read.table("/Users/chenshicong/Downloads/book/MLinAction/machinelearninginaction3x-master/Ch09/ex0.txt")
head(myData1)
createTree(myData1)


# ====== 对'ex2.txt'数据进行切分 =====
# 这棵树就很复杂了.

myData2 = read.table("/Users/chenshicong/Downloads/book/MLinAction/machinelearninginaction3x-master/Ch09/ex2.txt")
# myData2 = read.table(".../ex2.txt")
head(myData2)
createTree(myData2)

# 改变参数,将tolN改为20
createTree(myData2, tolS = 6, tolN = 24)













