

# 基于回归下的xgbosot

regLeaf = function(dataSet){
  # 计算每个叶子取值
  # 计算因变量的均值
  mean(dataSet[,ncol(dataSet) - 1])
}


# ===== 巨大变化 ==== 
regErr = function(dataSet){
  m = ncol(dataSet); n = nrow(dataSet)
  G = sum(2*(dataSet[,m] - dataSet[,m-1])) # sum(yhat - y)^2
  H = 2*n
  lambda = 0.5
  return(G^2/(H + lambda))
}


binSplitDataSet = function(dataSet, feature, value){
  if (is.numeric(dataSet[, feature])){ # 当变量是连续型时
    leftData = dataSet[dataSet[, feature] > value, ]
    rightData = dataSet[dataSet[, feature] <= value, ]
  } else {  # 当变量是类别型时
    leftData = dataSet[dataSet[, feature] == value, ]
    rightData = dataSet[dataSet[, feature] != value, ]
  }
  return(result = list(leftData = leftData, rightData = rightData))
}


chooseBestSplit = function(dataSet, tolS=1, tolN=4){

  # 得到全数据集的行列数
  n = dim(dataSet)[1]; m = dim(dataSet)[2]
  
  # 若因变量y全部都一样的化，直接输出
  if(length(unique(dataSet[, m-1])) == 1){
    return(list(NULL, regLeaf(dataSet)))
  }

  # 根据xgboost打分函数的变化程度大小来选择最优的分割变量
  S = regErr(dataSet) # 计算未分割前因变量的打分值
  # bestS: 最优的打分值
  # bestIndex: 最优分割变量位置
  # bestValue: 最优分割变量下的最优分割值
  bestS = Inf; bestIndex = 0; bestValue = 0
  
  for (featIndex in 1:(m-2)){ # 遍历所有特征(除掉最后2列,y,yhat)
    for (splitVal in unique(dataSet[, featIndex])){ # 遍历每一个特征所有取值的唯一值
      # 根据所遍历的特征以及相应的特征值对数据进行切分
      result = binSplitDataSet(dataSet, featIndex, splitVal)
      leftData = result[[1]]; rightData = result[[2]]
      # 如果切分出来的子树上边的样本个数小于给定的阈值时，不满足分裂条件，跳过这次迭代！！！
      # 此处next的用处等价于python中的continue.
      if ((nrow(leftData) < tolN) || (nrow(rightData) < tolN))  next
      
      # 当两个子树的样本数都大于tolN时，分别计算分裂后两棵子树上边的样本点因变量的RSS，然后求和
      newS = regErr(leftData) + regErr(rightData)

      if (newS < bestS){ 
        bestIndex = featIndex
        bestValue = splitVal
        bestS = newS
      }
    }
  }
  
  if ((S - bestS) < tolS) {
    return(list(NULL, regLeaf(dataSet)))
  }
  # 在最优分割变量下对数据进行重新分割
  result = binSplitDataSet(dataSet, bestIndex, bestValue)
  leftData = result[[1]]; rightData = result[[2]]

  if ((nrow(leftData) < tolN) || (nrow(rightData) < tolN)){
    return(list(NULL, regLeaf(dataSet)))
  }
  # 返回最终最优分割点，最优分割预测值
  return(list(bestIndex = bestIndex, bestValue = bestValue))
}

createTree = function(dataSet, tolS=1, tolN=4){
  # 此函数会调用以上的所有函数
  # dataSet: 全数据集,无需另外切分自变量以及因变量
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


treeForeCast = function(tree, inData){
  # tree: 上边createTree函数生成的树
  # inData: 自变量数据集中的某一行
  
  # 如果到达叶子节点,返回
  if (length(unlist(tree)) == 1)  return(tree)

  is_numeric = is.numeric(inData[tree[['spInd']]]) # 判断是否是连续型
  inDataValue = inData[tree[['spInd']]] # 给出观测数据在叶节点变量下的取值
  if ((is_numeric && inDataValue > tree[['spVal']]) || (!is_numeric && inDataValue == tree[['spVal']])) { 
    if (length(unlist(tree[['left']])) != 1)  treeForeCast(tree[['left']], inData)
    else tree[['left']]
  } else {
    if (length(unlist(tree[['right']])) != 1)  treeForeCast(tree[['right']], inData)
    else tree[['right']]
  }
}

createForeCast = function(tree, testData){
  n <- nrow(testData)
  yHat <- c()
  for (i in 1:n) {
    # 对测试集的数据逐行作出判断，给出预测
    yHat[i] <- treeForeCast(tree, as.matrix(testData[i, ]))
  }
  return(yHat)
}




xgboost = function(dataSet,threshold = 300, tolS = 1, tolN = 4) {
  
  n = nrow(dataSet);  # 整个数据集的行数
  m = ncol(dataSet) # 整个数据集的列数
  
  index = sample(n, 2*n/3) # 抽取训练集行数
  index_test = setdiff(1:n, index) # 测试集行数
  
  train = dataSet[index, ]; test = dataSet[-index, ]
  
  n1 = nrow(train)
  n2 = nrow(test)
  
  F_m_train = rep(mean(train[,m-1]),n1); # 训练集的F_0设置为数据中的均值
  F_m_test = rep(mean(train[,m-1]),n2) # 初始化F_m

  RSS_train = Inf; 
  iterations = 0
  
  residual0 = train[,m] - F_m_train # 初始化残差
  
  # === 更新训练数据集 ===
  # 用training构建树时需要用到residual,yhat
  data1 = cbind(train[, -m], residual0, F_m_train) 
  
  rssTrainVec = c(); mseTestVec = c()
  
  while(RSS_train > threshold) {
    myTree = createTree(data1, tolS, tolN) # 在更新后的训练集data1上训练树
    yHat = createForeCast(myTree, as.matrix(train[, -m])) # 在原训练集上作出预测
    F_m_train = F_m_train + yHat # 更新训练集的Fm. F1~Fm
    
    yHatTest = createForeCast(myTree, as.matrix(test[, -m])) # 在测试集data2上作出预测
    F_m_test = F_m_test + yHatTest # 更新测试集的Fm
    
    RSS_train = sum((train[, m]-F_m_train)^2) # 用y_i - F_m 更新训练集的RSS
    residual_train = train[,m] - F_m_train # 用原数据减Fm即 y_i - F_m 更新残差
    # 更新数据集的因变量，下一轮用残差进行拟合树
    data1 = cbind(train[, -m], residual_train, F_m_train) 
    
    # 记录每一轮训练/测试集的rss
    rssTrainVec = c(rssTrainVec, RSS_train) # 训练误差不可能到达0
    mseTestVec = c(mseTestVec, mean((test[, m]-F_m_test)^2))
    
    iterations = iterations + 1
  }
  mseTest = mean((test[, m]-F_m_test)^2)
  
  return(list(RSS_train = RSS_train, iterations = iterations, mseTest = mseTest))
}

xgboost(Boston, threshold = 12000, tolS = 1, tolN = 2.1)







