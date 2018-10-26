

# -*- coding:utf-8 -*- 

# 回归树/基于回归树下的bagging/randomForest/gbdt
# 自变量支持连续型与类别型


# ===========================
# ==== 单一回归树的构建 =====

regLeaf = function(dataSet){
  # 计算每个叶子取值
  # 计算因变量的均值
  mean(dataSet[,ncol(dataSet)])
}


regErr = function(dataSet){
  # 计算当前叶子结点样本的RSS，从而用于决定是否分裂
  # 计算因变量的RSS
  # 样本方差*(样本个数-1) = RSS
  var(dataSet[,ncol(dataSet)])*(nrow(dataSet)-1)
}


binSplitDataSet = function(dataSet, feature, value){
  
  # --- 根据特征及特征值切分数据集 --- 
  # dataSet: 数据集
  # feature: 所选取的特征
  # value: 所选取特征下的特征值
  
  # 添加判断变量为连续型or类别型
  # 当变量时数值型时，
  # 保持与《machine learning in action》一致的分裂法；
  # 将数据某列大于某个值别的样本添加在树左边，
  # 将小于这个值的样本放在树右边
  
  # 当变量为类别型时，将数据某列等于某个类别时的样本放在树左边，
  # 剩余的样本放在树的右边
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
  # 此函数将遍历所有特征所有取值，从而得到最优特征最优取值
  # tolS: 容许的误差下降值
  # tolN: 切分的样本最小数
  
  # ------ 约束条件一 ------
  # 如果因变量y只有一个类别，则不需作任何分割
  # 因此预测值为因变量下数据的平均值即可,运行结束,退出
  if(length(unique(dataSet[, ncol(dataSet)])) == 1){
    return(list(NULL, regLeaf(dataSet)))
  }
  
  # 得到全数据集的行列数
  m = dim(dataSet)[1]; n = dim(dataSet)[2]
  
  # 根据RSS的变化程度大小来选择最优的分割变量
  S = regErr(dataSet) # 计算未分割前因变量的RSS
  # bestS: 最优的RSS
  # bestIndex: 最优分割变量位置
  # bestValue: 最优分割变量下的最优分割值
  bestS = Inf; bestIndex = 0; bestValue = 0
  
  for (featIndex in 1:(n-1)){ # 遍历所有特征(除掉最后一列y因变量)
    for (splitVal in unique(dataSet[, featIndex])){ # 遍历每一个特征所有取值的唯一值
      # 根据所遍历的特征以及相应的特征值对数据进行切分
      result = binSplitDataSet(dataSet, featIndex, splitVal)
      leftData = result[[1]]; rightData = result[[2]]
      # 如果切分出来的子树上边的样本个数小于给定的阈值时，不满足分裂条件，跳过这次迭代！！！
      # 此处next的用处等价于python中的continue.
      if ((nrow(leftData) < tolN) || (nrow(rightData) < tolN))  next
      
      # 当两个子树的样本数都大于tolN时，分别计算分裂后两棵子树上边的样本点因变量的RSS，然后求和
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



# =============================
# ==== 单一树下的预测函数 =====

treeForeCast = function(tree, inData){
  # tree: 上边createTree函数生成的树
  # inData: 自变量数据集中的某一行
  
  # 如果到达叶子节点,返回
  if (length(unlist(tree)) == 1)  return(tree)
  
  # 按照连续型/类别型处理
  # 当变量是连续型观测值且大于已生成的树的阈值时，将此观测点置于树的左边
  # 当变量是类别型观测值且等于已生成的树的阈值时，将此观测点置于树的左边
  # 若数据放到的左边/右边刚好是叶子节点时，输出。否则递归，继续判断
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
  # testData 中已经删除掉了因变量列，使用时注意
  # 需要将testData矩阵化，原因未明
  m <- nrow(testData)
  yHat <- c()
  for (i in 1:m) {
    # 对测试集的数据逐行作出判断，给出预测
    yHat[i] <- treeForeCast(tree, testData[i, ])
  }
  return(yHat)
}


# ===================================
# ==== 基于回归树下的bagging ========

# ------ 基于交叉验证 ------
bagging = function(training, test, B = 100){
  # 其实m = m1；也可不算test的列数
  n = nrow(training); m = ncol(training)
  n1 = nrow(test); m1 = ncol(test)
  
  yHatMat = matrix(0, B, n1)
  
  for (i in 1:B) {
    # 利用boostrap抽样B次,分别拟合树模型
    index = sample(n, n, replace = T)
    trainBstrap = training[index, ]
    tree = createTree(trainBstrap, tolS = 1, tolN = 10)
    # 根据拟合出来的树模型对测试集进行预测
    yHatMat[i, ] = yHatMat[i, ] + createForeCast(tree, as.matrix(test[,-m1]))
  }
  
  yHat = colMeans(yHatMat)
  mse = mean((yHat - test[,m1])^2)
  return(mse)
}
bagging(Boston[train, ], Boston[-train, ])


# ----- 基于 out of bag -----
baggingOOB = function(dataSet, B = 100) {
  n = nrow(dataSet)
  m = ncol(dataSet)
  yHat = rep(0, n)
  ncount = rep(0, n)
  
  for (i in 1:B) {
    index = sample(n, 2*n/3) 
    index_test = setdiff(1:n, index)
    train = dataSet[index, ]; test = dataSet[index_test, ]
    tree = createTree(train, tolS = 1, tolN = 10)
    yHat[index_test] = yHat[index_test] + createForeCast(tree, as.matrix(test[,-m]))
    ncount[index_test] = ncount[index_test] + 1
  }
  
  yHatOOB = yHat/ncount
  mse = mean((yHatOOB - dataSet[,m])^2)
  
  return(list(yHatOOB = yHatOOB, mse = mse, ncount = ncount))
}



# ==================================
# === 基于回归树下的randomForest ===

# ------ 基于交叉验证 -----
randomForestChen = function(training, test, B = 400){
  
  n = nrow(training); m = ncol(training)
  n1 = nrow(test); m1 = ncol(test)
  p = m-1 # 自变量个数
  
  yHatMat = matrix(0, B, n1)
  
  for (i in 1:B) {
    # 利用boostrap抽样B次,分别拟合树模型
    index = sample(n, n, replace = T)
    index_var = sample(p, p/3) # 抽取自变量
    # 利用boostrap进行 行重抽样,同时只希望用p/3个变量建模
    trainBstrap = training[index, c(index_var, m)]
    tree = createTree(trainBstrap, tolS = 1, tolN = 10)
    # 根据拟合出来的树模型对测试集进行预测,此时要去除因变量列
    yHatMat[i, ] = yHatMat[i, ] + createForeCast(tree, as.matrix(test[, index_var]))
  }
  
  yHat = colMeans(yHatMat)
  mse = mean((yHat - test[,m1])^2)
  return(mse)
}


# ----- 基于out of bag -----
randomForestOOBChen= function(dataSet, B = 400) {
  n = nrow(dataSet); 
  m = ncol(dataSet)
  p = m-1
  yHat = rep(0, n)
  ncount = rep(0, n)
  
  for (i in 1:B) {
    index = sample(n, 2*n/3) 
    index_test = setdiff(1:n, index)
    
    index_var = sample(p, p/3)
    
    # 生成树的训练集是数据集全集, 预测的函数是去掉因变量的数据集
    train = dataSet[index, c(index_var, m)]; test = dataSet[index_test, index_var]
    
    tree = createTree(train, tolS = 1, tolN = 10)
    yHat[index_test] = yHat[index_test] + createForeCast(tree, as.matrix(test))
    ncount[index_test] = ncount[index_test] + 1
  }
  
  yHat_rf = yHat/ncount
  mse = mean((yHat_rf - dataSet[, m])^2)
  
  return(list(yHat_rf = yHat_rf, mse = mse, ncount = ncount))
}




# ==================================
# ===== 基于回归树下的gbdt =====

gbdt = function(dataSet,threshold = 300) {
  # dataSet:全数据集.与上边的bagging/randomForestChen保持一致
  # gbdt函数自动切分training与test.以下仅支持最小二乘作为损失函数。
  
  # 在training中训练完每一棵子树后,用形成的大树对测试集进行预测.
  # 由于目前尚未清楚如何存储每一棵子树以及如何调用子树进行预测,
  # 这里这么处理:训练一棵子树的同时用这棵对测试集进行预测,当训练终止，
  # 预测的最终结果也同时输出了
  
  n = nrow(dataSet);  # 整个数据集的行数
  m = ncol(dataSet) # 整个数据集的列数
  
  index = sample(n, 2*n/3) # 抽取训练集行数
  index_test = setdiff(1:n, index) # 测试集行数
  
  train = dataSet[index, ]; test = dataSet[-index, ]
  # 因后边要用到train和test中的数据，所以不改变这两数据
  # 故而创建两等价数据用于boosting中更新数据集
  # 因为gradient boosting decision tree是用初始的自变量以及每一次更新后的残差
  data1 = train; data2 = test # 每一次迭代需更新的数据集
  
  n1 = nrow(train)
  n2 = nrow(test)
  
  F_m_train = 0; F_m_test = 0 # 初始化F_m
  RSS_train = Inf; 
  iterations = 0
  
  while(RSS_train > threshold) {
    myTree = createTree(data1, tolS = 1, tolN = 10) # 在data1上训练树
    yHat = createForeCast(myTree, as.matrix(data1[, -m])) # 在训练集data1上作出预测
    F_m_train = F_m_train + yHat # 更新训练集的Fm
    
    yHatTest = createForeCast(myTree, as.matrix(data2[, -m])) # 在测试集data2上作出预测
    F_m_test = F_m_test + yHatTest # 更新测试集的Fm
    
    RSS_train = sum((train[, m]-F_m_train)^2) # 用y_i - F_m 更新训练集的RSS
    residual_train = train[,m] - F_m_train # 用原数据减Fm即 y_i - F_m 更新残差
    # 更新数据集的因变量，下一轮用残差进行拟合树
    data1 = cbind(data1[, -m], residual_train) 
    
    residual_test = test[,m] - F_m_test # 用原数据减Fm跟新残差
    
    # 更新数据集的因变量，下一轮用残差进行拟合树
    data2 = cbind(data2[, -m], residual_test) 
    
    iterations = iterations + 1
  }
  mseTest = mean((test[, m]-F_m_test)^2)
  
  return(list(RSS_train = RSS_train, iterations = iterations, mseTest = mseTest))
}











