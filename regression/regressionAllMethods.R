


# -*- coding:utf-8 -*- 

# ==== revised in 2018/11/16 ====
# 添加限制单一生成树的深度(max_depth)

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
  if (is.numeric(dataSet[, feature])){ # 当变量是连续型时
    leftData = dataSet[dataSet[, feature] > value, ]
    rightData = dataSet[dataSet[, feature] <= value, ]
  } else {  # 当变量是类别型时
    leftData = dataSet[dataSet[, feature] == value, ]
    rightData = dataSet[dataSet[, feature] != value, ]
  }
  return(result = list(leftData = leftData, rightData = rightData))
}


chooseBestSplit = function(dataSet){
  # 此函数将遍历所有特征所有取值，从而得到最优特征最优取值
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
      
      # 当变量为连续值时, 在边界点会出现树的一边不存在任何数据,
      # 会出现NA的情况, 此时令其rss为0
      leftErr = regErr(leftData); rightErr = regErr(rightData)
      lefterr = ifelse(is.na(leftErr), 0, leftErr)
      rightErr = ifelse(is.na(rightErr), 0, rightErr) 
      
      newS = lefterr + rightErr
      
      if (newS < bestS){ 
        bestIndex = featIndex
        bestValue = splitVal
        bestS = newS
      }
    }
  }
  return(list(bestIndex = bestIndex, bestValue = bestValue, bestS = bestS))
}


createTree = function(dataSet, tolS=1, tolN=10, max_depth = 3){
  retTree <- list() # 创建树的根结点
  bestSplit = chooseBestSplit(dataSet)
  bestIndex = bestSplit[[1]]; 
  bestValue = bestSplit[[2]];
  bestS = bestSplit[[3]]
  
  # 到达最大深度或者叶子结点的样本数小于阈值,
  # 直接返回预测值
  if (max_depth==0 || nrow(dataSet) < tolN){
    return(regLeaf(dataSet))
  }
  
  if (bestS < tolS) {
    return(regLeaf(dataSet))
  }
  
  retTree$spInd = bestIndex
  retTree$spVal = bestValue
  
  result = binSplitDataSet(dataSet, bestIndex, bestValue)
  lSet = result[[1]]; rSet = result[[2]]
  
  # 递归形成大树
  retTree$left = createTree(lSet, tolS, tolN, max_depth - 1)
  retTree$right = createTree(rSet, tolS, tolN, max_depth - 1)
  return(retTree)
}


# =============================
# ==== 单一树下的预测函数 =====

treeForeCast = function(tree, inData){
  # tree: 上边createTree函数生成的树
  # inData: 自变量数据集中的某一行, 无因变量列
  
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
  m <- nrow(testData)
  yHat <- c()
  for (i in 1:m) {
    # 对测试集的数据逐行作出判断，给出预测
    yHat[i] <- treeForeCast(tree, as.matrix(testData[i, ]))
  }
  return(yHat)
}



# ===================================
# ==== 基于回归树下的bagging ========

# ===== bagging.tree and bagging.predict ===== 
# we can use bagging to generate a lot of tree,
# and make a prediction in test dataset

bagging.tree = function(training, B = 100, tolS = 1, tolN = 10, max_depth = 3) {
  n = nrow(training); m = ncol(training)
  trees = list()
  for (i in 1:B) {
    index = sample(n, n, replace = T)
    trainBstrap = training[index, ]
    tree = createTree(trainBstrap, tolS, tolN, max_depth)
    trees[[i]] = tree
  }
  return(trees)
}

bagging.predict = function(predict_tree, test){
  yHat = rep(0, nrow(test))
  trees.num = length(predict_tree)
  for (i in 1:trees.num) {
    tree = predict_tree[[i]]
    yHat = yHat + createForeCast(tree, test[,-ncol(test)])
  }
  return(yHat/trees.num)
}

trees = bagging.tree(Boston[train, ], B = 10, tolS = 1, tolN = 10, max_depth = 10)
preds = bagging.predict(trees, Boston[-train, ])
mse_bagging_chen = mean((preds-Boston[-train, ncol(Boston)])^2)



# ==================================
# === 基于回归树下的randomForest ===
# ---- 预测会出现NA -----
# ---- Bug 待排查 -----

randomForest.tree = function(training, B = 100, mtry = 6, tolS = 1, tolN = 10, max_depth = 3) {
  n = nrow(training); m = ncol(training)
  p = m-1 # 自变量个数
  
  trees = list()
  varMat = matrix(0, B, mtry)
  for (i in 1:B) {
    index = sample(n, n, replace = T)
    index_var = sample(p, mtry) # 抽取自变量个数
    varMat[i, ] = index_var # 存储每一次抽取的变量,用于预测
    # 利用boostrap进行 行重抽样,同时只希望用mtry个变量建模
    trainBstrap = training[index, c(index_var, m)]
    tree = createTree(trainBstrap, tolS, tolN, max_depth)
    trees[[i]] = tree
  }
  return(list(trees=trees, varMat = varMat))
}

randomForest.predict = function(predict_tree, var, test){
  yHat = rep(0, nrow(test))
  trees.num = length(predict_tree)
  for (i in 1:trees.num) {
    tree = predict_tree[[i]]
    # make prediction base on the variable you used in 
    # generate tree
    yHat = yHat + createForeCast(tree, test[, var[i, ]])
  }
  return(yHat/trees.num)
}

randomForest_result = randomForest.tree(Boston[train, ], B = 100, mtry = 6, tolS = 1, tolN = 10, max_depth = 10)
trees = randomForest_result[[1]]; varMat = randomForest_result[[2]]
preds = randomForest.predict(trees, varMat, Boston[-train, ])
mse_randomForest_chen = mean((preds-Boston[-train, ncol(Boston)])^2)



# ==================================
# =====  基于回归树下的gbdt ======
# --- 损失函数取least square -----

gbdt.tree = function(training, B= 100, tolS = 1, tolN = 10, max_depth = 3) {
  n = nrow(training); m = ncol(training)
  F_m = rep(mean(training[,m]),n); # F_0设置为数据中y的均值

  residual0 = training[,m] - F_m # 初始化残差
  data = cbind(training[, -m], residual0) # 更新训练数据集
  
  trees = list()
  
  for (i in 1:B) {
    tree = createTree(data, tolS, tolN, max_depth)
    trees[[i]] = tree # store each tree in a list
    
    yHat = createForeCast(tree, data[, -m])
    F_m = F_m + yHat
    
    residual = training[,m] - F_m
    mse = mean((training[, m]-F_m)^2)
    
    data = cbind(data[, -m], residual) 
    print(paste0(i, "th iteration, current training mse = ", mse))
  }
  return(trees)
}


gbdt.tree1 = function(training, B= 100, tolS = 1, tolN = 10, max_depth = 3) {
  n = nrow(training); m = ncol(training)
  F_m = rep(0,n); # F_0设置为数据中y的均值
  
  data = training
  trees = list()
  
  for (i in 1:B) {
    tree = createTree(data, tolS, tolN, max_depth)
    trees[[i]] = tree # store each tree in a list
    
    yHat = createForeCast(tree, data[, -m])
    F_m = F_m + yHat
    
    residual = training[,m] - F_m
    mse = mean((training[, m]-F_m)^2)
    
    data = cbind(data[, -m], residual) 
    print(paste0(i, "th iteration, current training mse = ", mse))
  }
  return(trees)
}



gbdt.predict = function(predict_tree, training, test){
  # yHat0 = mean(training[,ncol(training)])
  yHat = rep(0, nrow(test))
  trees.num = length(predict_tree)
  for (i in 1:trees.num) {
    tree = predict_tree[[i]]
    yHat = yHat + createForeCast(tree, test[, -ncol(test)])
    mse = mean((yHat - test[, ncol(test)])^2)
    print(paste0(i, "th iteration, current test mse = ", mse))
  }
  return(yHat)
}

trees = gbdt.tree1(Boston[train, ], B = 30, tolS = 1, tolN = 10, max_depth = 2)
preds = gbdt.predict(trees, Boston[train, ], Boston[-train, ])
mse_gbdt_chen = mean((preds-Boston[-train, ncol(Boston)])^2)





