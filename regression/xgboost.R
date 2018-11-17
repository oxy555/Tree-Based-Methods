

# -*- coding:utf-8 -*- 

# ===========================
# ======== XGBoost ==========
# === 仅支持least square ====

# ===========================
# ==== 单一回归树的构建 =====

regLeaf = function(dataSet, lambda = 0.5){
  # 此处如此处理，dataSet最后一列为拟合值Fm
  # 倒数第二列为原数据集的因变量y
  # The score is -G/(H+λ).
  m = ncol(dataSet); n = nrow(dataSet)
  G = sum(dataSet[,m] - dataSet[,m-1]) 
  H = n
  return(-G/(H + lambda))
}


regErr = function(dataSet, lambda = 0.5){
  # 此处如此处理，dataSet最后一列为拟合值Fm
  # 倒数第二列为原数据集的因变量y
  # The minimized loss is -0.5*G^2/(H+λ)
  m = ncol(dataSet); n = nrow(dataSet)
  G = sum(dataSet[,m] - dataSet[,m-1]) 
  H = n
  return(-0.5*G^2/(H + lambda))
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


chooseBestSplit = function(dataSet, lambda=0.5){
  # 此处如此处理，dataSet最后一列为拟合值Fm
  # 倒数第二列为原数据集的因变量y
  # 得到全数据集的行列数
  m = dim(dataSet)[1]; n = dim(dataSet)[2]
  
  # 根据损失函数的变化程度大小来选择最优的分割变量
  S = regErr(dataSet) # 计算未分割前因变量的loss
  # bestS: 最优收益
  # bestIndex: 最优分割变量位置
  # bestValue: 最优分割变量下的最优分割值
  bestS = 0; bestIndex = 0; bestValue = 0
  
  for (featIndex in 1:(n-2)){ # 遍历所有特征(除掉最后两列)
    for (splitVal in unique(dataSet[, featIndex])){ # 遍历每一个特征所有取值的唯一值
      # 根据所遍历的特征以及相应的特征值对数据进行切分
      result = binSplitDataSet(dataSet, featIndex, splitVal)
      leftData = result[[1]]; rightData = result[[2]]
      
      # 当变量为连续值时, 在边界点会出现树的一边不存在任何数据,
      # 会出现NA的情况, 此时令其rightErr为0
      leftErr = regErr(leftData); rightErr = regErr(rightData)
      lefterr = ifelse(is.na(leftErr), 0, leftErr)
      rightErr = ifelse(is.na(rightErr), 0, rightErr) 
      
      gain = S - lefterr - rightErr # 计算分裂收益
      
      if (gain > bestS){ 
        bestIndex = featIndex
        bestValue = splitVal
        bestS = gain
      }
    }
  }
  return(list(bestIndex = bestIndex, bestValue = bestValue, bestS = bestS))
}


createTree = function(dataSet, lambda=0.5, tolS=1, tolN=10, max_depth = 3){
  retTree <- list() # 创建树的根结点
  bestSplit = chooseBestSplit(dataSet, lambda=0.5)
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
  retTree$left = createTree(lSet, lambda, tolS, tolN, max_depth - 1)
  retTree$right = createTree(rSet, lambda, tolS, tolN, max_depth - 1)
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
  # 因此testData相比于training少了两列(含预测值列)
  m <- nrow(testData)
  yHat <- c()
  for (i in 1:m) {
    # 对测试集的数据逐行作出判断，给出预测
    yHat[i] <- treeForeCast(tree, as.matrix(testData[i, ]))
  }
  return(yHat)
}


# ========================================
# === xgboost.tree & xgboost.predict =====
xgboost.tree = function(training, lambda=0.5, B= 100, tolS = 1, tolN = 10, max_depth = 3) {
  # 未添加预测值列时的数据
  n = nrow(dataSet);  
  m = ncol(dataSet) 
  F_m_train = rep(mean(training[,m]),n);
  trees = list()

  # --- 更新训练数据集 ---
  # 用training构建树时需要用到yhat
  data = cbind(training, F_m_train)
  
  for (i in 1:B) {
    tree = createTree(data, lambda, tolS, tolN, max_depth)
    trees[[i]] = tree # store each tree in a list
    
    F_m_train = F_m_train + createForeCast(tree, training[, -m])
    data = cbind(training, F_m_train) # 继续更新数据集
    
    mse = mean((training[, m]-F_m_train)^2)
    print(paste0(i, "th iteration, current training mse = ", mse))
  }
  return(trees)
}


xgboost.predict = function(predict_tree, training, test){
  yHat = mean(training[,ncol(training)])
  # yHat = rep(0, nrow(test))
  trees.num = length(predict_tree)
  for (i in 1:trees.num) {
    tree = predict_tree[[i]]
    yHat = yHat + createForeCast(tree, test[, -ncol(test)])
    mse = mean((yHat - test[, ncol(test)])^2)
    print(paste0(i, "th iteration, current test mse = ", mse))
  }
  return(yHat)
}

trees = xgboost.tree(Boston[train, ], lambda=0.5, B = 100, tolS = 1, tolN = 10, max_depth = 2)
preds = xgboost.predict(trees, Boston[train, ], Boston[-train, ])
mse_xgboost_chen = mean((preds-Boston[-train, ncol(Boston)])^2)








