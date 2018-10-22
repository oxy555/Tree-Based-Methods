

# -*- coding:utf-8 -*-  

# == classification and regression trees == 

# 此处怎么优化 以至于可以不需要一开始就输出一个值出来
# 如何封装进其余函数里边
is_real_continuou = function(dataSet) {
  # 此处以因变量的唯一值个数多少来区分因变量为连续型还是离散型.
  # 当因变量唯一值个数>2时，当连续型因变量处理，否则为类别型. 
  length(unique(dataSet[,ncol(dataSet)])) > 2
}

regLeaf = function(dataSet){
  # 根结点返回值
  # 当因变量为连续型时，根结点返回均值
  if(is_real_continuous){ # 生成树前需提前判定
    mean(dataSet[,ncol(dataSet)])
  } else {
    # 当因变量为类别型时，根结点返回最常出现的类
    labelFrame = as.data.frame(table(dataSet[, ncol(dataSet)]))
    index = which(labelFrame$Freq == max(labelFrame$Freq))
    labelFrame$Var1[index]
  }
}


regErr = function(dataSet){
  # 当因变量为连续型时，评判标准为RSS
  if(is_real_continuous){
    var(dataSet[,ncol(dataSet)])*(nrow(dataSet)-1)
  } else { # 当因变量为类别型时， 评判标准为分类错误率
    mean(dataSet[, ncol(dataSet)] != regLeaf(dataSet))
  }
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
  # tolN: 切分的样本最小数,tolN >= 2.
  
  # ------ 约束条件一 ------
  # 如果因变量y只有一个类别，则不需作任何分割
  # 因此预测值为因变量下数据的平均值即可,运行结束,退出
  if(length(unique(dataSet[, ncol(data)])) == 1){
    return(list(NULL, regLeaf(dataSet)))
  }
  
  # 得到全数据集的行列数
  m = dim(dataSet)[1]; n = dim(dataSet)[2]
  
  # 根据RSS/error_rate的变化程度大小来选择最优的分割变量
  S = regErr(dataSet) # 计算未分割前因变量的RSS/error_rate
  bestS = Inf; bestIndex = 0; bestValue = 0
  
  for (featIndex in 1:(n-1)){ # 遍历所有特征(除掉最后一列y因变量)
    for (splitVal in unique(dataSet[, featIndex])){ # 遍历每一个特征所有取值的唯一值
      # 根据所遍历的特征以及相应的特征值对数据进行切分
      result = binSplitDataSet(dataSet, featIndex, splitVal)
      leftData = result[[1]]; rightData = result[[2]]
      # 如果切分出来的子树上边的样本个数小于给定的阈值时，不满足分裂条件，跳过这次迭代！！！
      # 此处next的用处等价于python中的continue.
      if ((nrow(leftData) < tolN) || (nrow(rightData) < tolN))  next
      
      # 当两个子树的样本数都大于tolN时，分别计算分裂后两颗子树上边的样本点因变量的RSS/error_rate，然后求和
      newS = regErr(leftData) + regErr(rightData)
      
      # 如果分裂后得到RSS/error_rate的和比未分裂前的小，
      # 那么确认此次分裂是可行的.
      # 输出此次分裂所选取的分裂变量，分裂值，以及分裂后得到更新的RSS/error_rate。
      # 如果分裂后得到的RSS的和比未分裂前的大，那么不更新分裂变量，分裂值及RSS,
      if (newS < bestS){ 
        bestIndex = featIndex
        bestValue = splitVal
        bestS = newS
      }
    }
  }
  
  # ---- 约束条件二 ----
  # 如果最优分割下的RSS/error_rate减少值小于给定的阈值，
  # 此次分割作废,运行结束，退出
  # 继续返回 预测值为所有值的平均值
  if ((S - bestS) < tolS) {
    return(list(NULL, regLeaf(dataSet)))
  }
  # 在最优分割变量下对数据进行重新分割
  result = binSplitDataSet(dataSet, bestIndex, bestValue)
  leftData = result[[1]]; rightData = result[[2]]
  
  # ---- 约束条件三 -----
  # 为稳妥起见，如果最优RSS/error_rate分割下的子树个数小于阈值，退出
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


# ====回归树,处理自变量是连续值的数据====
# ===== 对'ex00.txt'数据进行切分 ======
myData = read.table(".../ex00.txt")
head(myData)
is_real_continuous = is_real_continuou(myData)
createTree(myData, tolS=0.0004, tolN=2) 
createTree(myData, tolS=0.4, tolN=5) 


# ====== 对'ex0.txt'数据进行切分 =====

myData1 = read.table(".../ex0.txt")
head(myData1)
is_real_continuous = is_real_continuou(myData1)
createTree(myData1, tolS=0.0004, tolN=2)
createTree(myData1, tolS=0.4, tolN=5) 


# ====== 对'ex2.txt'数据进行切分 =====

myData2 = read.table(".../ex2.txt")
head(myData2)
is_real_continuous = is_real_continuou(myData2)
createTree(myData2, tolS=0.0004, tolN=2)
createTree(myData2, tolS=0.4, tolN=5) 
createTree(myData2, tolS = 6, tolN = 24)


# ==== 回归树，处理自变量中具有连续型和类别型的====
# ==== 使用R自带的ISLR包里的Carseats数据集 ====
library(ISLR)
head(Carseats)
dim(Carseats)

# 因为我构建树的代码都默认最后一列为因变量,
# 所以这里需要把Sales放到最后一列去
myData3 = cbind(Carseats[,-1], Carseats$Sales)
names(myData3)[11] = "Sales" # 把最后一列的名字改回来
head(myData3)
is_real_continuous = is_real_continuou(myData3)
createTree(myData3)
createTree(myData3, tolS = 6, tolN = 24)
createTree(myData3, tolS = 3, tolN = 5)



# === 分类树，自变量中连续型和类别型都有 ====
myData4 = read.csv(".../credit.data.csv")
head(myData4)
is_real_continuous = is_real_continuou(myData4)
createTree(myData4, tolS=0.0004, tolN=1) # 为什么这里tolN用1可以?
createTree(myData4, tolS=0.00001, tolN=6)


