

# -*- coding:utf-8 -*- 

# 使用大量循环寻找每种方法下的最优参数
# 计算量过大，计算ing...

# ========================
# === single tree.chen ===
tolSS = seq(1,20,2)
tolNN = c(1.1, 2, 3, 5, 7, 10, 15, 20)
len1 = length(tols); len2 = length(tolN)
bestMse = Inf
mseMat = matrix(0, len1, len2)
numIters = 100

for (num in 1:numIters) {
  train = sample(1:nrow(Boston), nrow(Boston)/2) # 切分数据集
  for (i in 1:len1) {
    for (j in 1:len2) {
      BostonTrain = Boston[train, ]; BostonTest = Boston[-train, ]
      myTree = createTree(BostonTrain, tolS = tolSS[i], tolN = tolNN[j])
      yHat = createForeCast(myTree, as.matrix(BostonTest[,-ncol(BostonTest)]))
      temp = mean((yHat - BostonTest[,ncol(BostonTest)])^2) 
      mseMat[i,j] = mseMat[i,j] + temp
      if (temp < bestMse) {
        bestMse = temp
        bestIndex = c(i, j) 
      }
    }
  }
}
mseAveMat = mseMat/numIters # 求numIters次的均值
which(mseAveMat == min(mseAveMat), arr.ind = T) # 找出最优位置





# =========================
#  ======= bagging.chen ===
# 寻求最优的迭代次数B, tolS, tolN

BB = c(50,100,150,200,500)
tolSS = seq(1,20,2)
tolNN = c(1.1, 2, 3, 5, 7, 10, 15, 20)
len1=length(BB); len2 = length(tolSS); len3 = length(tolNN)
bestMse = Inf
mseMat = array(rep(NaN, len1*len2*len3), c(len1, len2, len3))
numIters = 100

for (num in 1:numIters) {
  train = sample(1:nrow(Boston), nrow(Boston)/2) # 切分数据集
  for (i in 1:len1) {
    for (j in 1:len2) {
      for (k in 1:len3) {
        temp = bagging(Boston[train, ], Boston[-train, ], B = BB[i], tolS = tolSS[j], tolN = tolNN[k])
        mseMat[i, j, k] = mseMat[i, j, k] + temp
        if (temp < bestMse) {
          bestMse = temp # 最佳mse
          bestIndex = c(i, j, k) # 最佳mse的位置索引
        }
      }
    }
  }
}
mseAveMat = mseMat/numIters # 求numIters次的均值
# 先按[,,3]找到每个面最小的，再在向量中找最小的
bestAveMse = min(apply(mseAveMat, 3, min)) 
bestAveIndex=which(mseAveMat == bestAveMse, arr.ind = T) # 找到平均mse最优的索引位置
finalResult = list(bestMse = bestMse, bestIndex = bestIndex, bestAveMse =bestAveMse, bestAveIndex =bestAveIndex)



# ===========================
#  ======= baggingOOB.chen ===

BB = c(50,100,150,200,500)
tolSS = seq(1,20,2)
tolNN = c(1.1, 2, 3, 5, 7, 10, 15, 20)
len1=length(BB); len2 = length(tolSS); len3 = length(tolNN)
bestMse = Inf
mseMat = array(rep(NaN, len1*len2*len3), c(len1, len2, len3))
numIters = 100

for (num in 1:numIters) {
  for (i in 1:len1) {
    for (j in 1:len2) {
      for (k in 1:len3) {
        temp = baggingOOB(Boston, B = BB[i], tolS = tolSS[j], tolN = tolNN[k])
        mseMat[i, j, k] = mseMat[i, j, k] + temp
        if (temp < bestMse) {
          bestMse = temp # 最佳mse
          bestIndex = c(i, j, k) # 最佳mse的位置索引
        }
      }
    }
  }
}
mseAveMat = mseMat/numIters # 求numIters次的均值
which(mseAveMat == min(mseAveMat), arr.ind = T) # 找出最优位置



# ============================
# ==== randomForest.chen =====

BB = c(50,100,150,200,500)
tolSS = seq(1,20,2)
tolNN = c(1.1, 2, 3, 5, 7, 10, 15, 20)
len1=length(BB); len2 = length(tolSS); len3 = length(tolNN)
bestMse = Inf
mseMat = array(rep(NaN, len1*len2*len3), c(len1, len2, len3))
numIters = 100

for (num in 1:numIters) {
  train = sample(1:nrow(Boston), nrow(Boston)/2) # 切分数据集
  for (i in 1:len1) {
    for (j in 1:len2) {
      for (k in 1:len3) {
        temp = randomForestChen(Boston[train, ], Boston[-train, ], B = BB[i], tolS = tolSS[j], tolN = tolNN[k])
        mseMat[i, j, k] = mseMat[i, j, k] + temp
        if (temp < bestMse) {
          bestMse = temp # 最佳mse
          bestIndex = c(i, j, k) # 最佳mse的位置索引
        }
      }
    }
  }
}
mseAveMat = mseMat/numIters # 求numIters次的均值
which(mseAveMat == min(mseAveMat), arr.ind = T) # 找出最优位置



# ===============================
# ==== randomForestOOB.chen =====

BB = c(50,100,150,200,500)
tolSS = seq(1,20,2)
tolNN = c(1.1, 2, 3, 5, 7, 10, 15, 20)
len1=length(BB); len2 = length(tolSS); len3 = length(tolNN)
bestMse = Inf
mseMat = array(rep(NaN, len1*len2*len3), c(len1, len2, len3))
numIters = 100

for (num in 1:numIters) {
  for (i in 1:len1) {
    for (j in 1:len2) {
      for (k in 1:len3) {
        temp = randomForestOOBChen(Boston, B = BB[i], tolS = tolSS[j], tolN = tolNN[k])
        mseMat[i, j, k] = mseMat[i, j, k] + temp
        if (temp < bestMse) {
          bestMse = temp # 最佳mse
          bestIndex = c(i, j, k) # 最佳mse的位置索引
        }
      }
    }
  }
}
mseAveMat = mseMat/numIters # 求numIters次的均值
which(mseAveMat == min(mseAveMat), arr.ind = T) # 找出最优位置



# ===============================
# ========= gbdt.chen ===========

thresholdd = c(3,5,10,15,20,50,70,100)
tolSS = seq(1,20,2)
tolNN = c(1.1, 2, 3, 5, 7, 10, 15, 20)
len1=length(thresholdd); len2 = length(tolSS); len3 = length(tolNN)
bestMse = Inf
mseMat = array(rep(NaN, len1*len2*len3), c(len1, len2, len3))
numIters = 100

for (num in 1:numIters) {
  for (i in 1:len1) {
    for (j in 1:len2) {
      for (k in 1:len3) {
        temp = randomForestOOBChen(Boston, threshold = thresholdd[i], tolS = tolSS[j], tolN = tolNN[k])
        mseMat[i, j, k] = mseMat[i, j, k] + temp
        if (temp < bestMse) {
          bestMse = temp # 最佳mse
          bestIndex = c(i, j, k) # 最佳mse的位置索引
        }
      }
    }
  }
}
mseAveMat = mseMat/numIters # 求numIters次的均值
which(mseAveMat == min(mseAveMat), arr.ind = T) # 找出最优位置









