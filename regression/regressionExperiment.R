
# ====== MASS包中的Boston数据进行预测 ======
# ====== R自带方法与自行编写方法比较 =======


# =========== R包自带方法 ==========
library(MASS)
library(tree)
train = sample(1:nrow(Boston), nrow(Boston)/2)

# ---- 单棵回归树 ----
tree.boston=tree(medv~.,Boston ,subset=train)
yhat=predict(tree.boston ,newdata=Boston[-train ,])
boston.test=Boston[-train ,"medv"]
mse_single_R= mean((yhat-boston.test)^2) 

# ---- bagging ------
library(randomForest)
bag.boston=randomForest(medv ~., data = Boston, subset = train, importance = T, mtry = 13)
yhat.bag = predict(bag.boston ,newdata=Boston[-train ,])
mse_bagging_R = mean((yhat.bag-boston.test)^2)

# ---- randomForest ----
rf.boston=randomForest(medv~.,data=Boston,subset=train,
                       mtry=6,importance =TRUE)
yhat.rf = predict(rf.boston ,newdata=Boston[-train ,])
mse_rf_R = mean((yhat.rf-boston.test)^2) # 10.37

# ---- boosting ------
boost.boston=gbm(medv~.,data=Boston[train,],distribution=
                   "gaussian",n.trees=5000, interaction.depth=4)
yhat.boost=predict(boost.boston,newdata=Boston[-train,], n.trees=5000)
mse_boost_R = mean((yhat.boost -boston.test)^2)

mse_treeBase_R = c(mse_single_R = mse_single_R, mse_bagging_R = mse_bagging_R, 
                          mse_rf_R = mse_rf_R, mse_boost_R = mse_boost_R)




# ======== 自行编写方法 =========
#  ---------- 调参 ---------

# ---- 单棵回归树.chen ----
BostonTrain = Boston[train, ]; BostonTest = Boston[-train, ]
myTree = createTree(BostonTrain, tolS = 1, tolN = 10)
yHat = createForeCast(myTree, as.matrix(BostonTest[,-ncol(BostonTest)]))
mse_singe_chen = mean((yHat - BostonTest[,ncol(BostonTest)])^2) 


# ---- bagging.chen -----
mse_bagging_chen = bagging(Boston[train, ], Boston[-train, ])


# ---- out of bag.chen ----
# 两个效果差不多，主观感觉第一个还快一点点
mse_OOB_chen = baggingOOB(Boston);
mse_OOB_chen[[2]]

mse_OOB_chen1 = baggingOOB1(Boston)
mse_OOB_chen1[[2]]


# ---- randomForest.chen ----
mse_rf_chen = randomForestChen(Boston[train, ], Boston[-train, ])


# ---- randomForest_out of bag.chen ----
mse_rf_OOB_chen = randomForestOOBChen(Boston)
mse_rf_OOB_chen[[2]]

# ---- gradeint boosting desicion tree ----
mse_gbdt_chen = gbdt(Boston)
mse_gbdt_chen[[3]]


mse_treeBase_chen = c(mse_singe_chen = mse_singe_chen, mse_bagging_chen = mse_bagging_chen, 
                         mse_OOB_chen = mse_OOB_chen[[2]], mse_rf_chen = mse_rf_chen, 
                         mse_rf_OOB_chen = mse_rf_OOB_chen[[2]], mse_gbdt_chen = mse_gbdt_chen[[3]])

method_compare = list(mse_treeBase_R, mse_treeBase_chen)
method_compare


