
# ====== MASS包中的Boston数据进行预测 ======
# ====== R自带方法与自行编写方法比较 =======


# =========== R包自带方法 ==========
library(MASS)
library(tree)
library(gbm)
library(randomForest)
train = sample(1:nrow(Boston), nrow(Boston)/2)

# ---- 单棵回归树 ----
tree.boston=tree(medv~.,Boston ,subset=train)
yhat=predict(tree.boston ,newdata=Boston[-train ,])
boston.test=Boston[-train ,"medv"]
mse_single_R= mean((yhat-boston.test)^2) 

# ---- bagging ------
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
                   "gaussian",n.trees=500, interaction.depth=2)
yhat.boost=predict(boost.boston,newdata=Boston[-train,], n.trees=500)
mse_boost_R = mean((yhat.boost -boston.test)^2)

mse_treeBase_R = c(mse_single_R = mse_single_R, mse_bagging_R = mse_bagging_R, 
                          mse_rf_R = mse_rf_R, mse_boost_R = mse_boost_R)




# ======== 自行编写方法 =========
#  ---------- 调参 ---------

# ---- 单棵回归树.chen ----
BostonTrain = Boston[train, ]; BostonTest = Boston[-train, ]
myTree = createTree(BostonTrain, tolS = 1, tolN = 10, max_depth = 10)
yHat = createForeCast(myTree, BostonTest[,-ncol(BostonTest)])
mse_singe_chen = mean((yHat - BostonTest[,ncol(BostonTest)])^2) 


# ---- bagging.chen -----
trees = bagging.tree(Boston[train, ], B = 300, tolS = 1, tolN = 10, max_depth = 10)
preds = bagging.predict(trees, Boston[-train, ])
mse_bagging_chen = mean((preds-Boston[-train, ncol(Boston)])^2)



# ---- randomForest.chen ----
randomForest_result = randomForest.tree(Boston[train, ], B = 100, mtry = 6, tolS = 1, tolN = 10, max_depth = 10)
trees = randomForest_result[[1]]; varMat = randomForest_result[[2]]
preds = randomForest.predict(trees, varMat, Boston[-train, ])
mse_randomForest_chen = mean((preds-Boston[-train, ncol(Boston)])^2)



# ---- gradeint boosting desicion tree ----
mse_gbdt_chen = gbdt1(Boston, threshold = 250, tolS = 0, tolN = 1.1)


mse_treeBase_chen = c(mse_singe_chen = mse_singe_chen, mse_bagging_chen = mse_bagging_chen, 
                         mse_OOB_chen = mse_OOB_chen, mse_rf_chen = mse_rf_chen, 
                         mse_rf_OOB_chen = mse_rf_OOB_chen, mse_gbdt_chen = mse_gbdt_chen)

method_compare = list(mse_treeBase_R = mse_treeBase_R, mse_treeBase_chen = mse_treeBase_chen)
method_compare


