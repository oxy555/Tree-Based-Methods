# Tree-Based-Methods

**regressionTree**因变量为连续性，自变量可支持连续性与类别型，树的每次分割基于RSS的变化程度。regressionTree已加入预测功能，目前仅支持连续型自变量。  

**classificationTree**因变量为类别型(支持二类别与多类别)，自变量支持连续性与类别型，树的每次分割目前仅基于分类错误率的变化程度。后续会加上熵的变化以及基尼系数的变化作为切分标准。  

**CART**则是将classificationTree 与 regressionTree 结合起来。可通过改变参数调用两种树。

**regression**文件夹里包含里回归树下的单一回归树，bagging, bagging_out_of_bag, randomForest, randomForest_out_of_bag, gbdt方法。运行完regressionAllMethod后，运行regressionExperiment即可得[output](https://github.com/ChenShicong/Tree-Based-Methods/blob/master/regression/output)。从output可以看到，自行编写的tree based methods效果还是相当不错的。

文档构建树的实现基于《机器学习实战》一书第九章。其python3代码实现可见[MachineLearningInAction](https://github.com/pbharrin/machinelearninginaction3x/tree/master/Ch09).   
本文档生成树时使用list存储每一个节点的信息的方法参照[基于R存储叶节点信息](https://github.com/kaustubhrpatil/HDDT/blob/master/HDDT.R)

