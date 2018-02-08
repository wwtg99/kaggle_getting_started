Kaggle Getting started Competitions Examples
============================================

Kaggle 入门竞赛的示例代码，其中 input 为输入文件存放位置。

# Titanic: Machine Learning from Disaster

URL：[https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
Script：titanic.py

## 简要介绍

### 确定问题

这是一个二分类问题，泰坦尼克号上人员的生存预测。

输入数据有 12 列，分别是
- PassengerId：人员 ID
- Survived：生存标签，1 生存，0 没有生存
- Pclass：票的类型
- Name：姓名
- Sex：性别
- Age：年龄
- SibSp：同行的同辈（兄弟、配偶）数量
- Parch：同行的父母、孩子数量
- Ticket：票号
- Fare：票价
- Cabin：船舱
- Embarked：登船地

### 数据预处理

首先查看一下各个特征的缺失情况及与标签的相关程度（脚本中没有）。初期可以直接去除缺失值较多或不太相关的特征，对于缺失值较少的特征，可以采用众数（如 Embarked）、均值或中位数（如 Fare）及其他更复杂的方法补充缺失值。

### 特征提取

对于相关性十分高的特征，可以直接使用，对于相关性不太高的特征，则可以采取合并、转换等操作，如从 Name 中提取 Title，将 SibSp 和 Parch 合并等。对于连续性特征，可以采用切分区段的方式（如 Age），对于分类特征，可以使用 One Hot 方式（如 Embarked）编码。

### 模型选择

可以试验对比多个模型的效果选择最终模型，也可以根据经验和数据分布类型等选择候选模型。这里使用了 XGBoost 模型，参数的调优可以参考这里。

### 预测

最后使用训练好的模型进行预测，预测的准确度除了算法外，主要依赖特征工程的处理，本脚本主要作为示例，成绩大概在 TOP 20% 左右。

# House Prices: Advanced Regression Techniques

URL：[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
Script: house_prices.py

## 简要介绍

### 确定问题

只是一个回归问题，用于预测房屋买卖价格。
输入数据一共有 79 列，比较多就不一一列出了，可以参考网站上及文档的描述。

### 数据预处理

首页查看一下各个特征的缺失情况及与房价的相关程度（脚本中没有）。初期可以直接去除缺失值较多或不太相关的特征，如 MiscFeature，对于缺失值较少的特征，可以采用众数（如 MSZoning）、均值或中位数（如 LotFrontage）、特殊值（如 MasVnrType）及其他更复杂的方法补充缺失值。

### 特征提取

对于相关性十分高的特征，可以直接使用，对于相关性不太高的特征，则可以采取合并、转换等操作，如将房价相近的 Neighborhood 中合并，将 OverallQual 和 OverallCond 合并成新的特征等。对于连续性特征，可以采用切分区段的方式，对于有序的分类特征，可以映射为数值，如 ExterQual，对于普通分类特征，可以使用 One Hot 方式（如 MSSubClass）编码。

### 模型选择

可以试验对比多个模型的效果选择最终模型，也可以根据经验和数据分布类型等选择候选模型。这里使用了 XGBoost 模型，参数的调优可以参考这里。

### 预测

最后使用训练好的模型进行预测，预测的准确度除了算法外，主要依赖特征工程的处理，本脚本主要作为示例，成绩大概在 TOP 20% 左右。
