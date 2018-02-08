import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


pd.options.mode.chained_assignment = None

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

print('Train dataset shape: ', train_df.shape)
print('Test dataset shape: ', test_df.shape)

combine = [train_df, test_df]

# Embarked deal with missing values
mode_embarked = train_df['Embarked'].mode()[0]
for d in combine:
    d['Embarked'].fillna(mode_embarked, inplace=True)
# Cabin deal with missing values
for d in combine:
    d['Cabin'].fillna('N', inplace=True)
    d['Cabin'] = d['Cabin'].apply(lambda x: x[0])
# Cabin only exists in train dataset
train_df.drop(339, inplace=True)

# replace sex
for d in combine:
    d['Sex'] = d['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Fare deal with missing values
mean_fare = train_df['Fare'].dropna().median()
for d in combine:
    d['Fare'].fillna(mean_fare, inplace=True)
# Fare bin
fare_bin_num = 10
for d in combine:
    d['FareBin'] = pd.cut(d['Fare'], fare_bin_num, labels=[i for i in range(fare_bin_num)])
    d['FareBin'] = d['FareBin'].astype('int')

# extract title
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for d in combine:
    d['Title'] = d['Name'].str.extract('(\w+)\.', expand=False)
    d['Title'] = d['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')
    d['Title'] = d['Title'].map(title_mapping)


# impute age by title
imputed_age_df = train_df[['Age', 'Title']].append(test_df[['Age', 'Title']]).groupby('Title').median().reset_index()
for d in combine:
    for t in imputed_age_df['Title']:
        mage = imputed_age_df[imputed_age_df['Title'] == t]['Age']
        d.loc[(d['Age'].isnull()) & (d['Title'] == t), 'Age'] = mage.values[0]
# Age bin
age_bin_num = 8
for d in combine:
    d['AgeBin'] = pd.cut(d['Age'], age_bin_num, labels=[i for i in range(age_bin_num)])
    d['AgeBin'] = d['AgeBin'].astype('int')


# extract name length
def name_length_group(x):
    if len(x) < 20:
        return 0
    elif 20 < len(x) < 35:
        return 1
    elif 35 < len(x) < 45:
        return 2
    else:
        return 3


for d in combine:
    d['NameLen'] = d['Name'].apply(name_length_group)

# has cabin
for d in combine:
    d['HasCabin'] = [0 if i == 'N' else 1 for i in d['Cabin']]
# cabin encode
cabin_encoder = LabelEncoder()
for d in combine:
    d['Cabin'] = cabin_encoder.fit_transform(d['Cabin'])
# extract family size
for d in combine:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1


# extract family group
def family_group(size):
    a = ''
    if size <= 1:
        a = 'loner'
    elif size <= 4:
        a = 'small'
    else:
        a = 'large'
    return a


for d in combine:
    d['FamilyGroup'] = d['FamilySize'].apply(family_group)
# extract is alone
for d in combine:
    d['isAlone'] = [1 if i < 2 else 0 for i in d['FamilySize']]

# extract family survive
default_survival = 0.5
for d in combine:
    d['LastName'] = d['Name'].apply(lambda x: str.split(x, ",")[0])
    d['Family_Survival'] = default_survival
    d['Fare'] = d['Fare'].astype('str')

for grp, grp_df in train_df[train_df['isAlone'] == 0][['Survived', 'LastName', 'Fare', 'PassengerId']].groupby(['LastName', 'Fare']):
    if len(grp_df) != 1:
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if smax == 1.0:
                train_df.loc[train_df['PassengerId'] == passID, 'Family_Survival'] = 1
                test_df.loc[(test_df['LastName'] == grp[0]) & (test_df['Fare'] == grp[1]), 'Family_Survival'] = 1
            elif smin == 0.0:
                train_df.loc[train_df['PassengerId'] == passID, 'Family_Survival'] = 0
                test_df.loc[(test_df['LastName'] == grp[0]) & (test_df['Fare'] == grp[1]), 'Family_Survival'] = 0

for d in combine:
    d['Fare'] = d['Fare'].astype('float')

# dummies
dummies_features = ['Title', 'Cabin', 'Embarked']
train_df = pd.get_dummies(train_df, columns=dummies_features)
test_df = pd.get_dummies(test_df, columns=dummies_features)

# drop unused features
drop_features = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'NameLen', 'LastName', 'FamilyGroup']
train_set = train_df.drop(drop_features, axis=1)
test_set = test_df.drop(drop_features, axis=1)

print('Train dataset shape: ', train_set.shape)
print('Test dataset shape: ', test_set.shape)

# prepare datasets
train_y = train_set['Survived']
train_x = train_set.drop('Survived', axis=1)
# train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.25)
test_x = train_x
test_y = train_y

# cv for xgboost
n_estimators = 1000
params = {
    'eta': 0.01,
    'objective': 'binary:logistic',
    'subsample': 1,
    'colsample_bytree': 1,
    'min_child_weight': 1.1,
    'max_depth': 5,
    'tree_method': 'exact',
}
xgtrain = xgb.DMatrix(train_x, label=train_y)
cvresult = xgb.cv(params, xgtrain, num_boost_round=n_estimators, nfold=5, metrics='auc', seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(50)
        ])
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

# train
params = {
    'n_estimators': num_round_best,
    'learning_rate': 0.01,
    'objective': 'binary:logistic',
    'subsample': 1,
    'colsample_bytree': 1,
    'min_child_weight': 1.1,
    'max_depth': 5,
    'gamma': 0,
    # 'reg_alpha': 1,
    # 'reg_lambda': 10,
    'tree_method': 'exact'
}
model = XGBClassifier(**params)

# params = {
#     'objective': 'binary',
#     'num_leaves': 31,
#     'learning_rate': 0.01,
#     'n_estimators': 1000,
#     'min_child_weight': 1,
#     'min_child_samples': 20,
#     # 'reg_alpha': 1,
#     'reg_lambda': 1,
# }
# model = LGBMClassifier(**params)

# cross validation
scores = cross_val_score(model, train_x, train_y, cv=5)
print('Cross validation scores: ', scores)
print('Mean score: ', np.mean(scores))

model.fit(train_x, train_y)
pred_y = model.predict(test_x)
auc = roc_auc_score(test_y, pred_y)
print('AUC: {0:.3f}'.format(auc))
acc = accuracy_score(test_y, pred_y)
print('Accuracy: {0:.3f}'.format(acc))

# predict
test_pred = model.predict(test_set)
pred = test_df[['PassengerId']]
pred['Survived'] = test_pred
pred.to_csv('output.csv', index=None)
