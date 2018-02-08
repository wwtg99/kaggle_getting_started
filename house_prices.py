import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


pd.options.mode.chained_assignment = None

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

print('Load train dataset shape: ', train_df.shape)
print('Load test dataset shape: ', test_df.shape)


def fit_transform_label(field, train_set, test_set):
    encoder = LabelEncoder()
    d1 = train_set[field].astype('str').unique()
    d2 = test_set[field].astype('str').unique()
    d = np.append(d1, d2)
    encoder.fit(d)
    train_set[field] = encoder.transform(train_set[field].astype('str'))
    test_set[field] = encoder.transform(test_set[field].astype('str'))
    return train_set, test_set


def fillna_with_mode(field, train_set, test_set):
    mode = train_set[[field]].append(test_set[[field]])[field].mode()[0]
    train_set[field] = train_set[field].fillna(mode)
    test_set[field] = test_set[field].fillna(mode)
    return train_set, test_set


def fillna_with_zero(field, train_set, test_set):
    train_set[field] = train_set[field].fillna('0')
    test_set[field] = test_set[field].fillna('0')
    return train_set, test_set


def fillna_with_mean(field, train_set, test_set):
    mean = train_set[[field]].append(test_set[[field]])[field].mean()
    train_set[field] = train_set[field].fillna(mean)
    test_set[field] = test_set[field].fillna(mean)
    return train_set, test_set


def check_nan(train_set, test_set):
    combined_set = train_set.append(test_set)
    total = combined_set.isnull().sum().sort_values(ascending=False)
    percent = (combined_set.isnull().sum() / combined_set.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)


# check_nan(train_df, test_df)

# drop outliers
train_df = train_df[train_df['GrLivArea'] < 4500]

# fillna
features_fillna_with_mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional', 'SaleType', 'KitchenQual']
for f in features_fillna_with_mode:
    train_df, test_df = fillna_with_mode(f, train_df, test_df)

features_fillna_with_zero = ['LotShape', 'MasVnrType', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond']
for f in features_fillna_with_zero:
    train_df, test_df = fillna_with_zero(f, train_df, test_df)

features_fillna_with_mean = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
for f in features_fillna_with_mean:
    train_df, test_df = fillna_with_mean(f, train_df, test_df)

# encode
features_label_encoded = ['MSSubClass', 'MSZoning', 'LotConfig', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'SaleType', 'SaleCondition']
for f in features_label_encoded:
    train_df, test_df = fit_transform_label(f, train_df, test_df)

# feature mapping
features_to_mapping = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
ext_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, '0': 0}

for f in features_to_mapping:
    train_df[f].replace(ext_mapping, inplace=True)
    test_df[f].replace(ext_mapping, inplace=True)

spe_mapping = {
    "GarageFinish": {"0": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "BsmtExposure": {"0": 0, 'No': 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"0": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"0": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandContour": {"Low": 1, "HLS": 2, "Bnk": 3, "Lvl": 4},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Street": {"Grvl": 1, "Pave": 2},
    "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "GarageType": {"0": 0, "Detchd": 1, "CarPort": 2, "BuiltIn": 3, "Basment": 4, "Attchd": 5, "2Types": 6},
    "Neighborhood": {"MeadowV": 0, "IDOTRR": 0, "BrDale": 0, "OldTown": 0, "Edwards": 0, "BrkSide": 0, "Sawyer": 0, "Blueste": 1, "SWISU": 1, "NAmes": 1, "NPkVill": 1, "Mitchel": 1, "SawyerW": 1, "Gilbert": 2, "NWAmes": 2, "Blmngtn": 2, "CollgCr": 2, "ClearCr": 2,  "Crawfor": 2, "Veenker": 3, "Somerst": 3, "Timber": 3,  "StoneBr": 3, "NoRidge": 3, "NridgHt": 3}
}
train_df.replace(spe_mapping, inplace=True)
test_df.replace(spe_mapping, inplace=True)

# merge features
features_to_merge = {
    'TotalOveral': ['OverallQual', 'OverallCond'],
    'TotalBsmt': ['BsmtQual', 'BsmtCond'],
    'TotalBsmtFinSF': ['BsmtFinSF1', 'BsmtFinSF2'],
    'TotalExter': ['ExterQual', 'ExterCond'],
    'TotalGarage': ['GarageCond', 'GarageQual', 'GarageFinish'],
    'TotalBsmtFinType': ['BsmtFinType1', 'BsmtFinType2'],
    'TotalLand': ['LandSlope', 'LotShape']
}

for f in features_to_merge:
    traind = 0
    testd = 0
    for k in features_to_merge[f]:
        traind += train_df[k].astype('int')
        testd += test_df[k].astype('int')
    train_df[f] = traind
    test_df[f] = testd
train_df['TotalBathrooms'] = train_df['BsmtFullBath'] + (train_df['BsmtHalfBath'] * 0.5) + train_df['FullBath'] + (train_df['HalfBath'] * 0.5)
test_df['TotalBathrooms'] = test_df['BsmtFullBath'] + (test_df['BsmtHalfBath'] * 0.5) + test_df['FullBath'] + (test_df['HalfBath'] * 0.5)

# drop unused features
predict_df = test_df[['Id']]

features_to_drop = ['Id', 'Fence', 'MiscFeature', 'PoolQC', 'Alley']

train_df.drop(features_to_drop, axis=1, inplace=True)
test_df.drop(features_to_drop, axis=1, inplace=True)

# dummies
num_train = train_df.shape[0]
num_test = test_df.shape[0]
dummies_features = ['MSSubClass', 'MSZoning', 'LotConfig', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'SaleType', 'SaleCondition']
combined_df = train_df.append(test_df)
combined_df = pd.get_dummies(combined_df, columns=dummies_features)
train_df = combined_df.iloc[:num_train]
test_df = combined_df.iloc[num_train:].drop('SalePrice', axis=1)

print('Train shape: ', train_df.shape)
print('Test shape: ', test_df.shape)

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

num_rounds = 3000
params = {
    'eta': 0.01,
    'objective': 'reg:linear',
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 1.1,
    'max_depth': 5,
}

dt = xgb.DMatrix(X, label=y)
cv = xgb.cv(params, dt, num_boost_round=num_rounds, nfold=5, early_stopping_rounds=30, metrics='rmse', callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(30)
        ])
num_rounds = cv.shape[0] - 1
print('Best rounds: ', num_rounds)

params = {
    'n_estimators': num_rounds,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 1.1,
    'max_depth': 5,
}

model = XGBRegressor(**params)

print('Starting Cross Validation...')
score = cross_val_score(model, X, y, cv=5)
print('Score: ', score)
print('Mean CV scores: ', np.mean(score))

print('Training...')
model.fit(X, y)
print('Predicting...')
pred_y = model.predict(test_df)
predict_df['SalePrice'] = pred_y
predict_df.to_csv('output.csv', index=None)
