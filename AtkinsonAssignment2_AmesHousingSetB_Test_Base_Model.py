# Assignment 2 - Base Regression Model

import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

housing = pd.read_csv('./data/AmesHousingSetA.csv.')

# Get a list of the categorical features for a given dataframe. 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. 	
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)
	
# Remove any columns that lack significant relevance to the Sale Price.
del housing['PID']
del housing['Street']

# Transform Categorical Predictors to One-Hot Encoding
housing = pd.get_dummies(housing, columns=cat_features(housing))

# Get all SalePrice columns and use as features/predictors.
data_x = housing[list(housing)[:-2]] 

# Get SalePrice column and use as response variable.
data_y = housing[list(housing)[-1]] 

# Imputing the column means for missing values (strategy=mean) by column (axis=0. axis=1 means by row).
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x = imp.fit_transform(data_x)

# Split data into  train/test sets
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)

# Fit and test regression model.

# Create a least squares linear regression model.
base_mod = linear_model.LinearRegression()
base_mod.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = base_mod.predict(x_test)
print('R^2 (Base Model): ' + str(r2_score(y_test, preds)))

# Test Base Model on Housing Set B

housing = pd.read_csv('./data/AmesHousingSetB.csv.')

# Remove any columns that lack significant relevance to the Sale Price.
del housing['PID']
del housing['Street']

# Transform Categorical Predictors to One-Hot Encoding
housing = pd.get_dummies(housing, columns=cat_features(housing))

# Get all SalePrice columns and use as features/predictors.
data_x = housing[list(housing)[:-2]] 

# Get SalePrice column and use as response variable.
data_y = housing[list(housing)[-1]] 

# Imputing the column means for missing values (strategy=mean) by column (axis=0. axis=1 means by row).
data_x = imp.fit_transform(data_x)

# Split data into  train/test sets
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)

# Fit and test regression model.

# Create a least squares linear regression model.
base_mod = linear_model.LinearRegression()
base_mod.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = base_mod.predict(x_test)
print('R^2 (Base Validation Model): ' + str(r2_score(y_test, preds)))