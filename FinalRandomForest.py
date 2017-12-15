# Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

# Illustrates a random forest classifier on the glass data.

import pandas as pd
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from data_util import *

default = pd.read_csv('./data/credit_card_test.csv.')

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
	
# Remove any columns that lack significant relevance to default

#del default['LIMIT_BAL'] # Big error coming back saying 'LIMIT_BAL' is a key-error?
#del default['SEX']
#del default['EDUCATION']
#del default['MARRIAGE']
#del default['AGE']
#del default['PAY_0']
#del default['PAY_1']
#del default['PAY_2']
#del default['PAY_3'] # This one
#del default['PAY_4']
#del default['PAY_5']
#del default['PAY_6']
#del default['BILL_AMT1'] # This one
#del default['BILL_AMT2']
#del default['BILL_AMT3']
#del default['BILL_AMT4']
#del default['BILL_AMT5'] # This one
#del default['BILL_AMT6']
#del default['PAY_AMT1']
#del default['PAY_AMT2']
#del default['PAY_AMT3']
#del default['PAY_AMT4']
#del default['PAY_AMT5']
#del default['PAY_AMT6']
del default['ID']

# Get all non default columns and use as features/predictors.
data_x = default[list(default)[:-2]] 

# Get default column and use as response variable.
data_y = default[list(default)[-1]] 

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [50]
depth = [6]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_binary_classif_error_report(y_test, preds)
		
		
# Now using the Validation Data

default = pd.read_csv('./data/credit_card_validation.csv.')
	
# Remove any columns that lack significant relevance to default

#del default['LIMIT_BAL'] # Big error coming back saying 'LIMIT_BAL' is a key-error?
#del default['SEX']
#del default['EDUCATION']
#del default['MARRIAGE']
#del default['AGE']
#del default['PAY_0']
#del default['PAY_1']
#del default['PAY_2']
#del default['PAY_3'] # This one
#del default['PAY_4']
#del default['PAY_5']
#del default['PAY_6']
#del default['BILL_AMT1'] # This one
#del default['BILL_AMT2']
#del default['BILL_AMT3']
#del default['BILL_AMT4']
#del default['BILL_AMT5'] # This one
#del default['BILL_AMT6']
#del default['PAY_AMT1']
#del default['PAY_AMT2']
#del default['PAY_AMT3']
#del default['PAY_AMT4']
#del default['PAY_AMT5']
#del default['PAY_AMT6']
del default['ID']

# Get all non default columns and use as features/predictors.
data_x = default[list(default)[:-2]] 

# Get default column and use as response variable.
data_y = default[list(default)[-1]] 

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [50]
depth = [6]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_binary_classif_error_report(y_test, preds)