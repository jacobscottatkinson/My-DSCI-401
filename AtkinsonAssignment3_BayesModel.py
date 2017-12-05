# This illustrates a niave bayesian classifier (Gaussian) on the churn data set.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from data_util import *

churn = pd.read_csv('./data/churn_data.csv.')

#print(churn.head())

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

# Remove any columns that lack significant relevance to the Churn

del churn['CustID']

# Transform Categorical Predictors to One-Hot Encoding
churn = pd.get_dummies(churn, columns=cat_features(churn))

del churn['Churn_No']
del churn['Gender_Female']
del churn['Income_Lower']
#del churn['FamilySize']
del churn['Education']
#del churn['Age']
del churn['Visits'] # Makes no differnece in accuracy when removed but improves the F-1 score.
#del churn['Income_Upper'] # Makes no difference between Lower/Upper Income
#del churn['Gender_Male'] # Makes no difference between Male/Female
#del churn['Calls']

print churn.head()

# Select x and y data
features = list(churn)
features.remove('Churn_Yes')
data_x = churn[features]
data_y = churn['Churn_Yes']

# Convert the different class labels to unique numbers with label encoding.
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_binary_classif_error_report(y_test, preds)

# Illustrate recoding numeric classes back into original (text-based) labels.
y_test_labs = le.inverse_transform(y_test)
pred_labs = le.inverse_transform(preds)
print('(Actual, Predicted): \n' + str(zip(y_test_labs, pred_labs)))



# Now running on the validation data

churn = pd.read_csv('./data/churn_validation.csv.')

# Remove any columns that lack significant relevance to the Churn

del churn['CustID']

# Transform Categorical Predictors to One-Hot Encoding
churn = pd.get_dummies(churn, columns=cat_features(churn))

del churn['Churn_No']
del churn['Gender_Female']
del churn['Income_Lower']
#del churn['FamilySize']
#del churn['Education']
#del churn['Age']
del churn['Visits'] # Makes no differnece in accuracy when removed but improves the F-1 score.
#del churn['Income_Upper'] # Makes no difference between Lower/Upper Income
del churn['Gender_Male'] # Makes no difference between Male/Female
#del churn['Calls']

print churn.head()

# Select x and y data
features = list(churn)
features.remove('Churn_Yes')
data_x = churn[features]
data_y = churn['Churn_Yes']

# Convert the different class labels to unique numbers with label encoding.
data_y = le.fit_transform(data_y)

# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_binary_classif_error_report(y_test, preds)

# Illustrate recoding numeric classes back into original (text-based) labels.
y_test_labs = le.inverse_transform(y_test)
pred_labs = le.inverse_transform(preds)
print('(Actual, Predicted): \n' + str(zip(y_test_labs, pred_labs)))