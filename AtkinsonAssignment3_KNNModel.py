# Illustrates k-nearest neighbors on the web churn data.

import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

churn = pd.read_csv('./data/churn_data.csv.')

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

# Do a pairs plot to see potential relationships
pp = pd.plotting.scatter_matrix(churn, diagonal='kde')
#plt.tight_layout()
#plt.show()

# Transform Categorical Predictors to One-Hot Encoding
churn = pd.get_dummies(churn, columns=cat_features(churn))

del churn['Churn_No']
#del churn['Gender_Female']
#del churn['Income_Lower']
#del churn['FamilySize']
#del churn['Education']
#del churn['Age']
#del churn['Visits']
#del churn['Income_Upper']
del churn['Gender_Male']
#print churn.head()

# Get all non Churn columns and use as features/predictors.
data_x = churn[list(churn)[:-2]] 

# Get Churn_Yes column and use as response variable.
data_y = churn[list(churn)[-1]] 

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for k = 2, 4, 6, 8, ..., 20.
ks = [2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 16, 18, 20]
for k in ks:
	# Create model and fit.
	mod = neighbors.KNeighborsClassifier(n_neighbors=k)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('---------- EVALUATING MODEL: k = ' + str(k) + ' -------------------')
	# Look at results.
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

	
# Now running on the validation data

churn = pd.read_csv('./data/churn_validation.csv.')

# Remove any columns that lack significant relevance to the Churn

del churn['CustID']


# Transform Categorical Predictors to One-Hot Encoding
churn = pd.get_dummies(churn, columns=cat_features(churn))

del churn['Churn_No']
#del churn['Gender_Female']
#del churn['Income_Lower']
#del churn['FamilySize']
#del churn['Education']
#del churn['Age']
#del churn['Visits']
#del churn['Income_Upper']
del churn['Gender_Male']
#print churn.head()

# Get all non Churn columns and use as features/predictors.
data_x = churn[list(churn)[:-2]] 

# Get Churn_Yes column and use as response variable.
data_y = churn[list(churn)[-1]] 

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for k = 2, 4, 6, 8, ..., 20.
ks = [2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 16, 18, 20]
for k in ks:
	# Create model and fit.
	mod = neighbors.KNeighborsClassifier(n_neighbors=k)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('---------- EVALUATING MODEL: k = ' + str(k) + ' -------------------')
	# Look at results.
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))