# Illustrates 2-class logistic regression.

import pandas as pd
from sklearn import linear_model
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

# Transform Categorical Predictors to One-Hot Encoding
churn = pd.get_dummies(churn, columns=cat_features(churn))

#del churn['Churn_No']
#del churn['Gender_Female']
#del churn['Income_Lower']
#del churn['FamilySize']
#del churn['Education']
#del churn['Age']
#del churn['Visits']
#del churn['Income_Upper']
#del churn['Gender_Male']
#del churn['Calls']


#print churn.head()

# Get all non Churn columns and use as features/predictors.
data_x = churn[list(churn)[:-2]] 

# Get Churn_Yes column and use as response variable.
data_y = churn[list(churn)[-1]] 

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build the model.
log_mod = linear_model.LogisticRegression()
log_mod.fit(x_train, y_train)

# Make predictions - both class labels and predicted probabilities.
preds = log_mod.predict(x_test)
pred_probs = log_mod.predict_proba(x_test)
prob_pos = pred_probs.transpose()[1]  # P(X = 1) is column 1
prob_neg = pred_probs.transpose()[0]  # P(X = 0) is column 0

#print(pred_probs)

# Print out common error metrics for the binary classifications.
def print_binary_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

# Look at results.
pred_df = pd.DataFrame({'Actual':y_test, 'Predicted Class':preds, 'P(1)':prob_pos, 'P(0)':prob_neg})
print(pred_df.head(15))
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

#del churn['Churn_No']
#del churn['Gender_Female']
#del churn['Income_Lower']
#del churn['FamilySize']
#del churn['Education']
#del churn['Age']
#del churn['Visits']
#del churn['Income_Upper']
#del churn['Gender_Male']
#del churn['Calls']


#print churn.head()

# Get all non Churn columns and use as features/predictors.
data_x = churn[list(churn)[:-2]] 

# Get Churn_Yes column and use as response variable.
data_y = churn[list(churn)[-1]] 

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build the model.
log_mod = linear_model.LogisticRegression()
log_mod.fit(x_train, y_train)

# Make predictions - both class labels and predicted probabilities.
preds = log_mod.predict(x_test)
pred_probs = log_mod.predict_proba(x_test)
prob_pos = pred_probs.transpose()[1]  # P(X = 1) is column 1
prob_neg = pred_probs.transpose()[0]  # P(X = 0) is column 0

#print(pred_probs)

# Print out common error metrics for the binary classifications.
def print_binary_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

# Look at results.
pred_df = pd.DataFrame({'Actual':y_test, 'Predicted Class':preds, 'P(1)':prob_pos, 'P(0)':prob_neg})
print(pred_df.head(15))
print('Accuracy: ' + str(accuracy_score(y_test, preds)))
print('Precison: ' + str(precision_score(y_test, preds)))
print('Recall: ' + str(recall_score(y_test, preds)))
print('F1: ' + str(f1_score(y_test, preds)))
print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))