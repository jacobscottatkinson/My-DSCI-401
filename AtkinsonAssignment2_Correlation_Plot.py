import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing

housing = pd.read_csv('./data/AmesHousingSetA.csv.')

#Correlation Plot

data_a = housing[list(housing)[-1][0:16]]
# Pairs Plot 
sm = pd.plotting.scatter_matrix(data_a, diagonal='kde')
# plt.tight_layout()
plt.show()

#cp = pd.DataFrame({'a':housing[-1], 'b':housing[0], 'c':housing[1], 'd':housing[2], 'e':housing[3], 'f':housing[4], 'g':housing[5],})

#print(cp)

#This was an attempt to view correlation between the columns, unfortunately I could not scale it to see the results well.
# Correction Matrix Plot
names = list(housing[0:16])
data = pd.read_csv('./data/AmesHousingSetA.csv.')
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,15,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
