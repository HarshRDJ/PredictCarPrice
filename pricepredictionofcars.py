"""
Project : Car price prediction

Description:
This dataset contains information about used cars.
This data can be used for a lot of purposes such as price prediction to exemplify the use of linear regression in Machine Learning.
The columns in the given dataset are as follows:

name
year
selling_price
km_driven
fuel
seller_type
transmission
Owner

"""

# importing the packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import dataset
df = pd.read_csv('car data.csv')
#print(df.head(2))
#print(df.shape)
#print(df['Seller_Type'].unique())
#print(df['Transmission'].unique())
#print(df['Owner'].unique())

# checking missing or null values
#print(df.isnull().sum())

#print(df.columns)
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset['Current_Year']=2020
#print(final_dataset.head(2))

final_dataset['no_of_years_old'] = final_dataset['Current_Year'] - final_dataset['Year']
#print(final_dataset.head(2))

final_dataset.drop(['Year'], axis=1, inplace=True)
final_dataset.drop(['Current_Year'], axis=1, inplace=True)
#print(final_dataset.head(2))

final_dataset = pd.get_dummies(final_dataset, drop_first=True)
#print(final_dataset.head(5))
#print(final_dataset.columns)

#print(final_dataset.corr())
#sns.pairplot(final_dataset,height=1)
#plt.show()

"""
# plot the heat map for some more information
corrmat = final_dataset.corr()
top_corr_feature = corrmat.index
plt.figure(figsize=(20,20))
h = sns.heatmap(final_dataset[top_corr_feature].corr(), annot=True, cmap='RdYlGn')
#plt.show()
"""

# independent and dependent feature
X = final_dataset.iloc[:, 1:]
y = final_dataset.iloc[:, 0]

# feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
#print(model.feature_importances_)

# plot graph of feature importance for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
#plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#print(X_train.shape)

# Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
#print(n_estimators)

# Randomize Search CV
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               }
#print(random_grid)

def random_search():
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
    rf_random.fit(X_train, y_train)
    return rf_random


predictions =  random_search().predict(X_test)
print(predictions)
sns.displot(y_test - predictions)
plt.show()



plt.scatter(y_test, predictions)
plt.show()

rf_random = random_search()

import pickle
# open a file where you want to store a data
file = open('random_forest_regression_model.pkl', 'wb')
# dump information to that file
pickle.dump(rf_random, file)
