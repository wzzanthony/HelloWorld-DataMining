# coding=utf-8
# in case that program is not able to decode Chinese characters for python's default decode format is ASCII

#import warnings package to ignore warnings。
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
#missingno provides a small toolset of flexible and easy-to-use missing data visualizations and 
#utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset
import scipy.stats as st
import os
import pandas_profiling

# 1) load train set and test set
path = os.getcwd() +  '/datalab/231784/'
Train_data = pd.read_csv(path+'used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(path+'used_car_testA_20200313.csv', sep=' ')
"""
# 2) a brief preview of data
preview_train_data = Train_data.head().append(Train_data.tail())
print("This is the preview of Train_data:")
print(preview_train_data)
print("This is the shape of Train_data: {}".format(Train_data.shape))

preview_test_data = Test_data.head().append(Test_data.tail())
print("This is the preview of Test_data:")
print(preview_test_data)
print("This is the shape of Test_data: {}".format(Test_data.shape))

# 3) data overview
# describe() including total count, mean, std, min and median(25%, 50%, 75%)
# info() print the type of the data <class: Nonetype>
overview_train_data = Train_data.describe()
print("This is the info of Train_data:")
Train_data.info()
print(type(a))
print(type(overview_train_data))
print("this is the overview of Train_data:")
print(overview_train_data)
#print(info_train_data)

overview_test_data = Test_data.describe()
print("This is the info of Test_data:")
Test_data.info()
print("this is the overview of Test_data:")
print(overview_test_data)
#print(info_test_data)

# 4) view missing data or abnormal data
null_train_data = Train_data.isnull().sum()
print("this is the total number of missing data(nan) in different columns of Train_data:")
print(null_train_data)
#data visualization 
missing_visual_data = null_train_data[null_train_data > 0]
missing_visual_data.sort_values(inplace = True)
missing_visual_data.plot.bar()
plt.show()
msno.matrix(Train_data.sample(250))
plt.show()
msno.bar(Train_data.sample(1000))
plt.show()

null_total_train_data = null_train_data.sum()
print("this is the number of missing data of Train_data in total:")
print(null_total_train_data)

null_test_data = Test_data.isnull().sum()
print("This is the total number of missing data(nan) in different columns of Test_data:")
print(null_test_data)
null_total_test_data = null_test_data.sum()
print("this is the number of missing data(nan) of Test_data in total:")
print(null_total_test_data)

#have a brief overview of the column notRepairedDamage, we will find that '-' also represents missing value
print(Train_data['notRepairedDamage'].value_counts())
#replace '-' with nan
Train_data['notRepairedDamage'].replace('-', np.nan, inplace = True)
#check it
print(Train_data['notRepairedDamage'].value_counts())
#The same operation for Test_data
Test_data['notRepairedDamage'].replace('-', np.nan, inplace = True)
#Data Skew 
print(Train_data['seller'].value_counts())
print(Train_data['offerType'].value_counts())
print(Test_data['seller'].value_counts())
print(Test_data['offerType'].value_counts())
#delete these two features
"""
Train_data['notRepairedDamage'].replace('-', np.nan, inplace = True)
Test_data['notRepairedDamage'].replace('-', np.nan, inplace = True)
del Train_data['seller']
del Train_data['offerType']
del Test_data['seller']
del Test_data['offerType']
"""
#check it
print(Train_data.shape)
print(Test_data.shape)

# 5) know the distribution of the prediction value
#From the figure, we can conclude that the best fit is Johnson distribution
price = Train_data['price']

plt.figure(1)
plt.title("Johnson SU")
sns.distplot(price, kde = False, fit = st.johnsonsu)
plt.figure(2)
plt.title("Normal")
sns.distplot(price, kde = False, fit = st.norm)
plt.figure(3)
plt.title("Log Normal")
sns.distplot(price, kde = False, fit = st.lognorm)
plt.show()

#view skewness and kurtosis
sns.distplot(price)
plt.show()
print("Skewness: {}".format(price.skew()))
print("Kurtosis: {}".format(price.kurt()))

print(Train_data.skew())
print(Test_data.kurt())

sns.distplot(Train_data.skew(), color = 'blue', axlabel = 'Skewness')

sns.distplot(Train_data.skew(), color = 'orange', axlabel = 'Kurtness')
plt.show()

#see the specific value
plt.hist(Train_data['price'], orientation = 'vertical', histtype = 'bar', color = 'red')
plt.show()
#from the frequency of the picture, we can conclue that value over 20000 can be treated as abnormal value 
# which can be replaced, transformed, deleted for the number of it is so small
plt.hist(np.log(Train_data['price']), orientation = 'vertical', histtype = 'bar', color ='red') 
plt.show()

# 6) features are divided into categorical feature and numeric feature and we will see the unique distribution of category feature
price = Train_data['price']
#below are the apis that can be called. However, it applies for data without label coding
numeric_features = Train_data.select_dtypes(include = [np.number])
print(numeric_features.columns)

categorical_features = Train_data.select_dtypes(include = [np.object])
print(categorical_features.columns)
"""
numerical_features =['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4',
 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']

"""
# the unique distribution of categorical features
for cat_fea in categorical_features:
    print(cat_fea + "'s unique distribution:")
    print("{} feature hs {} different values".format(cat_fea, Train_data[cat_fea].nunique()))
    print(Train_data[cat_fea].value_counts())
"""
# 7) numerical feature analysis
numerical_features.append('price')

#correlation analysis
price_numeric = Train_data[numerical_features]
correlation = price_numeric.corr()
"""
print(correlation['price'].sort_values(ascending = False), '\n')

figure, ax = plt.subplots(figsize = (7, 7))
plt.title("Correlation of Numberical Features with Price", y = 1, size = 16)
sns.heatmap(correlation, square = True, vmax=0.8)
plt.show()

del price_numeric['price']
#see the features, skew and kurt
for col in numerical_features:
    print('{:15}'.format(col), 
    'Skewness: {:05.2f}'.format(Train_data[col].skew()),
    '    ',
    'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())
    )
#visualize every numerical feature 
figure = pd.melt(Train_data, value_vars=numerical_features)
g = sns.FacetGrid(figure, col = "variable", col_wrap = 2, sharex = False, sharey = False)
g = g.map(sns.distplot, "value")
plt.show()

#visualize the relation between numerical features
sns.set()
columns = ['price', 'v_12', 'v_8', 'v_0', 'power', 'v_5', 'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns], size = 2, kind = 'scatter', diag_kind = 'kde')
plt.show()

#visualize the regression relation between multiple variables
Y_train = Train_data['price']
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
plt.show()
"""

# 8)categorical feature
categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']
#data preprocess
for cat_fea in categorical_features:
    Train_data[cat_fea] = Train_data[cat_fea].astype('category')
    if Train_data[cat_fea].isnull().any():
        Train_data[cat_fea] = Train_data[cat_fea].cat.add_categories(['MISSING'])
        Train_data[cat_fea] = Train_data[cat_fea].fillna("MISSING")
"""
# Box plot visualization
for cat_fea in categorical_features:
    Train_data[cat_fea] = Train_data[cat_fea].astype('category')
    if Train_data[cat_fea].isnull().any():
        Train_data[cat_fea] = Train_data[cat_fea].cat.add_categories(['MISSING'])
        Train_data[cat_fea] = Train_data[cat_fea].fillna("MISSING")

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation = 90)

f = pd.melt(Train_data, id_vars = ['price'], value_vars = categorical_features)
g = sns.FacetGrid(f, col = "variable", col_wrap = 2, sharex = False, sharey = False, size = 5)
g = g.map(boxplot, "value", "price")
plt.show()

# violin plot visualization

target = "price"
for cat_fea in categorical_features:
    sns.violinplot(x = cat_fea, y = target, data = Train_data)
    plt.show()

#bar plot vissualization
def barplot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation = 90)

f = pd.melt(Train_data, id_vars = ['price'], value_vars = categorical_features)
g = sns.FacetGrid(f, col = "variable", col_wrap = 2, sharex = False, sharey = False, size = 5)
g = g.map(barplot, "value", "price")
plt.show()

#visualize the frequency of each category
def countplot(x, **kwargs):
    sns.countplot(x=x)
    x = plt.xticks(rotation=90)

f = pd.melt(Train_data, value_vars = categorical_features)
g = sns.FacetGrid(f, col = "variable", col_wrap = 2, sharex = False, sharey = False, size = 5)
g = g.map(countplot, "value")
plt.show()
"""
# 9)generate one data report through pandas_profiling
pfr = pandas_profiling.ProfileReport(Train_data)
pfr.to_file("./example.html")