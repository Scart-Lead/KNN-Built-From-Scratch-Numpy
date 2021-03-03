#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##########################
## Data Mining Assignment 
## Luke Carty 116379621
## Mar '21
## KNN Classifier
## Wine Dataset
##########################
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

############
## Just a few comments:
## The KNN and WKNN functions accept the same arguments
## knn(training dataframe, testing dataframe, normalization type, distance type)
## Normalisation type = 0 if uniform, 1 if Min-Max and 2 if Z-Score 
## Distance Type = 1 if Euclidean and 2 if Manhattan
##
## When adding a new dataset, ensure it is split into two, a training and testing df 
## In order to get the correct label/feature set, change the feature names in the functions below
############


# In[ ]:


# Loading the Datasets
df_train = pd.read_csv('C:\\Users\\lukec\\OneDrive\\Documents\\wine-data-project-train.csv')
df_train.head()
df_test = pd.read_csv('C:\\Users\\lukec\\OneDrive\\Documents\\wine-data-project-test.csv')


# In[ ]:


# Functions to split the Datasets into Features and Labels
def getFeatureSet(data_frame):
    return data_frame[['fixed acidity', 'volatile acidity', 'citric acid',
                      'residual sugar','chlorides','free sulfur dioxide',
                      'total sulfur dioxide','density','pH','sulphates', 'alcohol']]

def getLabelSet(data_frame):
    return data_frame['Quality']

# True Y values for sample confusion matrix
ytrue = getLabelSet(df_test)


# In[ ]:


# Distance Functions
def euclideanDistance(df, query_point, norm_type):
    features = getFeatureSet(df)
    
    # What scaling to apply
    if norm_type == 0:
        features = features
    else:
        if norm_type == 1:
            features = min_max_normal(features)
        else:
            if norm_type == 2:
                features = stdnormal(features)

                
    labels = getLabelSet(df)
    distance = []
    
    # Iterate through training set rows, calculating distance between query test point
    for i,j in features.iterrows():
        row = features.iloc[i]
        dist =0
        for k in range(0, len(row)):
            point_dist = np.square(row[k] - query_point[k])
            dist += point_dist
            sqrt_dist = np.sqrt(dist)
        distance.append([sqrt_dist, labels[i]])
        
    return distance

def manhattanDistance(df, query_point, norm_type):
    features = getFeatureSet(df)
    
    # What scaling to apply
    if norm_type == 0:
        features = features
    else:
        if norm_type == 1:
            features = min_max_normal(features)
        else:
            if norm_type == 2:
                features = stdnormal(features)
    labels = getLabelSet(df)
    distance = []
    
    # Iterate through training rows, calculating manhattan distance between query test point
    for i,j in features.iterrows():
        row = features.iloc[i]
        dist = 0
        for k in range(0, len(row)):
            point_dist = (abs(row[k] - query_point[k]))
            dist += point_dist
        distance.append([dist, labels[i]])
    return distance


# In[ ]:


# Function to standardize the Features
def stdnormal(df):
    result = df.copy()
    for column in df.columns:
        result[column] = (df[column] - df[column].mean()) / df[column].std()
    return result


# In[ ]:


# Function to apply min-max scaling to the features 
def min_max_normal(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[ ]:


# Main Uniformly Weighted Function
def knn(train, test, K, norm_type, distanceType):
    pred, train_size = [], len(df_train)
    
    features_test = getFeatureSet(df_test)
    labels_test = getLabelSet(df_test)
    
    # Normalising the test features
    if norm_type == 0:
        features_test = features_test
    else:
        if norm_type == 1:
            features_test = min_max_normal(features_test)
        else:
            if norm_type == 2:
                features_test = stdnormal(features_test)
        
    # Calculating Distance from query point to all training points
    for i,j in features_test.iterrows():
        row = features_test.iloc[i]
        
        if distanceType == 1:
                dist = euclideanDistance(df_train, row, norm_type)
        else:
            if distanceType == 2:
                dist = manhattenDistance(df_train, row, norm_type)
        
        # Determining K-Nearest Neighbors
        k_neighbors = sorted(dist, key=lambda x:x[0])[:K]
        excel = 0.0
        poor = 0.0
        for a in k_neighbors:
            if a[1] == 1:
                excel += 1
            else:
                poor += 1
    
        if excel > poor:
            pred.append(1)
        else:
            pred.append(-1)
            
    correct = 0 
     # Calculating the number of correct classifications   
    for sample in range(0, len(features_test)):
        ytrue = labels_test.iloc[sample]
        pred_label = pred[sample]
        if pred_label == ytrue:
            correct += 1
    return correct


# In[1]:


# Main Weighted KNN Function
def wknn(train, test, K, norm_type, distanceType):
    pred, train_size = [], len(df_train)
    
    features_test = getFeatureSet(df_test)
    labels_test = getLabelSet(df_test)
    
   # Normalizing the test features 
    if norm_type == 0:
        features_test = features_test
    else:
        if norm_type == 1:
            features_test = min_max_normal(features_test)
        else:
            if norm_type == 2:
                features_test = stdnormal(features_test)
            else:
                print('Invalid Norm Type')
        
    # Calculating Distance from query point to all training points
    for i,j in features_test.iterrows():
        row = features_test.iloc[i]
      
   
        if distanceType == 1:
                dist = euclideanDistance(df_train, row, norm_type)
        else:
            if distanceType == 2:
                dist = manhattanDistance(df_train, row, norm_type)
            else:
                print('Invalid Distance Type')
        
        # Determining K-Nearest Neighbors
        k_neighbors = sorted(dist, key=lambda x:x[0])[:K]
        excel = 0.0
        poor = 0.0
       
        # Applying Inverse Distance Squared Weighting   
        for a in k_neighbors:
            if a[1] == 1:
                 excel += (1/np.square(a[0]))
            else:
                poor += (1/np.square(a[0]))
    
        if excel > poor:
            pred.append(1)
        else:
            pred.append(-1)
                
    print(pred)
    correct = 0 
    # calculating number of correct classifications    
    for sample in range(0, len(features_test)):
        ytrue = labels_test.iloc[sample]
        pred_label = pred[sample]
        if pred_label == ytrue:
            correct += 1
    return correct


# In[2]:


# Testing the Classifier
euclidw = wknn(df_train, df_test, 19, 2, 1)
euclid = knn(df_train, df_test, 11, 0, 1)
manhatten = knn(df_train, df_test, 25, 2, 2)
manhattenw = wknn(df_train, df_test, 15, 2, 2)

print('Correct Predictions: %d' % euclid)
print('Accuracy: %.2f%%' % (100*euclid/len(df_test)))

print('Correct Predictions: %d' % euclidw)
print('Accuracy: %.2f%%' % (100*euclidw/len(df_test)))

print('Correct Predictions: %d' % manhatten)
print('Accuracy: %.2f%%' % (100*manhatten/len(df_test)))

print('Correct Predictions: %d' % manhattenw)
print('Accuracy: %.2f%%' % (100*manhattenw/len(df_test)))


# In[ ]:


## Calculating Accuracies for odd values of k from 2-40

# Results of UnWeighted Euclidean Distance with no normalisation
allResults =[]
correct = 0
acc_valuewe = 0.0
n = len(df_test)
for k in range(3,40,2):
    correct = knn(df_train, df_test, k, 0, 1)
    acc_valuewe = ((100*correct) / n)
    allResults.append(acc_valuewe)
    

# Results of Weighted Euclidean Distance with no normalisation
allResults_w =[]
correct_w = 0
acc_value_w = 0.0
n = len(df_test)
for k in range(3,40,2):
    correct_w = wknn(df_train, df_test, k, 0, 1)
    acc_value_w = ((100*correct_w) / n)
    allResults_w.append(acc_value_w)

# Results of UnWeighted Manhattan Distance with no normalisation
allResults_uwm =[]
correct_uwm = 0
acc_value_uwm = 0.0
n = len(df_test)
for k in range(3,40,2):
    correct_uwm = knn(df_train, df_test, k, 0, 2)
    acc_value_uwm = ((100*correct_uwm) / n)
    allResults_uwm.append(acc_value_uwm)

# Results of Weighted Manhattan Distance with no normalisation
allResults_wm =[]
correct_wm = 0
acc_value_wm = 0.0
n = len(df_test)
for k in range(3,40,2):
    correct_wm = wknn(df_train, df_test, k, 0, 2)
    acc_value_wm = ((100*correct_wm) / n)
    allResults_wm.append(acc_value_wm)


# In[ ]:


# Graphing Initial Implementation Results
sns.set_style('darkgrid')
plt.plot( list(range(3, 40, 2)), allResults, label = 'Unweighted Euclidean Distance')
plt.plot( list(range(3,40, 2)), allResults_w, label='Weighted Euclidean Distance')
plt.plot( list(range(3,40, 2)), allResults_uwm, label='Unweigthed Manhattan Distance')
plt.plot( list(range(3,40, 2)), allResults_wm, label='Weighted Manhattan Distance')

plt.ylabel('Accuracy- %')

plt.xlabel('Value for k')

plt.title('Initial Implementation Results')

plt.ylim(50,85)

plt.legend()
plt.show


# In[ ]:


## Calculating Accuracies of Normalized Models with odd values for k 

# Results of UnWeighted Euclidean Distance with std normalisation
allResults_estd = []
correct_estd = 0
acc_value_estd = 0.0
for k in range(3, 40, 2):
    correct_estd = knn(df_train, df_test, k, 2, 1)
    acc_value_estd = ((100*correct_estd) / n)
    allResults_estd.append(acc_value_estd)
    
# Results of UnWeighted Euclidean Distance with min_max normalisation
allResults_emm = []
correct_emm = 0
acc_value_emm = 0.0
for k in range(3, 40, 2):
    correct_emm = knn(df_train, df_test, k, 1, 1)
    acc_value_emm = ((100*correct_emm) / n)
    allResults_emm.append(acc_value_emm)
    
# Results of Weighted Euclidean Distance with std normalisation
allResults_westd = []
correct_westd = 0
acc_value_westd = 0.0
for k in range(3, 40, 2):
    correct_westd = wknn(df_train, df_test, k, 2, 1)
    acc_value_westd = ((100*correct_westd) / n)
    allResults_westd.append(acc_value_westd)
    
# Results of Weighted Euclidean Distance with min_max normalisation
allResults_wem = []
correct_wem = 0
acc_value_wem= 0.0
for k in range(3, 40, 2):
    correct_wem = wknn(df_train, df_test, k, 1, 1)
    acc_value_wem = ((100*correct_wem) / n)
    allResults_wem.append(acc_value_wem)
    
# Results of UnWeighted Manahattan Distance with std normalisation
allResults_uwmstd = []
correct_uwmstd = 0
acc_value_uwmstd = 0.0
for k in range(3, 40, 2):
    correct_uwmstd = knn(df_train, df_test, k, 2, 2)
    acc_value_uwmstd = ((100*correct_uwmstd) / n)
    allResults_uwmstd.append(acc_value_uwmstd)
    
# Results of UnWeighted Manhattan Distance with min_max normalisation
allResults_uwmm = []
correct_uwmm = 0
acc_value_uwmm= 0.0
for k in range(3, 40, 2):
    correct_uwmm = knn(df_train, df_test, k, 1, 2)
    acc_value_uwmm = ((100*correct_uwmm) / n)
    allResults_uwmm.append(acc_value_uwmm)
    
# Results of Weighted Manahattan Distance with std normalisation
allResults_wmstd = []
correct_wmstd = 0
acc_value_wmstd = 0.0
for k in range(3, 40, 2):
    correct_wmstd = wknn(df_train, df_test, k, 2, 2)
    acc_value_wmstd = ((100*correct_wmstd) / n)
    allResults_wmstd.append(acc_value_wmstd)
    
# Results of Weighted Manhattan Distance with min_max normalisation
allResults_wmm = []
correct_wmm = 0
acc_value_wmm= 0.0
for k in range(3, 40, 2):
    correct_wmm = wknn(df_train, df_test, k, 1, 2)
    acc_value_wmm = ((100*correct_wmm) / n)
    allResults_wmm.append(acc_value_wmm)


# In[ ]:


# Unweighted Euclidean Distance Graph
sns.set_style('darkgrid')
plt.plot( list(range(3, 40, 2)), allResults, label='Pure Euclidean Distance')
plt.plot( list(range(3,40,2)), allResults_emm, label='Min-Max Normalized Euclidean Distance')
plt.plot( list(range(3,40,2)), allResults_estd, label='Standard Normalized Euclidean Distance')
plt.xlabel('k-values')

plt.ylabel('Accuracy- %')
plt.ylim(50,90)
plt.title('Unweighted Euclidean Distance')

plt.legend()
plt.show


# In[ ]:


# Weighted Euclidean Distance Graph
sns.set_style('darkgrid')
plt.plot( list(range(3,40, 2)), allResults_w, label='Pure Weighted Euclidean Distance')
plt.plot( list(range(3,40,2)), allResults_wem, label='Min-Max Normalized Weighted Euclidean Distance')
plt.plot( list(range(3,40,2)), allResults_westd, label='Standard Normalized Weighted Euclidean Distance')
plt.xlabel('k-values')

plt.ylabel('Accuracy- %')
plt.ylim(50,90)
plt.title('Weighted Euclidean Distance')

plt.legend()
plt.show     


# In[ ]:


# Unweighted Manhattan Distance Graph
sns.set_style('darkgrid')
plt.plot( list(range(3, 40, 2)), allResults_uwm, label='Pure UnWeighted Manhattan Distance')
plt.plot( list(range(3,40,2)), allResults_uwmm, label='Min-Max Normalized UnWeighted Manhattan Distance')
plt.plot( list(range(3,40,2)), allResults_uwmstd, label='Standard Normalized UnWeighted Manhattan Distance')
plt.xlabel('k-values')

plt.ylabel('Accuracy- %')
plt.ylim(50,90)
plt.title('Unweighted Manhattan Distance')

plt.legend()
plt.show  


# In[ ]:


# Weighted Manhattan Distance Graph
sns.set_style('darkgrid')
plt.plot( list(range(3, 40, 2)), allResults_wm, label='Pure Weighted Manhattan Distance')
plt.plot( list(range(3,40,2)), allResults_wmm, label='Min-Max Normalized Weighted Manhattan Distance')
plt.plot( list(range(3,40,2)), allResults_wmstd, label='Standard Normalized Weighted Manhattan Distance')
plt.xlabel('k-values')

plt.ylabel('Accuracy- %')
plt.ylim(50,90)
plt.title('Weighted Manhattan Distance')

plt.legend()
plt.show  


# In[3]:


# Confusion Matrix for Top Four Performers
manhattenw_k15_predictions = [-1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1]

ytrue = getLabelSet(df_test)
ytrue = ytrue.tolist()

pred = pd.Categorical(manhattenw_predictions, categories=[Excellent, Poor])
actual = pd.Categorical(ytrue, categories=[Excellent,Poor])
conf_matrix = pd.crosstab(pred, actual)
sns.heatmap(conf_matrix, annot=True)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Standard Weighted Manhattan @ K=15')
plt.show()


# In[ ]:


euclid_k19_pred = [-1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1]

pred = pd.Categorical(euclid_k19_pred, categories=[1, -1])
actual = pd.Categorical(ytrue, categories=[1,-1])
conf_matrix = pd.crosstab(pred, actual)
sns.heatmap(conf_matrix, annot=True)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Standard Weighted Euclidean @ K=19')
plt.show()


# In[ ]:


manhattan_k7_preds = [-1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1]

pred = pd.Categorical(manhattan_k7_preds, categories=[1, -1])
actual = pd.Categorical(ytrue, categories=[1,-1])
conf_matrix = pd.crosstab(pred, actual)
sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu')
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Min-Max Weighted Manhattan @ K=7')
plt.show()


# In[ ]:


euclid_k27_preds = [-1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1]

pred = pd.Categorical(euclid_k27_preds, categories=[1, -1])
actual = pd.Categorical(ytrue, categories=[1,-1])
conf_matrix = pd.crosstab(pred, actual)
sns.heatmap(conf_matrix, annot=True, cmap='BuPu')
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Min-Max Weighted Euclidean @ K=27')
plt.show()

