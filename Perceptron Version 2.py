#!/usr/bin/env python
# coding: utf-8

# ##MODEL SETUP: In this phase we will set up the requirments for this program
# 

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# #PERCEPTRON API

# In[20]:


#Define Perceptron class
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# In[21]:


#ADALINE API -Gradient Descent


# In[22]:


# Define AdalineGD class (Gradient Descent)
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# In[23]:


#ADALINE API -Stchastic 


# In[24]:



# Define AdalineSGD class (Stochastic Gradient Descent)
class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# Function to plot decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

# Function to run and visualize models with 2-15 iterations
def run_perceptron_model(X, y, n_iter_list):
    for n_iter in n_iter_list:
        ppn = Perceptron(eta=0.1, n_iter=n_iter)
        ppn.fit(X, y)
        plt.figure()
        plot_decision_regions(X, y, classifier=ppn)
        plt.title(f'Perceptron Model with {n_iter} iterations')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper left')
        plt.show()

def run_adaline_model(X, y, n_iter_list, model_type="GD"):
    for n_iter in n_iter_list:
        if model_type == "GD":
            model = AdalineGD(n_iter=n_iter, eta=0.01)
        else:
            model = AdalineSGD(n_iter=n_iter, eta=0.01, random_state=1)

        model.fit(X, y)
        plt.figure()
        plot_decision_regions(X, y, classifier=model)
        plt.title(f'Adaline {model_type} with {n_iter} iterations')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper left')
        plt.show()


# ## Loaded Data
# The assignment asks us to load the iris data set for analysis. 

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset (ensure you adjust the path or file name as needed)
df = pd.read_csv(r"C:\Users\17063\Downloads\Iris.csv")
print(df.head())

print(df.tail())


# # Prelimanary Data Analysis The assignment requires us to identify two class and to find feature sets in groups of 2,3, and 4 which are linearly and non-linearly seperated. To perofrm this task i compared all 3 classes, aginst their features using a scatter plot. We can see from the scatter plots that Iris-setosa is linearly seperated from verisocolor and virginica. However versicolor and virginica in not linearly seperate, as it show severl instances of overlap

# In[41]:


df['Class'].value_counts()


# # Plot Data Across 2 Class, and 4 Features

# In[43]:


import matplotlib.pyplot as plt

# Plot Data Across 2 Classes and 4 Features
DA_Fig, DA_Axs = plt.subplots(4, 4, figsize=(15, 15))

# Define labels for each axis
y_labels = ['Sepal Length [cm]', 'Sepal Width [cm]', 'Petal Length [cm]', 'Petal Width [cm]']
x_labels = ['Sepal Length [cm]', 'Sepal Width [cm]', 'Petal Length [cm]', 'Petal Width [cm]']

# Set the labels for y-axis
for i in range(4):
    DA_Axs[i, 0].set_ylabel(y_labels[i])

# Set the labels for x-axis
for i in range(4):
    DA_Axs[3, i].set_xlabel(x_labels[i])

# Define species and colors
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['blue', 'orange', 'green']
markers = ['o', 'X', '^']
ms = 10

# Define feature pairs for scatter plots
features = [
    ("Sepal_Length", "Sepal_Length"), ("Sepal_Width", "Sepal_Length"), ("Petal_Length", "Sepal_Length"), ("Petal_Width", "Sepal_Length"),
    ("Sepal_Length", "Sepal_Width"), ("Sepal_Width", "Sepal_Width"), ("Petal_Length", "Sepal_Width"), ("Petal_Width", "Sepal_Width"),
    ("Sepal_Length", "Petal_Length"), ("Sepal_Width", "Petal_Length"), ("Petal_Length", "Petal_Length"), ("Petal_Width", "Petal_Length"),
    ("Sepal_Length", "Petal_Width"), ("Sepal_Width", "Petal_Width"), ("Petal_Length", "Petal_Width"), ("Petal_Width", "Petal_Width")
]

# Iterate over each subplot and plot data for each class
for i, (x_feature, y_feature) in enumerate(features):
    row, col = divmod(i, 4)
    for cls, color, marker in zip(species, colors, markers):
        DA_Axs[row, col].scatter(df.loc[df['Class'] == cls, x_feature], 
                                df.loc[df['Class'] == cls, y_feature], 
                                ms,
                                color=color, 
                                marker=marker, 
                                label=cls if i == 0 else "")

# Add a legend to the plot
handles, labels = DA_Axs[0, 0].get_legend_handles_labels()
DA_Fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the legend
plt.show()


# ##Plotting Perceptron and Adeline Model

# In[27]:



# Function to prepare data for different species combinations
def prepare_binary_classification_data_species(df, class1, class2, feature1='Sepal_Length', feature2='Petal_Length'):
    df_binary = df[df['Class'].isin([class1, class2])]
    X = df_binary[[feature1, feature2]].values
    y = np.where(df_binary['Class'] == class1, -1, 1)
    return X, y

# Function to run Perceptron and Adaline models for given species
def run_models_species_combinations(df, class1, class2):
    X, y = prepare_binary_classification_data_species(df, class1, class2)

    # Perceptron model
    for n_iter in range(2, 17):
        ppn = Perceptron(eta=0.1, n_iter=n_iter)
        ppn.fit(X, y)
        plt.figure()
        plot_decision_regions(X, y, classifier=ppn)
        plt.title(f'Perceptron (n_iter={n_iter}) - {class1} vs {class2}')
        plt.xlabel('Sepal Length [cm]')
        plt.ylabel('Petal Length [cm]')
        plt.legend(loc='upper left')
        plt.show()

    # Standardizing the features for Adaline
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # Adaline model
    for n_iter in range(2, 17):
        ada = AdalineGD(n_iter=n_iter, eta=0.01)
        ada.fit(X_std, y)
        plt.figure()
        plot_decision_regions(X_std, y, classifier=ada)
        plt.title(f'Adaline (n_iter={n_iter}) - {class1} vs {class2}')
        plt.xlabel('Sepal Length [standardized]')
        plt.ylabel('Petal Length [standardized]')
        plt.legend(loc='upper left')
        plt.show()

# Example runs for different species combinations
run_models_species_combinations(df, 'Iris-setosa', 'Iris-versicolor')
run_models_species_combinations(df, 'Iris-versicolor', 'Iris-virginica')
run_models_species_combinations(df, 'Iris-setosa', 'Iris-virginica')


# #Data prperation:The following code extracts the setos data and versicolor data as two seperate data frames sample each dataset to create a training and test set, using an 70/30 split. combine the training data for both classes into one training data set, combine the test data for both classes into as single test set.  columns to the test data.
# 

# In[33]:


# Create a Class A Data Set from Iris-Setosa Data
T1_DataSet_ClassA = df.loc[(df['Class'] == 'Iris-setosa')]
# I added the class discrimenator to the dataset
T1_DataSet_ClassA['Classs'] = -1

# Randomly Sample 70% of Class A Data Set for Training 
T1_DataSet_ClassA_Train = T1_DataSet_ClassA.sample(frac=0.7,random_state=200)

# Create Test Data Set A by removing the training set 
T1_DataSet_ClassA_Test = T1_DataSet_ClassA.drop(T1_DataSet_ClassA_Train.index)

#Add columns to hold test results
T1_DataSet_ClassA_Test['T1-2D-PNN-Result'] = 0
T1_DataSet_ClassA_Test['T1-3D-PNN-Result'] = 0
T1_DataSet_ClassA_Test['T1-4D-PNN-Result'] = 0
T1_DataSet_ClassA_Test['T1-2D-ADL-Result'] = 0
T1_DataSet_ClassA_Test['T1-3D-ADL-Result'] = 0
T1_DataSet_ClassA_Test['T1-4D-ADL-Result'] = 0


# Create a Class B Data Set from Iris-Versicolor Data
T1_DataSet_ClassB = df.loc[(df['Classs'] == 'Iris-versicolor')]
T1_DataSet_ClassB['Classs'] = 1


# Randomly Sample 80% of Class B Data Set for Training 
T1_DataSet_ClassB_Train = T1_DataSet_ClassB.sample(frac=0.8,random_state=200)


# Create Test Data Set B by removing the training set 
T1_DataSet_ClassB_Test = T1_DataSet_ClassB.drop(T1_DataSet_ClassB_Train.index)

# Add columns to hold test results 
T1_DataSet_ClassB_Test['T1-2D-PNN-Result'] = 0
T1_DataSet_ClassB_Test['T1-3D-PNN-Result'] = 0
T1_DataSet_ClassB_Test['T1-4D-PNN-Result'] = 0
T1_DataSet_ClassB_Test['T1-2D-ADL-Result'] = 0
T1_DataSet_ClassB_Test['T1-3D-ADL-Result'] = 0
T1_DataSet_ClassB_Test['T1-4D-ADL-Result'] = 0


#Combine the Class A and B Training Set
T1_DataSet_Train = pd.concat([T1_DataSet_ClassA_Train,T1_DataSet_ClassB_Train], ignore_index=True, sort=False)

#Combine the Class A and B Test Set
T1_DataSet_Test = pd.concat([T1_DataSet_ClassA_Test,T1_DataSet_ClassB_Test], ignore_index=True, sort=False)


# In[ ]:




