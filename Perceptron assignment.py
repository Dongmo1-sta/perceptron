#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define Perceptron class
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


# In[8]:



# Load the dataset (ensure you adjust the path or file name as needed)
df = pd.read_csv(r"C:\Users\17063\Downloads\Iris.csv")

# Ensure that 'Class' is mapped to integers and the necessary features are available
df['Class'] = df['Class'].map({'Iris-setosa': -1, 'Iris-versicolor': 1})

# Define the feature combinations
combinations = {
    'Sepal_Length and Petal_Width': ['Sepal_Length', 'Petal_width'],
    'Sepal_Width and Petal_Length': ['Sepal_Width', 'Petal_Length'],
    'Sepal_Width and Petal_Width': ['Sepal_Width', 'Petal_width']
}

# Function to run models for all feature combinations
def run_models_for_combinations(df, combinations):
    for combination_name, feature_columns in combinations.items():
        X = df[feature_columns].values
        y = df['Class'].values
        print(f"Processing combination: {combination_name}")

        # Running Perceptron models with 2-15 iterations
        print(f"Perceptron Model with {combination_name}")
        run_perceptron_model(X, y, list(range(2, 16)))

        # Running Adaline models with Gradient Descent (GD) for 2-15 iterations
        print(f"Adaline GD Model with {combination_name}")
        run_adaline_model(X, y, list(range(2, 16)), model_type="GD")

        # Running Adaline models with Stochastic Gradient Descent (SGD) for 2-15 iterations
        print(f"Adaline SGD Model with {combination_name}")
        run_adaline_model(X, y, list(range(2, 16)), model_type="SGD")

# Run models for all combinations
run_models_for_combinations(df, combinations)


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset (ensure you adjust the path or file name as needed)
df = pd.read_csv(r"C:\Users\17063\Downloads\Iris.csv")

class Perceptron:
    def __init__(self, eta=0.1, n_iter=50, random_state=1):
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
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

# Prepare data using 'Sepal_Width' and 'Petal_Width' for Iris-setosa and Iris-versicolor
def prepare_binary_classification_data(df, class1, class2):
    df_binary = df[df['Class'].isin([class1, class2])]
    X = df_binary[['Sepal_Width', 'Petal_width']].values
    y = np.where(df_binary['Class'] == class1, -1, 1)
    return X, y

# Running the Perceptron model
X, y = prepare_binary_classification_data(df, 'Iris-setosa', 'Iris-versicolor')

for n_iter in range(2, 17):
    ppn = Perceptron(eta=0.1, n_iter=n_iter)
    ppn.fit(X, y)
    plt.figure()
    plot_decision_regions(X, y, classifier=ppn)
    plt.title(f'Perceptron (n_iter={n_iter}) - Iris-setosa vs Iris-versicolor (Sepal_Width and Petal_Width)')
    plt.xlabel('Sepal Width [cm]')
    plt.ylabel('Petal Width [cm]')
    plt.legend(loc='upper left')
    plt.show()


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset (ensure you adjust the path or file name as needed)
df = pd.read_csv(r"C:\Users\17063\Downloads\Iris.csv")

class AdalineGD:
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
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

# Standardizing the features for Adaline
def standardize(X):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std

# Running the Adaline model
X, y = prepare_binary_classification_data(df, 'Iris-setosa', 'Iris-versicolor')
X_std = standardize(X)

for n_iter in range(2, 17):
    ada = AdalineGD(n_iter=n_iter, eta=0.01)
    ada.fit(X_std, y)
    plt.figure()
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title(f'Adaline (n_iter={n_iter}) - Iris-setosa vs Iris-versicolor (Sepal_Width and Petal_Width)')
    plt.xlabel('Sepal Width [standardized]')
    plt.ylabel('Petal Width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset (ensure you adjust the path or file name as needed)
df = pd.read_csv(r"C:\Users\17063\Downloads\Iris.csv")

# Function to prepare data for different species combinations
def prepare_binary_classification_data_species(df, class1, class2, feature1='Sepal_Width', feature2='Petal_width'):
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
        plt.xlabel('Sepal Width [cm]')
        plt.ylabel('Petal Width [cm]')
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
        plt.xlabel('Sepal Width [standardized]')
        plt.ylabel('Petal Width [standardized]')
        plt.legend(loc='upper left')
        plt.show()

# Example runs for different species combinations
run_models_species_combinations(df, 'Iris-setosa', 'Iris-versicolor')
run_models_species_combinations(df, 'Iris-versicolor', 'Iris-virginica')
run_models_species_combinations(df, 'Iris-setosa', 'Iris-virginica')


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset (ensure you adjust the path or file name as needed)
df = pd.read_csv(r"C:\Users\17063\Downloads\Iris.csv")

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


# In[ ]:




