# Classic ML Algorithms

# Supervised Learning - The target $y$ is known
## 1. Linear Regression - Continious Target $y$ $\to$ Regression Task
$$\hat{y} = \sum_{i=0}^n \beta_i x_i = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$$
1. `Generated_Data_Linear_Regression.ipynb` - Practicing with Generated Data.
2. `Linear Regression - Practice.ipynb` - Using `DATA/Advertising.csv` dataset.
3. `Linear_Poly Regression - Project.ipynb` - My small project for scientific conference. Using `DATA/Salary Data.csv`

## One way to improve Linear regression is Polynomial regression.
This approach enables the creation of feature combinations (Synergy — where a feature is significant only in conjunction with another) and the raising of features to powers to identify non-linear relationships.

To find the appropriate degree of a polynomial, you can use iteration of degrees in a loop and save the metric (RMSE) for further plotting errors based on the complexity of the model. This approach is implemented in the `Polynomial Regression - Practice.ipynb`

## 2. Logisctic Regression - Categorical Target $y$ $\to$ Classification Task
$$\hat{y} = \frac{1} {1 + e^{-\sum^n_{i=0} \beta_i x_i}}$$
1. `Generated_Data_Logistic_Regression.ipynb` - Practicing with Generated Data.
2. `Logistic Regression - Practice.ipynb` - Using `DATA/hearing_test.csv`.


# Unsupervised Learning - The target $y$ is unknown
## 1. K-Means Clustering.
The main objective of K-Means is to find the centroids that minimize the squared distance between each data point and it's assigned cluster.

**Cost Function (Inertia):**

$$J = \sum_{j=1}^K \sum_{x_i \in C_j} ||x_i - \mu_j||^2$$
1. `K-Means Clustering - Practice.ipynb` - Using `DATA/bank-full.csv` dataset.
2. `K-Means - Naked` - Realization using Python functions only.
