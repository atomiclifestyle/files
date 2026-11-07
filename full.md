Here is the Python code for the questions presented in the lab examination paper. Since the actual datasets are not provided, the code uses the filenames mentioned in the questions and assumes standard target column names (like 'Outcome', 'diagnosis', etc.). You may need to adjust the column names to match your specific CSV/Excel files.
Q1 a) Decision Tree Classifier (Diabetes Dataset)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1) # Assuming 'Outcome' is the target
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scikit-learn's CART algorithm supports 'gini' and 'entropy' (Information Gain). 
criteria = ['gini', 'entropy']
best_acc = 0
best_crit = ""

for crit in criteria:
    clf = DecisionTreeClassifier(criterion=crit, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy using {crit}: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_crit = crit

print(f"\nBest criterion is '{best_crit}' with accuracy: {best_acc:.4f}")

Q1 b) Pearson's Correlation
import pandas as pd

# Load data
df_ads = pd.read_excel('advertisement_sales.xlsx')

# Compute Pearson correlation
# Assuming columns are named 'Advertisement Cost' and 'Sales'
correlation = df_ads['Advertisement Cost'].corr(df_ads['Sales'], method='pearson')
print(f"Pearson's correlation coefficient: {correlation:.4f}")

Q2 a) Principal Component Analysis (PCA)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assume df_students is loaded with 5 metrics
# df_students = pd.read_csv('student_metrics.csv')
# X_std = StandardScaler().fit_transform(df_students)

# i. Standardize
# ii. Covariance matrix
# cov_matrix = np.cov(X_std.T)
# print("Covariance Matrix:\n", cov_matrix)

# iii. Eigenvalues and Eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# print("\nEigenvalues:\n", eigenvalues)

# iv. Number of components (Eigenvalue > 1 rule)
# retain_count = sum(eigenvalues > 1)
# print(f"\nComponents to retain (Eigenvalue > 1): {retain_count}")

# v. & vi. Projection using sklearn PCA for simplicity
from sklearn.decomposition import PCA
# pca = PCA(n_components=retain_count)
# X_pca = pca.fit_transform(X_std)
# print("\nComponent Loadings:\n", pca.components_)
# print("\nProjected Data (First 5 rows):\n", X_pca[:5])

Q2 b) K-Means Clustering & Silhouette Score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# df_retail = pd.read_csv('retail.csv')
# X = df_retail[['Age', 'Annual Income', 'Spending Score']]

# ii. Elbow Method
# wcss = []
# for i in range(1, 12):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 12), wcss)
# plt.title('Elbow Method')
# plt.show()

# iii. Evaluate configurations
# k_values = [3, 5, 7, 9, 11]
# best_score = -1
# best_k = 0

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X)
#     score = silhouette_score(X, labels)
#     print(f"Silhouette Score for k={k}: {score:.4f}")
#     if score > best_score:
#         best_score = score
#         best_k = k
# print(f"\nBest clustering configuration is k={best_k}")

Q3 a) MLP and b) SVM Classification
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Assuming X_train, X_test, y_train, y_test are already prepared from the tumor dataset
# MLP Model
mlp = MLPClassifier(random_state=42, max_iter=500)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
print(f"MLP Accuracy: {mlp_acc:.4f}")
print("MLP Confusion Matrix:\n", confusion_matrix(y_test, mlp_pred))

# SVM Model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"\nSVM Accuracy: {svm_acc:.4f}")

Q4 b) Chi-Square Feature Selection (Breast Cancer)
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load Breast Cancer Dataset
# df_bc = pd.read_csv('breast_cancer.csv')
# X = df_bc.drop('diagnosis', axis=1)
# y = df_bc['diagnosis']

# Preprocessing: Handle missing values and scale (Chi2 requires non-negative values)
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_imputed)

# Apply Chi-Square
# chi2_selector = SelectKBest(chi2, k=10)
# chi2_selector.fit(X_scaled, y)

# scores = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi2_selector.scores_})
# top_10 = scores.nlargest(10, 'Chi2 Score')
# print("\nTop 10 Features based on Chi-Square:")
# print(top_10.to_string(index=False))

Q5 a) Linear Regression
from sklearn.linear_model import LinearRegression
import numpy as np

# Assuming X (quantity) and y (sales) are arrays
# X = np.array([...]).reshape(-1, 1)
# y = np.array([...])

# lr = LinearRegression()
# lr.fit(X, y)
# slope = lr.coef_[0]
# intercept = lr.intercept_

# print(f"Slope (m): {slope:.4f}")
# print(f"Intercept (c): {intercept:.4f}")
# print(f"Regression Equation: Y = {slope:.4f}X + {intercept:.4f}")

Q7 a) Random Forest (Titanic) & Q7 b) Agglomerative (Wine)
from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)
# ... accuracy calculation ...
# Feature importance
# importances = rf.feature_importances_
# plt.barh(X.columns, importances)
# plt.show()

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
# clustering = AgglomerativeClustering(n_clusters=3)
# labels = clustering.fit_predict(X_scaled_wine)
# link_matrix = linkage(X_scaled_wine, 'ward')
# dendrogram(link_matrix)
# plt.show()