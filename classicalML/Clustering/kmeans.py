# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
iris = datasets.load_iris()
data = iris.data

# It's a good practice to scale the data for KMeans
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add clusters to dataset for visualization
iris_df = pd.DataFrame(scaled_data, columns=iris.feature_names)
iris_df["Cluster"] = clusters

# Visualize the clusters
# Using only two features for visualization: sepal length and sepal width
plt.figure(figsize=(10, 6))
sns.scatterplot(x=iris_df['sepal length (cm)'], y=iris_df['sepal width (cm)'], hue=iris_df["Cluster"], palette="deep", edgecolor='w', s=150)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title("KMeans Clustering of Iris Dataset")
plt.legend()
plt.show()

# Check silhouette score (optional, for cluster validation)
score = silhouette_score(scaled_data, kmeans.labels_, metric='euclidean')
print(f"Silhouette Score: {score:.2f}")

