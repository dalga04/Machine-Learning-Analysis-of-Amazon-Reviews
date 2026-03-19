# ==========================================
# Amazon Reviews Analysis Project
# Phase 2 + Phase 3 + Phase 4 (Clustering)
# ==========================================

# 📌 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ==========================================
# 📌 2. Load Data
# ==========================================
df = pd.read_csv("data/amazon.csv")

print("Initial Shape:", df.shape)

# ==========================================
# 📌 3. Phase 2 - EDA
# ==========================================

df["review_content"] = df["review_content"].astype(str)
df["review_length"] = df["review_content"].apply(len)

# Histogram
plt.figure()
plt.hist(df["rating"], bins=5)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.savefig("images/histogram.png")
plt.close()

# Scatter
plt.figure()
plt.scatter(df["review_length"], df["rating"], alpha=0.3)
plt.title("Review Length vs Rating")
plt.xlabel("Review Length")
plt.ylabel("Rating")
plt.savefig("images/scatter.png")
plt.close()

# Boxplot
plt.figure()
plt.boxplot(df["review_length"])
plt.title("Boxplot of Review Length")
plt.savefig("images/boxplot.png")
plt.close()

# ==========================================
# 📌 4. Phase 3 - Data Preprocessing
# ==========================================

# Missing values
df["review_title"] = df["review_title"].fillna("")
df["category"] = df["category"].fillna("Unknown")

# Remove duplicates
df = df.drop_duplicates()

# Remove invalid text
df = df[df["review_content"].notna()]

# Convert rating to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["rating"])

# Outlier handling (IQR)
Q1 = df["review_length"].quantile(0.25)
Q3 = df["review_length"].quantile(0.75)
IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

df["review_length"] = np.where(
    df["review_length"] > upper_bound,
    upper_bound,
    np.where(df["review_length"] < lower_bound, lower_bound, df["review_length"])
)

# ==========================================
# 📌 5. Feature Selection
# ==========================================

features = df[["review_length", "rating"]].copy()

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# ==========================================
# 📌 6. PCA
# ==========================================

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("Explained Variance:", pca.explained_variance_ratio_)

# ==========================================
# 📌 7. Clustering
# ==========================================

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)

# DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=7)
dbscan_labels = dbscan.fit_predict(pca_data)

# ==========================================
# 📌 8. Evaluation
# ==========================================

print("\n=== KMeans Evaluation ===")
print("Silhouette:", silhouette_score(pca_data, kmeans_labels))
print("Davies-Bouldin:", davies_bouldin_score(pca_data, kmeans_labels))
print("Calinski-Harabasz:", calinski_harabasz_score(pca_data, kmeans_labels))

if len(set(dbscan_labels)) > 1:
    print("\n=== DBSCAN Evaluation ===")
    print("Silhouette:", silhouette_score(pca_data, dbscan_labels))
    print("Davies-Bouldin:", davies_bouldin_score(pca_data, dbscan_labels))
    print("Calinski-Harabasz:", calinski_harabasz_score(pca_data, dbscan_labels))
else:
    print("\nDBSCAN could not form meaningful clusters.")

# ==========================================
# 📌 9. Visualization
# ==========================================

plt.figure(figsize=(12,5))

# KMeans
plt.subplot(1,2,1)
plt.scatter(pca_data[:,0], pca_data[:,1], c=kmeans_labels)
plt.title("KMeans Clustering")

# DBSCAN
plt.subplot(1,2,2)
plt.scatter(pca_data[:,0], pca_data[:,1], c=dbscan_labels)
plt.title("DBSCAN Clustering")

plt.tight_layout()
plt.savefig("images/clustering.png")
plt.show()
