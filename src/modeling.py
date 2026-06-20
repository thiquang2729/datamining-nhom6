import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================

print("Đang tải dữ liệu...")

X = joblib.load("data/tfidf_features.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
df = pd.read_csv("data/processed_news.csv")

print(f"Số bài báo: {len(df)}")
print(f"Kích thước TF-IDF: {X.shape}")

# =========================
# KMEANS
# =========================

K = 6

print(f"\nĐang chạy KMeans với K = {K}")

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

df["cluster"] = clusters

# =========================
# SILHOUETTE SCORE
# =========================

score = silhouette_score(X, clusters)

print("\n====================")
print("KẾT QUẢ KMEANS")
print("====================")
print("Silhouette Score =", round(score, 4))

# =========================
# TOP KEYWORDS
# =========================

terms = vectorizer.get_feature_names_out()

print("\n====================")
print("TOP KEYWORDS")
print("====================")

for i in range(K):

    center = kmeans.cluster_centers_[i]

    top_ids = center.argsort()[-15:][::-1]

    words = [terms[idx] for idx in top_ids]

    print(f"\nCluster {i}")
    print(", ".join(words))

# =========================
# THỐNG KÊ CLUSTER
# =========================

print("\n====================")
print("SỐ LƯỢNG BÀI")
print("====================")

cluster_counts = df["cluster"].value_counts().sort_index()

print(cluster_counts)

# =========================
# DBSCAN (MẪU NHỎ)
# =========================

print("\n====================")
print("DBSCAN")
print("====================")

sample_size = min(3000, X.shape[0])

X_sample = X[:sample_size]

dbscan = DBSCAN(
    eps=0.7,
    min_samples=5,
    metric="cosine"
)

dbscan_labels = dbscan.fit_predict(X_sample)

n_clusters_dbscan = len(set(dbscan_labels)) - (
    1 if -1 in dbscan_labels else 0
)

n_noise = list(dbscan_labels).count(-1)

print("Số cụm DBSCAN:", n_clusters_dbscan)
print("Số điểm nhiễu:", n_noise)

# =========================
# BIỂU ĐỒ DBSCAN
# =========================

import pandas as pd
import matplotlib.pyplot as plt

dbscan_counts = pd.Series(dbscan_labels).value_counts().sort_index()

plt.figure(figsize=(10, 6))

plt.bar(
    dbscan_counts.index.astype(str),
    dbscan_counts.values,
    width=0.8
)

plt.title("Phan bo cac cum DBSCAN")
plt.xlabel("Cluster")
plt.ylabel("So luong bai bao")

plt.tight_layout()

plt.savefig(
    "notebooks/dbscan_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

# =========================
# GÁN NHÃN THỦ CÔNG
# =========================

cluster_mapping = {
    0: "General_Tech",
    1: "Digital_Transformation",
    2: "Apple_iPhone",
    3: "Samsung_Galaxy",
    4: "Space_Technology",
    5: "AI_and_Digital_Economy"
}

df["label"] = df["cluster"].map(cluster_mapping)

# =========================
# TRAIN / VAL / TEST
# =========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# =========================
# SAVE DATASET
# =========================

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

df.to_csv("data/clustered_news.csv", index=False)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

# =========================
# BIỂU ĐỒ 1
# PHÂN BỐ CLUSTER
# =========================

plt.figure(figsize=(10, 6))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Phan bo bai bao theo Cluster")
plt.xlabel("Cluster")
plt.ylabel("So luong bai bao")

plt.tight_layout()

plt.savefig(
    "notebooks/cluster_distribution.png",
    dpi=300
)

# =========================
# BIỂU ĐỒ 2
# SO SÁNH KMEANS VS DBSCAN
# =========================

comparison = pd.DataFrame({
    "Metric": ["KMeans", "DBSCAN"],
    "Clusters": [
        K,
        n_clusters_dbscan
    ]
})

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Metric"],
    comparison["Clusters"]
)

plt.title("So sanh so cum KMeans va DBSCAN")
plt.ylabel("So cum")

plt.tight_layout()

plt.savefig(
    "notebooks/kmeans_vs_dbscan.png",
    dpi=300
)

plt.show()

# =========================
# TOP 20 TF-IDF
# =========================

import numpy as np

tfidf_mean = np.asarray(X.mean(axis=0)).ravel()

terms = vectorizer.get_feature_names_out()

top_n = 20

top_idx = tfidf_mean.argsort()[-top_n:][::-1]

top_words = [terms[i] for i in top_idx]
top_scores = [tfidf_mean[i] for i in top_idx]

plt.figure(figsize=(10, 7))

plt.barh(top_words[::-1], top_scores[::-1])

plt.title("Top 20 từ/cụm từ nổi bật theo TF-IDF")
plt.xlabel("Điểm TF-IDF trung bình")
plt.ylabel("Từ/cụm từ")

plt.tight_layout()

plt.savefig(
    "notebooks/top20_tfidf.png",
    dpi=300,
    bbox_inches="tight"
)
# =========================
# THÔNG BÁO
# =========================

print("\n====================")
print("ĐÃ LƯU FILE")
print("====================")

print("data/clustered_news.csv")
print("data/train.csv")
print("data/val.csv")
print("data/test.csv")

print("\nBiểu đồ:")
print("notebooks/cluster_distribution.png")
print("notebooks/kmeans_vs_dbscan.png")

print("\nModel:")
print("models/kmeans_model.pkl")