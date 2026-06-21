import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
 
 
# ============================================================
# ĐƯỜNG DẪN
# Tính theo vị trí thực tế của file này (__file__) thay vì "../",
# nên chạy đúng dù gọi từ thư mục gốc dự án hay từ trong src/.
# ============================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
 
DEFAULT_CONFIG = {
    "tfidf_matrix_path": os.path.join(DATA_DIR, "tfidf_features.pkl"),
    "vectorizer_path": os.path.join(MODELS_DIR, "vectorizer.pkl"),
    "processed_data_path": os.path.join(DATA_DIR, "processed_news.csv"),
    "n_clusters": 6,
    "k_search_range": range(2, 11),
    "dbscan_eps": 0.7,
    "dbscan_min_samples": 5,
    "dbscan_metric": "cosine",
    # None = chạy DBSCAN trên TOÀN BỘ dữ liệu (không lấy mẫu).
    # Đặt một số nguyên (vd 3000) nếu máy chạy quá lâu / hết RAM và muốn
    # quay lại lấy mẫu để chạy thử nhanh.
    "dbscan_sample_size": None,
    "random_state": 42,
}
 
# ============================================================
# NHÃN GÁN CHO TỪNG CỤM
#
# !!! QUAN TRỌNG: mapping này được gán dựa trên việc đọc TOP KEYWORDS
# in ra ở lần chạy gần nhất. Nếu chạy lại trên dữ liệu mới, số bài thay
# đổi, hoặc đổi K -> nội dung từng cụm có thể thay đổi thứ tự/ý nghĩa.
# => Luôn đối chiếu lại với file notebooks/cluster_keywords.txt (được
#    lưu tự động mỗi lần chạy) trước khi tin tưởng mapping bên dưới.
#
# Ghi chú thêm: các nhãn hiện tại (theo hãng/chủ đề: Apple_iPhone,
# Samsung_Galaxy, Space_Technology...) CHƯA khớp với khung 5 thể loại
# đã thống nhất ban đầu của nhóm (Phần cứng, Phần mềm, AI, Thiết bị di
# động, An ninh mạng - xem "Nhóm 6 DataMining.md"). Cần họp nhóm để
# quyết định: (a) gộp/đổi tên cụm cho khớp 5 thể loại gốc, hoặc
# (b) cập nhật lại phần Business Understanding trong báo cáo cho khớp
# với nhãn thực tế tìm được từ dữ liệu.
# ============================================================
# ĐÃ ĐỐI CHIẾU với data/cluster_keywords.txt từ lần chạy thực tế trên dữ
# liệu của nhóm (xác nhận bằng tay, đọc từng cụm) - mapping bên dưới khớp
# đúng nội dung từng cụm tại thời điểm này. Nếu chạy lại modeling.py sau
# khi dữ liệu đầu vào (tfidf_features.pkl) thay đổi, PHẢI mở lại
# data/cluster_keywords.txt và kiểm tra lại, vì chỉ số cụm KMeans gán ra
# không cố định theo ý nghĩa - cùng một chủ đề có thể rơi vào số cụm khác.
CLUSTER_MAPPING = {
    0: "Digital_Transformation",
    1: "Apple_iPhone",
    2: "General_Tech",
    3: "Space_Technology",
    4: "Samsung_Galaxy",
    5: "AI_and_Digital_Economy",
}
 
 
def ensure_dirs():
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
 
 
def load_inputs(tfidf_matrix_path, vectorizer_path, processed_data_path):
    print("Đang tải dữ liệu...")
    X = joblib.load(tfidf_matrix_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(processed_data_path)
 
    print(f"Số bài báo: {len(df)}")
    print(f"Kích thước TF-IDF: {X.shape}")
    return df, X, vectorizer
 
 
def plot_elbow(X, k_range, output_path, random_state=42):
    print("\nĐang chạy Elbow Method...")
    inertia = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(X)
        inertia.append(model.inertia_)
 
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return inertia
 
 
def run_kmeans(X, k, random_state=42):
    print(f"\nĐang chạy KMeans với K = {k}")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters
 
 
def evaluate_clusters(X, clusters):
    score = silhouette_score(X, clusters)
    db_score = davies_bouldin_score(X.toarray(), clusters)
 
    print("\n====================")
    print("KẾT QUẢ KMEANS")
    print("====================")
    print("Silhouette Score =", round(score, 4))
    print("Davies-Bouldin Score =", round(db_score, 4))
    return score, db_score
 
 
def get_top_keywords(kmeans, vectorizer, n_clusters, top_n=15):
    terms = vectorizer.get_feature_names_out()
    keywords_per_cluster = {}
 
    print("\n====================")
    print("TOP KEYWORDS")
    print("====================")
 
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_ids = center.argsort()[-top_n:][::-1]
        words = [terms[idx] for idx in top_ids]
        keywords_per_cluster[i] = words
 
        print(f"\nCluster {i}")
        print(", ".join(words))
 
    return keywords_per_cluster
 
 
def save_keywords_report(keywords_per_cluster, cluster_mapping, output_path):
    """Lưu top keywords + nhãn đang gán ra file text.
    Dùng để: (1) làm bằng chứng cho báo cáo, (2) dễ kiểm tra lại xem
    CLUSTER_MAPPING ở trên có còn hợp lý với dữ liệu hiện tại không."""
    lines = []
    for cluster_id, words in keywords_per_cluster.items():
        label = cluster_mapping.get(cluster_id, "CHƯA_GÁN_NHÃN")
        lines.append(f"Cluster {cluster_id} -> {label}")
        lines.append(", ".join(words))
        lines.append("")
 
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
 
    print(f"Đã lưu báo cáo từ khóa: {output_path}")
 
 
def run_dbscan(X, eps, min_samples, metric, sample_size=None, random_state=42):
    """
    sample_size=None (mặc định) -> chạy trên TOÀN BỘ dữ liệu.
 
    Lưu ý chi phí tính toán: với metric="cosine", sklearn không dùng được
    KD-Tree/Ball-Tree (chỉ hỗ trợ vài metric như euclidean), nên sẽ tự động
    chuyển sang thuật toán brute-force: tính khoảng cách giữa MỌI cặp điểm
    -> độ phức tạp O(n^2) cả về thời gian lẫn bộ nhớ. Với n ~ 19.000 bài,
    ma trận khoảng cách trung gian có thể chiếm vài GB RAM và chạy khá lâu.
    Nếu máy không đủ RAM hoặc chạy quá lâu, truyền sample_size (vd 5000) để
    quay lại lấy mẫu ngẫu nhiên thay vì dùng toàn bộ.
    """
    print("\n====================")
    print("DBSCAN")
    print("====================")
 
    if sample_size is None or sample_size >= X.shape[0]:
        print(f"Chạy DBSCAN trên TOÀN BỘ {X.shape[0]} bài báo (không lấy mẫu).")
        print("-> Có thể mất nhiều thời gian / tốn nhiều RAM do brute-force O(n^2).")
        X_sample = X
        sample_idx = None
    else:
        print(f"Lấy mẫu ngẫu nhiên {sample_size}/{X.shape[0]} bài báo để chạy DBSCAN.")
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        sample_idx.sort()
        X_sample = X[sample_idx]
 
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan_labels = dbscan.fit_predict(X_sample)
 
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
 
    print("Số cụm DBSCAN:", n_clusters_dbscan)
    print("Số điểm nhiễu:", n_noise)
 
    return dbscan_labels, n_clusters_dbscan, n_noise, sample_idx
 
 
def plot_dbscan_distribution(dbscan_labels, output_path):
    dbscan_counts = pd.Series(dbscan_labels).value_counts().sort_index()
 
    plt.figure(figsize=(10, 6))
    plt.bar(dbscan_counts.index.astype(str), dbscan_counts.values, width=0.8)
    plt.title("Phan bo cac cum DBSCAN")
    plt.xlabel("Cluster")
    plt.ylabel("So luong bai bao")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
 
 
def plot_cluster_distribution(cluster_counts, output_path):
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
    plt.title("Phan bo bai bao theo Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("So luong bai bao")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
 
 
def plot_kmeans_vs_dbscan(k, n_clusters_dbscan, output_path):
    comparison = pd.DataFrame({
        "Metric": ["KMeans", "DBSCAN"],
        "Clusters": [k, n_clusters_dbscan],
    })
 
    plt.figure(figsize=(8, 5))
    plt.bar(comparison["Metric"], comparison["Clusters"])
    plt.title("So sanh so cum KMeans va DBSCAN")
    plt.ylabel("So cum")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
 
 
def plot_pca(X, clusters, output_path):
    print("\nĐang tạo PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
 
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=5)
    plt.title("PCA Cluster Visualization")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
 
 
def plot_top_tfidf(X, vectorizer, output_path, top_n=20):
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
 
    top_idx = tfidf_mean.argsort()[-top_n:][::-1]
    top_words = [terms[i] for i in top_idx]
    top_scores = [tfidf_mean[i] for i in top_idx]
 
    plt.figure(figsize=(10, 7))
    plt.barh(top_words[::-1], top_scores[::-1])
    plt.title("Top 20 từ/cụm từ nổi bật theo TF-IDF")
    plt.xlabel("Điểm TF-IDF trung bình")
    plt.ylabel("Từ/cụm từ")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
 
 
def safe_to_csv(df, path, **kwargs):
    """Lưu CSV, kèm thông báo lỗi rõ ràng hơn nếu gặp PermissionError.
 
    Lỗi PermissionError (Errno 13) trên Windows rất hay gặp khi file CSV
    đang được mở sẵn ở chương trình khác (Excel, VS Code preview...), khi
    file bị đặt thuộc tính Read-only, hoặc khi thư mục đang được OneDrive
    đồng bộ và tạm khóa file.
    """
    try:
        df.to_csv(path, index=False, **kwargs)
    except PermissionError:
        print(f"\n[LỖI] Không ghi được vào: {path}")
        print("Nguyên nhân thường gặp trên Windows:")
        print("  1. File đang mở sẵn trong Excel / VS Code / chương trình khác.")
        print("  2. File đang bị đặt thuộc tính Read-only.")
        print("  3. Thư mục đang được OneDrive đồng bộ và tạm khóa file.")
        print("=> Đóng file/chương trình đang mở file đó rồi chạy lại:")
        print("   python src/modeling.py")
        raise
 
 
def split_dataset(df, label_column="label", test_size=0.30, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[label_column],
        random_state=random_state,
    )
    return train_df, val_df, test_df
 
 
def cluster_and_label(df, X, vectorizer, config=None, cluster_mapping=None):
    """
    Hàm chính cho bước Phân cụm & Gán nhãn.
 
    Đây là hàm mà src/pipeline.py (step_clustering, hiện đang là
    PLACEHOLDER) có thể import và gọi trực tiếp để tích hợp vào pipeline
    chung của nhóm:
 
        from modeling import cluster_and_label
        df, kmeans, keywords = cluster_and_label(df, X_features, vectorizer)
 
    Tham số:
        df: DataFrame đầu ra bước tiền xử lý NLP (Thành viên 3) - PHẢI
            cùng số dòng và cùng thứ tự với X.
        X: ma trận đặc trưng TF-IDF tương ứng với df.
        vectorizer: TfidfVectorizer đã fit (dùng để lấy tên feature).
        config: dict ghi đè một phần DEFAULT_CONFIG (vd: {"n_clusters": 5}).
        cluster_mapping: dict {cluster_id: nhãn}. Nếu không truyền sẽ dùng
            CLUSTER_MAPPING mặc định trong file. NÊN truyền vào khi áp
            dụng cho dữ liệu mới vì nội dung từng cụm có thể đổi.
 
    Trả về: (df_đã_có_cột_cluster_và_label, kmeans_model, keywords_per_cluster)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    mapping = cluster_mapping or CLUSTER_MAPPING
    k = cfg["n_clusters"]
 
    kmeans, clusters = run_kmeans(X, k=k, random_state=cfg["random_state"])
 
    df = df.copy()
    df["cluster"] = clusters
 
    evaluate_clusters(X, clusters)
    keywords_per_cluster = get_top_keywords(kmeans, vectorizer, k)
 
    missing = set(range(k)) - set(mapping.keys())
    if missing:
        raise ValueError(
            f"cluster_mapping đang thiếu nhãn cho cụm {sorted(missing)}. "
            "Xem TOP KEYWORDS ở trên để gán nhãn cho đủ trước khi tiếp tục."
        )
 
    df["label"] = df["cluster"].map(mapping)
    return df, kmeans, keywords_per_cluster
 
 
# ============================================================
# CHẠY ĐỘC LẬP: python src/modeling.py
# Giữ nguyên toàn bộ output/file như bản gốc, chỉ tổ chức lại thành hàm.
# ============================================================
if __name__ == "__main__":
    ensure_dirs()
    cfg = DEFAULT_CONFIG
 
    df, X, vectorizer = load_inputs(
        cfg["tfidf_matrix_path"],
        cfg["vectorizer_path"],
        cfg["processed_data_path"],
    )
 
    plot_elbow(
        X,
        cfg["k_search_range"],
        os.path.join(NOTEBOOKS_DIR, "elbow_method.png"),
        random_state=cfg["random_state"],
    )
 
    df, kmeans, keywords_per_cluster = cluster_and_label(df, X, vectorizer, cfg)
    clusters = df["cluster"].values
 
    save_keywords_report(
        keywords_per_cluster,
        CLUSTER_MAPPING,
        os.path.join(DATA_DIR, "cluster_keywords.txt"),
    )
 
    print("\n====================")
    print("SỐ LƯỢNG BÀI")
    print("====================")
    cluster_counts = df["cluster"].value_counts().sort_index()
    print(cluster_counts)
 
    dbscan_labels, n_clusters_dbscan, n_noise, dbscan_sample_idx = run_dbscan(
        X,
        cfg["dbscan_eps"],
        cfg["dbscan_min_samples"],
        cfg["dbscan_metric"],
        cfg["dbscan_sample_size"],
        random_state=cfg["random_state"],
    )
    plot_dbscan_distribution(
        dbscan_labels, os.path.join(NOTEBOOKS_DIR, "dbscan_distribution.png")
    )
 
    # Gán nhãn thủ công + lưu dữ liệu
    safe_to_csv(df, os.path.join(DATA_DIR, "labeled_news.csv"))
 
    # Train / Val / Test (Stratified Split)
    train_df, val_df, test_df = split_dataset(
        df, label_column="label", random_state=cfg["random_state"]
    )
    safe_to_csv(train_df, os.path.join(DATA_DIR, "train.csv"))
    safe_to_csv(val_df, os.path.join(DATA_DIR, "val.csv"))
    safe_to_csv(test_df, os.path.join(DATA_DIR, "test.csv"))
 
    safe_to_csv(df, os.path.join(DATA_DIR, "clustered_news.csv"))
 
    # Lưu model
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
 
    # Biểu đồ
    plot_cluster_distribution(
        cluster_counts, os.path.join(NOTEBOOKS_DIR, "cluster_distribution.png")
    )
    plot_kmeans_vs_dbscan(
        cfg["n_clusters"],
        n_clusters_dbscan,
        os.path.join(NOTEBOOKS_DIR, "kmeans_vs_dbscan.png"),
    )
    plot_pca(X, clusters, os.path.join(NOTEBOOKS_DIR, "pca_cluster.png"))
    plot_top_tfidf(X, vectorizer, os.path.join(NOTEBOOKS_DIR, "top20_tfidf.png"))
 
    print("\n====================")
    print("ĐÃ LƯU FILE")
    print("====================")
    print(os.path.join(DATA_DIR, "clustered_news.csv"))
    print(os.path.join(DATA_DIR, "train.csv"))
    print(os.path.join(DATA_DIR, "val.csv"))
    print(os.path.join(DATA_DIR, "test.csv"))
 
    print("\nBiểu đồ:")
    print(os.path.join(NOTEBOOKS_DIR, "cluster_distribution.png"))
    print(os.path.join(NOTEBOOKS_DIR, "kmeans_vs_dbscan.png"))
 
    print("\nModel:")
    print(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    print(os.path.join(DATA_DIR, "labeled_news.csv"))
    print(os.path.join(NOTEBOOKS_DIR, "elbow_method.png"))
    print(os.path.join(NOTEBOOKS_DIR, "pca_cluster.png"))
    print(os.path.join(DATA_DIR, "cluster_keywords.txt"))