import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from sklearn.metrics import silhouette_samples
 
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

    # Slide yêu cầu 5 cụm
    "n_clusters": 5,

    "k_search_range": range(2, 11),

    "dbscan_eps": 0.7,
    "dbscan_min_samples": 5,
    "dbscan_metric": "cosine",

    "dbscan_sample_size": None,

    "random_state": 42,
}


# ==========================
# GÁN NHÃN CHO CỤM
# ==========================
CLUSTER_MAPPING = {
    0: "Software",
    1: "Phan_cung",
    2: "Mobile",
    3: "an_ninh_mang",
    4: "Tri_tue_nhan_tao",
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

def generate_wordclouds(keywords_per_cluster, output_dir):

    print("\nĐang tạo WordCloud...")

    for cluster_id, words in keywords_per_cluster.items():

        text = " ".join(words)

        wc = WordCloud(
            width=1200,
            height=700,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(10, 6))

        plt.imshow(wc)

        plt.axis("off")

        plt.title(
            f"Cluster {cluster_id} - {CLUSTER_MAPPING.get(cluster_id)}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"wordcloud_cluster_{cluster_id}.png"
            ),
            dpi=300
        )

        plt.close()

    print("Đã lưu WordCloud")
 
 
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
 
# ==========================
# HIỂN THỊ MA TRẬN TF-IDF
# ==========================
def save_tfidf_matrix(X, vectorizer, output_path, rows=30, cols=20):

    print("\nĐang lưu ma trận TF-IDF...")

    terms = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(
        X[:rows, :cols].toarray(),
        columns=terms[:cols]
    )

    tfidf_df.to_csv(output_path, index=False)

    print("Đã lưu:", output_path)


# ==========================
# HEATMAP TF-IDF
# ==========================
def plot_tfidf_heatmap(
    X,
    vectorizer,
    output_path,
    rows=30,
    cols=20
):

    print("\nĐang tạo Heatmap TF-IDF...")

    terms = vectorizer.get_feature_names_out()

    tfidf_sample = X[:rows, :cols].toarray()

    plt.figure(figsize=(16, 8))

    plt.imshow(
        tfidf_sample,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="TF-IDF Score")

    plt.xticks(
        range(cols),
        terms[:cols],
        rotation=90
    )

    plt.yticks(
        range(rows),
        [f"Doc {i+1}" for i in range(rows)]
    )

    plt.title("Heatmap TF-IDF Matrix")

    plt.xlabel("Terms")
    plt.ylabel("Documents")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Đã lưu Heatmap:", output_path)
 
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

def plot_silhouette(X, clusters, output_path):

    print("\nĐang tạo Silhouette Plot...")

    sample_values = silhouette_samples(X, clusters)

    plt.figure(figsize=(10,8))

    y_lower = 10

    n_clusters = len(np.unique(clusters))

    for i in range(n_clusters):

        values = sample_values[clusters == i]

        values.sort()

        size = len(values)

        y_upper = y_lower + size

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values
        )

        plt.text(
            -0.05,
            y_lower + size/2,
            str(i)
        )

        y_lower = y_upper + 10

    avg_score = np.mean(sample_values)

    plt.axvline(
        x=avg_score,
        linestyle="--"
    )

    plt.title("Silhouette Analysis")

    plt.xlabel("Silhouette Score")

    plt.ylabel("Cluster")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("Đã lưu Silhouette Plot")

def plot_cluster_distance(
    X,
    kmeans,
    clusters,
    output_path
):

    print("\nĐang tạo Distance Plot...")

    distances = kmeans.transform(X)

    min_dist = distances.min(axis=1)

    df_dist = pd.DataFrame({
        "cluster": clusters,
        "distance": min_dist
    })

    avg = (
        df_dist
        .groupby("cluster")
        ["distance"]
        .mean()
    )

    plt.figure(figsize=(8,6))

    plt.bar(
        avg.index.astype(str),
        avg.values
    )

    plt.title(
        "Average Distance To Centroid"
    )

    plt.xlabel("Cluster")

    plt.ylabel("Distance")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("Đã lưu")

def plot_cluster_centers_heatmap(
    kmeans,
    vectorizer,
    output_path,
    top_n=20
):

    print("\nĐang tạo Cluster Center Heatmap...")

    centers = kmeans.cluster_centers_

    terms = vectorizer.get_feature_names_out()

    top_idx = np.argsort(
        centers.mean(axis=0)
    )[-top_n:]

    data = centers[:, top_idx]

    plt.figure(figsize=(14,8))

    plt.imshow(
        data,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(top_n),
        terms[top_idx],
        rotation=90
    )

    plt.yticks(
        range(
            centers.shape[0]
        ),
        [
            f"Cluster {i}"
            for i in range(
                centers.shape[0]
            )
        ]
    )

    plt.title(
        "KMeans Cluster Centers Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("Đã lưu Heatmap")


# ==========================
# DATASET SPLITTING PIE CHART
# ==========================

def plot_dataset_split_pie(output_path):

    print("\nĐang tạo Pie Chart Dataset Split...")

    labels = [
        "Train Set\n(70%)",
        "Validation Set\n(15%)",
        "Test Set\n(15%)"
    ]

    sizes = [70, 15, 15]

    colors = [
        "#4CAF50",   # xanh lá
        "#FFA726",   # cam
        "#EF5350"    # đỏ
    ]

    explode = (
        0.03,
        0.08,
        0.08
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={
            "fontsize": 14,
            "weight": "bold"
        }
    )

    plt.title(
        "Dataset Splitting (70% - 15% - 15%)",
        fontsize=22,
        weight="bold",
        pad=20
    )

    plt.legend(
        [
            "Train - Huấn luyện",
            "Validation - Tinh chỉnh",
            "Test - Đánh giá"
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3
    )

    plt.figtext(
        0.5,
        -0.05,
        "Stratified Split giúp giữ cân bằng tỷ lệ các lớp",
        ha="center",
        fontsize=12
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Đã lưu:", output_path)

# ==========================
# XUẤT MẪU ĐỂ GÁN NHÃN TAY + VẼ BIỂU ĐỒ
# ==========================
def export_sample_for_labeling(
    df,
    output_path,
    chart_path=None,
    n_per_cluster=200,
    random_state=42
):
    """
    Trích 200 bài/cụm × 5 cụm = 1000 bài
    → Xuất CSV
    → Vẽ biểu đồ phân bố mẫu để đưa vào báo cáo
    """

    samples = []

    for cluster_id in sorted(df["cluster"].unique()):

        cluster_df = df[df["cluster"] == cluster_id]

        n = min(n_per_cluster, len(cluster_df))

        sample = cluster_df.sample(
            n=n,
            random_state=random_state
        )

        samples.append(sample)

    result = pd.concat(
        samples,
        ignore_index=True
    )

    # thêm cột gán nhãn tay
    result.insert(
        0,
        "manual_label",
        ""
    )

    cols_to_keep = [
        "manual_label",
        "cluster",
        "label"
    ]

    for col in [
        "title",
        "content",
        "processed_content",
        "url"
    ]:
        if col in result.columns:
            cols_to_keep.append(col)

    result = result[cols_to_keep]

    result = (
        result
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    # lưu CSV
    safe_to_csv(
        result,
        output_path
    )

    # =====================
    # VẼ BIỂU ĐỒ
    # =====================
    if chart_path:

        counts = (
            result["cluster"]
            .value_counts()
            .sort_index()
        )

        labels = [
            CLUSTER_MAPPING.get(i)
            for i in counts.index
        ]

        plt.figure(
            figsize=(10, 6)
        )

        bars = plt.bar(
            labels,
            counts.values
        )

        plt.title(
            "Phân bố 1000 bài",
            fontsize=16,
            weight="bold"
        )

        plt.xlabel(
            "Cluster"
        )

        plt.ylabel(
            "Số bài"
        )

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width()/2,
                height + 5,
                str(int(height)),
                ha="center"
            )

        plt.tight_layout()

        plt.savefig(
            chart_path,
            dpi=300
        )

        plt.close()

        print(
            f"Đã lưu biểu đồ: {chart_path}"
        )

    print("\n====================")
    print("XUẤT MẪU GÁN NHÃN TAY")
    print("====================")

    print(
        f"Tổng số bài xuất: {len(result)}"
    )

    print(
        result["cluster"]
        .value_counts()
        .sort_index()
    )

    return result

# ==========================
# SO SÁNH NHÃN TAY vs K-MEANS
# ==========================
def compare_manual_vs_kmeans(labeled_path, output_path=None):
    """
    Sau khi gán nhãn tay xong → gọi hàm này để đánh giá K-Means chính xác bao nhiêu %.
    """
    df_labeled = pd.read_csv(labeled_path)

    df_done = df_labeled[
        df_labeled["manual_label"].notna() &
        (df_labeled["manual_label"].str.strip() != "")
    ]

    if len(df_done) == 0:
        print("[!] Chưa có dòng nào được gán nhãn tay.")
        return

    total = len(df_done)
    match = (
        df_done["manual_label"].str.strip() ==
        df_done["label"].str.strip()
    ).sum()
    accuracy = match / total * 100

    print("\n====================")
    print("SO SÁNH NHÃN TAY vs K-MEANS")
    print("====================")
    print(f"Tổng bài đã gán nhãn tay : {total}")
    print(f"Số bài khớp với K-Means  : {match}")
    print(f"Độ chính xác K-Means     : {accuracy:.2f}%")
    print("\nChi tiết theo cụm:")

    for cluster_id in sorted(df_done["cluster"].unique()):
        sub = df_done[df_done["cluster"] == cluster_id]
        sub_match = (
            sub["manual_label"].str.strip() ==
            sub["label"].str.strip()
        ).sum()
        print(f"  Cluster {cluster_id} "
              f"({CLUSTER_MAPPING.get(cluster_id)}): "
              f"{sub_match}/{len(sub)} khớp "
              f"({sub_match/len(sub)*100:.1f}%)")

    if output_path:
        df_done = df_done.copy()
        df_done["is_match"] = (
            df_done["manual_label"].str.strip() ==
            df_done["label"].str.strip()
        )
        safe_to_csv(df_done, output_path)
        print(f"\nĐã lưu kết quả so sánh: {output_path}")

    return df_done

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

    # Xuất 1000 bài mẫu để gán nhãn tay (200 bài × 5 cụm)
    export_sample_for_labeling(
        df,
        os.path.join(
            DATA_DIR,
            "sample_for_labeling.csv"
        ),
        chart_path=os.path.join(
            NOTEBOOKS_DIR,
            "sample_label_distribution.png"
        ),
        n_per_cluster=200,
        random_state=cfg["random_state"]
    )
 
    save_keywords_report(
        keywords_per_cluster,
        CLUSTER_MAPPING,
        os.path.join(DATA_DIR, "cluster_keywords.txt"),
    )

    generate_wordclouds(
        keywords_per_cluster,
        NOTEBOOKS_DIR
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
    # Lưu ma trận TF-IDF
    save_tfidf_matrix(
        X,
        vectorizer,
        os.path.join(DATA_DIR, "tfidf_matrix.csv"),
        rows=30,
        cols=20
    )

    # Vẽ Heatmap TF-IDF
    plot_tfidf_heatmap(
        X,
        vectorizer,
        os.path.join(NOTEBOOKS_DIR, "tfidf_heatmap.png"),
        rows=50,
        cols=30
    )

    plot_silhouette(
        X,
        clusters,
        os.path.join(
            NOTEBOOKS_DIR,
            "silhouette_plot.png"
        )
    )

    plot_cluster_distance(
        X,
        kmeans,
        clusters,
        os.path.join(
            NOTEBOOKS_DIR,
            "cluster_distance.png"
        )
    )

    plot_cluster_centers_heatmap(
        kmeans,
        vectorizer,
        os.path.join(
            NOTEBOOKS_DIR,
            "kmeans_center_heatmap.png"
        )
    )

    plot_dataset_split_pie(
        os.path.join(
            NOTEBOOKS_DIR,
            "dataset_split_pie.png"
        )
    )
 
    print("\n====================")
    print("ĐÃ LƯU FILE")
    print("====================")
    print(os.path.join(DATA_DIR, "clustered_news.csv"))
    print(os.path.join(DATA_DIR, "train.csv"))
    print(os.path.join(DATA_DIR, "val.csv"))
    print(os.path.join(DATA_DIR, "test.csv"))
    print(os.path.join(DATA_DIR, "sample_for_labeling.csv"))  # thêm dòng này

    print("\nBiểu đồ:")
    print(os.path.join(NOTEBOOKS_DIR, "cluster_distribution.png"))
    print(os.path.join(NOTEBOOKS_DIR, "kmeans_vs_dbscan.png"))
    print(os.path.join(NOTEBOOKS_DIR, "dataset_splitting.png"))
    print("\nModel:")
    print(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    print(os.path.join(DATA_DIR, "labeled_news.csv"))
    print(os.path.join(NOTEBOOKS_DIR, "elbow_method.png"))
    print(os.path.join(NOTEBOOKS_DIR, "pca_cluster.png"))
    print(os.path.join(DATA_DIR, "cluster_keywords.txt"))