"""
Module Phân cụm K-Means & Gán nhãn Sub-category (Cluster News).

Đọc data/processed_news.csv, thực hiện phân cụm K-Means (K=5) dựa trên
văn bản (tags + sub_category + title, tuỳ cột nào có sẵn), sau đó GÁN
NHÃN vào cột "sub_category" theo bảng CLUSTER_LABELS bên dưới.

QUAN TRỌNG: CLUSTER_LABELS là mapping THỦ CÔNG — bạn tự xem output của
`inspect_clusters.py` (hoặc log "Top từ khóa mỗi cụm" khi chạy chính file
này), rồi tự quyết định Cluster nào ứng với nhãn nào và điền vào dict bên
dưới. KHÔNG có logic tự động đoán nhãn — vì random_state cố định (42) nên
với cùng 1 file processed_news.csv, thứ tự cluster_id sẽ ổn định, nhưng
nếu bạn đổi dữ liệu đầu vào thì phải chạy inspect lại rồi cập nhật dict.

Mapping hiện tại (theo bạn xác nhận từ log của mình):
    Cluster 0 -> Hang_khong_vu_tru   (tên_lửa, tàu, vũ_trụ, mặt_trăng, ...)
    Cluster 1 -> Tri_tue_nhan_tao    (ai, google, ứng_dụng, dữ_liệu, ...)
    Cluster 2 -> Phan_cung           (iphone, apple, pro, màn_hình, ...)
    Cluster 3 -> Software            (việt_nam, doanh_nghiệp, chuyển_đổi, ...)
    Cluster 4 -> Mobile              (galaxy, samsung, s26, điện_thoại, ...)

Output: data/clustered_news.csv (giữ nguyên toàn bộ cột gốc + thêm
        cột "cluster" (số cụm) và ghi đè/gán cột "sub_category" bằng nhãn).

Chạy độc lập: python cluster_news.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed_news.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "clustered_news.csv")

N_CLUSTERS = 5
RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════
# MAPPING THỦ CÔNG — SỬA CHỖ NÀY nếu thứ tự cụm thay đổi sau khi bạn chạy
# inspect_clusters.py và thấy cluster_id không khớp nữa.
# ═══════════════════════════════════════════════════════════════════════
CLUSTER_LABELS = {
    0: "Hang_khong_vu_tru",
    1: "Tri_tue_nhan_tao",
    2: "Phan_cung",
    3: "Software",
    4: "Mobile",
}


# ═══════════════════════════════════════════════════════════════════════
# HÀM CHÍNH
# ═══════════════════════════════════════════════════════════════════════

def run_clustering(processed_csv=PROCESSED_CSV, out_csv=OUT_CSV,
                    n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
                    cluster_labels=CLUSTER_LABELS):
    """
    Đọc processed_news.csv, phân cụm K-Means, gán nhãn (theo mapping thủ
    công cluster_labels) vào sub_category, lưu kết quả ra clustered_news.csv.

    Args:
        processed_csv  (str): Đường dẫn tới processed_news.csv
        out_csv        (str): Đường dẫn file output
        n_clusters     (int): Số cụm K-Means (mặc định 5)
        random_state   (int): Seed để kết quả tái lập được
        cluster_labels (dict): Mapping {cluster_id: label} do bạn tự khai báo

    Returns:
        pd.DataFrame: DataFrame kết quả (đã có cột "cluster" và "sub_category")
    """
    print(f"[cluster_news] Đang đọc {processed_csv} ...")
    df = pd.read_csv(processed_csv)

    text, used_cols = _build_text_column(df)
    print(f"[cluster_news] Dùng các cột văn bản để vector hóa: {used_cols}")

    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X = vectorizer.fit_transform(text)

    print(f"[cluster_news] Đang chạy K-Means (K={n_clusters}) ...")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = model.fit_predict(X)

    # In top từ khóa mỗi cụm để bạn đối chiếu lại với CLUSTER_LABELS
    top_terms = _top_terms_per_cluster(model, vectorizer)
    print("\n[cluster_news] Top từ khóa mỗi cụm (đối chiếu với CLUSTER_LABELS bên dưới):")
    for ci, terms in top_terms.items():
        label = cluster_labels.get(ci, "??? (chưa khai báo)")
        print(f"  Cluster {ci} -> {label}")
        print(f"    {', '.join(terms[:12])}")

    # Cảnh báo nếu có cluster_id chưa được khai báo nhãn
    missing = [ci for ci in range(n_clusters) if ci not in cluster_labels]
    if missing:
        print(f"\n[CẢNH BÁO] Các cluster chưa có nhãn trong CLUSTER_LABELS: {missing}. "
              f"Các dòng thuộc cụm này sẽ có sub_category = NaN. "
              f"Hãy bổ sung vào dict CLUSTER_LABELS ở đầu file rồi chạy lại.")

    df["cluster"] = cluster_ids
    df["sub_category"] = df["cluster"].map(cluster_labels)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[cluster_news] ✓ Đã lưu: {out_csv}")

    print("\n[cluster_news] Phân bổ sub_category sau khi gán nhãn:")
    print(df["sub_category"].value_counts(dropna=False))

    return df


# ═══════════════════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ
# ═══════════════════════════════════════════════════════════════════════

def _build_text_column(df):
    """
    Ghép các cột văn bản có sẵn thành 1 cột duy nhất để vector hóa.

    Ưu tiên "processed_content" trước tiên vì cột này đã được tách từ sẵn
    (từ ghép gạch dưới như "màn_hình", "tên_lửa") — khớp đúng định dạng
    keyword dùng để đặt tên nhãn cụm. Ghép thêm "tags" và "title" (nếu có)
    để tăng tín hiệu, vì tags thường cô đọng và rất đặc trưng cho chủ đề.
    """
    candidates = ["processed_content", "tags", "title"]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        # fallback nếu vì lý do gì đó không có processed_content/tags/title
        fallback = ["sub_category", "main_content", "description"]
        cols = [c for c in fallback if c in df.columns]
    if not cols:
        raise ValueError(
            "Không tìm thấy cột văn bản phù hợp trong processed_news.csv "
            f"(đã thử: {candidates + ['sub_category', 'main_content', 'description']})"
        )
    text = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        text = text + " " + df[c].fillna("").astype(str)
    return text, cols


def _top_terms_per_cluster(model, vectorizer, n_terms=15):
    """Lấy top n từ khóa đại diện cho từng cụm dựa trên centroid TF-IDF."""
    terms = np.array(vectorizer.get_feature_names_out())
    order = model.cluster_centers_.argsort()[:, ::-1]
    top = {}
    for i in range(model.n_clusters):
        top[i] = terms[order[i, :n_terms]].tolist()
    return top


# ═══════════════════════════════════════════════════════════════════════
# CHẠY TRỰC TIẾP
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_clustering()