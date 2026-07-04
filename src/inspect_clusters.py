"""
Script kiểm tra nhanh (READ-ONLY): xem processed_news.csv có cấu trúc gì,
cột sub_category hiện tại có những giá trị nào, và nếu chạy K-Means (K=5)
trên dữ liệu văn bản thì mỗi cụm sẽ có top từ khóa gì.

Script này KHÔNG ghi đè/lưu file nào cả — chỉ in ra console để bạn đối
chiếu với 5 nhãn (Phan_cung, Hang_khong_vu_tru, Tri_tue_nhan_tao,
Software, Mobile) trước khi chạy cluster_news.py (bản ghi thật).

Chạy: python inspect_clusters.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed_news.csv")

N_CLUSTERS = 5
RANDOM_STATE = 42
N_TOP_TERMS = 15


def main(processed_csv=PROCESSED_CSV):
    print(f"[inspect] Đang đọc: {processed_csv}\n")
    df = pd.read_csv(processed_csv)

    # ── 1. Tổng quan cấu trúc file ──────────────────────────────────────
    print("=" * 70)
    print(f"Số dòng: {len(df):,}")
    print(f"Các cột: {df.columns.tolist()}")
    print("=" * 70)

    # ── 2. Giá trị hiện có trong cột sub_category (nếu có) ──────────────
    if "sub_category" in df.columns:
        print("\n[sub_category] Phân bổ giá trị hiện tại:")
        print(df["sub_category"].value_counts(dropna=False))
    else:
        print("\n[sub_category] Cột này CHƯA tồn tại trong file.")

    # ── 3. Chọn cột văn bản để vector hóa ────────────────────────────────
    # Ưu tiên processed_content (đã tách từ sẵn, dạng "màn_hình", "tên_lửa")
    candidates = ["processed_content", "tags", "title"]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        fallback = ["sub_category", "main_content", "description"]
        cols = [c for c in fallback if c in df.columns]
    if not cols:
        print(f"\n[LỖI] Không tìm thấy cột văn bản nào trong {candidates}. "
              f"Không thể chạy K-Means. Gửi lại df.columns ở trên cho mình xem.")
        return

    print(f"\n[inspect] Dùng các cột văn bản để vector hóa: {cols}")
    text = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        text = text + " " + df[c].fillna("").astype(str)

    # ── 4. TF-IDF + K-Means ──────────────────────────────────────────────
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X = vectorizer.fit_transform(text)

    print(f"[inspect] Đang chạy K-Means (K={N_CLUSTERS}) ...")
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = model.fit_predict(X)

    # ── 5. In top từ khóa từng cụm + số bài viết mỗi cụm ─────────────────
    terms = np.array(vectorizer.get_feature_names_out())
    order = model.cluster_centers_.argsort()[:, ::-1]

    print("\n" + "=" * 70)
    print("KẾT QUẢ PHÂN CỤM (chỉ để xem, chưa ghi file):")
    print("=" * 70)
    counts = pd.Series(cluster_ids).value_counts().sort_index()
    for i in range(N_CLUSTERS):
        top_terms = terms[order[i, :N_TOP_TERMS]].tolist()
        n_articles = counts.get(i, 0)
        print(f"\nCluster {i}  ({n_articles:,} bài viết)")
        print(f"  {', '.join(top_terms)}")

    print("\n" + "=" * 70)
    print("Đối chiếu các Cluster ở trên với 5 nhãn:")
    print("  Phan_cung, Hang_khong_vu_tru, Tri_tue_nhan_tao, Software, Mobile")
    print("Nếu khớp đúng thứ tự -> chạy tiếp: python src/cluster_news.py")
    print("Nếu KHÔNG khớp (lệch thứ tự / gộp nhầm 2 chủ đề vào 1 cụm) -> "
          "gửi lại toàn bộ output này cho mình để chỉnh lại logic gán nhãn.")
    print("=" * 70)


if __name__ == "__main__":
    main()