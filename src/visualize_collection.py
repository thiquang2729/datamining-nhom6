"""
Module Trực quan hóa Dữ liệu Thu thập (Data Collection Visualization).
 
Vẽ và lưu 7 hình phục vụ phần Data Collection – Visualization & Insight:
    1. Metric cards tổng quan
    2. Phân bổ nguồn báo (donut)
    3. Phân bổ sub-category (horizontal bar)
    4. Timeline thu thập theo tháng (line chart)
    5. Phân bổ nhãn K-Means sau clustering (donut)
    6. Top tags nổi bật (horizontal bar)
    7. Phân bổ độ dài nội dung bài viết (histogram)
 
Output: lưu tất cả ảnh vào thư mục  notebooks/
Chạy độc lập: python src/visualize_collection.py
Hoặc gọi từ pipeline: from visualize_collection import run_collection_viz

LƯU Ý: cột "label" trong clustered_news.csv giờ được sinh ra bởi
src/cluster_news.py (chạy K-Means trên data/processed_news.csv), với
Cluster 1 nay là "Hang_khong_vu_tru" (Hàng không - Vũ trụ) thay cho
"An_ninh_mang" trước đây. Nếu bạn dùng pipeline mới, hãy chạy:
    python src/cluster_news.py
trước khi chạy file này, để data/clustered_news.csv được cập nhật.
"""
 
import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
 
# ── Thêm thư mục src vào sys.path (để import logger nếu cần) ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
# ── Đường dẫn gốc dự án ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_CSV      = os.path.join(PROJECT_ROOT, "data", "news_data.csv")
CLUSTERED_CSV = os.path.join(PROJECT_ROOT, "data", "clustered_news.csv")
OUT_DIR       = os.path.join(PROJECT_ROOT, "notebooks")   # cùng thư mục với các hình khác
 
# ── Bảng màu nhất quán với cả project ──────────────────────────────────────
BLUE   = "#185FA5"
GREEN  = "#0F6E56"
CORAL  = "#993C1D"
PURPLE = "#534AB7"
AMBER  = "#BA7517"
DKGRN  = "#3B6D11"
GRAY   = "#888780"
PALETTE = [
    "#1F77B4",   # Phần cứng
    "#FF7F0E",   # AI
    "#2CA02C",   # Hàng không - Vũ trụ
    "#D62728",   # Mobile
    "#9467BD",   # Software
]
 
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


# ── Helper: tự dò tên cột (phòng trường hợp tên cột thực tế khác) ──────────
def _find_col(df, candidates, required=True, what=""):
    """Trả về tên cột đầu tiên trong candidates có mặt trong df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Không tìm thấy cột {what or candidates} trong dữ liệu. "
            f"Các cột hiện có: {df.columns.tolist()}"
        )
    return None
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# HÀM CHÍNH – gọi từ bên ngoài hoặc từ pipeline
# ═══════════════════════════════════════════════════════════════════════════════
 
def run_collection_viz(news_csv=NEWS_CSV, clustered_csv=CLUSTERED_CSV, out_dir=OUT_DIR):
    """
    Đọc dữ liệu và vẽ toàn bộ 7 hình Data Collection.
 
    Args:
        news_csv      (str): Đường dẫn tới news_data.csv
        clustered_csv (str): Đường dẫn tới clustered_news.csv (sinh ra bởi
                              src/cluster_news.py từ data/processed_news.csv)
        out_dir       (str): Thư mục lưu ảnh (mặc định: notebooks/)
    """
    os.makedirs(out_dir, exist_ok=True)
 
    # ── Đọc dữ liệu ────────────────────────────────────────────────────────
    print("[visualize_collection] Đang đọc dữ liệu...")
    df = pd.read_csv(news_csv)

    date_col = _find_col(
        df,
        ["published_time", "publish_time", "published_date", "publish_date",
         "date", "time", "created_at", "crawl_time"],
        required=False,
        what="ngày đăng (published_time)",
    )
    has_date = date_col is not None
    if has_date:
        df["published_time"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df["year"]  = df["published_time"].dt.year
        df["month"] = df["published_time"].dt.to_period("M")
    else:
        print("[visualize_collection] CẢNH BÁO: không tìm thấy cột ngày đăng trong "
              f"news_data.csv (các cột hiện có: {df.columns.tolist()}). "
              "Sẽ bỏ qua hình Timeline (collection_04_timeline.png).")
 
    df_cl = pd.read_csv(clustered_csv)
    print("=" * 50)
    print(df_cl.columns.tolist())
    print(df_cl.head())
    print(df_cl["label"].value_counts(dropna=False))
    print("=" * 50)
    # cluster_news.py ghi nhãn vào cột "sub_category" (không phải "label")
    label_col = _find_col(df_cl, ["sub_category", "label"], what="nhãn cụm")
 
    # ── Chuẩn bị các giá trị dùng chung ────────────────────────────────────
    domain_col = _find_col(
        df, ["domain", "source", "site", "news_source"],
        required=False, what="nguồn báo (domain)",
    )
    has_domain = domain_col is not None
    if has_domain:
        domain_counts = df[domain_col].value_counts()
        dom_labels = ["VnExpress", "VietnamNet", "Thanh Niên"]
        dom_values = [
            domain_counts.get("vnexpress.net", 0),
            domain_counts.get("vietnamnet.vn",  0),
            domain_counts.get("thanhnien.vn",   0),
        ]
    else:
        # Không tìm thấy cột domain -> vẫn vẽ được nhưng để giá trị 0
        print("[visualize_collection] CẢNH BÁO: không tìm thấy cột domain/source, "
              "sẽ bỏ qua hình phân bổ nguồn báo (collection_02_domain_donut.png).")
        dom_labels = ["VnExpress", "VietnamNet", "Thanh Niên"]
        dom_values = [0, 0, 0]
    dom_colors = [BLUE, GREEN, CORAL]
 
    sub = df["sub_category"].value_counts().head(8)
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if has_date:
            timeline = (
                df[df["year"] >= 2024]
                .groupby(df["month"].astype(str))
                .size()
                .reset_index(name="count")
                .sort_values("month")
            )
        else:
            timeline = pd.DataFrame(columns=["month", "count"])
 
    cluster_dist = df_cl[label_col].value_counts()
    label_map = {
        "Phan_cung":         "Phần cứng",
        "Tri_tue_nhan_tao":  "AI",
        "Hang_khong_vu_tru": "Hàng không - Vũ trụ",
        "Mobile":            "Mobile",
        "Software":          "Software",
    }
    cl_labels = [label_map.get(l, l) for l in cluster_dist.index]
    cl_values = cluster_dist.values.tolist()
 
    tags_series = (
        df["tags"].dropna()
        .str.split(",").explode()
        .str.strip().str.lower()
        .replace("", pd.NA).dropna()
    )
    top_tags = tags_series.value_counts().head(9)
 
    df["content_len"] = df["main_content"].str.len().fillna(0)
    bins_len   = [0, 500, 1000, 2000, 3000, 4000, 5000, 7000, 20000]
    labels_len = ["0–500","500–1k","1k–2k","2k–3k","3k–4k","4k–5k","5k–7k","7k+"]
    counts_len, _ = np.histogram(df["content_len"], bins=bins_len)
 
    # ── Vẽ từng hình ───────────────────────────────────────────────────────
    _fig1_metric_cards(df, df_cl, label_col, out_dir)
    if has_domain:
        _fig2_domain_donut(dom_labels, dom_values, dom_colors, out_dir)
    else:
        print("[visualize_collection] Bỏ qua collection_02_domain_donut.png (thiếu dữ liệu domain).")
    _fig3_subcategory(sub, out_dir)
    if has_date and len(timeline) > 0:
        _fig4_timeline(timeline, out_dir)
    else:
        print("[visualize_collection] Bỏ qua collection_04_timeline.png (thiếu dữ liệu ngày).")
        print("label_col =", label_col)
        print("cl_labels =", cl_labels)
        print("cl_values =", cl_values)
    _fig5_cluster_donut(cl_labels, cl_values, out_dir)
    _fig6_top_tags(top_tags, out_dir)
    _fig7_content_length(labels_len, counts_len, out_dir)
    _fig8_dashboard(dom_labels if has_domain else None,
                    dom_values if has_domain else None,
                    dom_colors,
                    sub, timeline if (has_date and len(timeline) > 0) else None,
                    cl_labels, cl_values,
                    labels_len, counts_len, out_dir)
 
    print(f"[visualize_collection] ✓ Đã lưu 8 hình vào: {out_dir}")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# CÁC HÀM VẼ TỪNG HÌNH
# ═══════════════════════════════════════════════════════════════════════════════
 
 
def _fig2_domain_donut(labels, values, colors, out_dir):
    """Hình 2 – Donut phân bổ nguồn báo."""
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, _, autotexts = ax.pie(
        values, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_color("white"); at.set_fontweight("500")
    ax.legend(
        [mpatches.Patch(color=c) for c in colors],
        [f"{l}  ({v:,})" for l, v in zip(labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, fontsize=9, frameon=False,
    )
    ax.set_title("Phân bổ bài viết theo nguồn báo", fontsize=12, pad=14, color="#333")
    plt.tight_layout()
    _save(fig, out_dir, "collection_02_domain_donut.png")
 
 
def _fig3_subcategory(sub, out_dir):
    """Hình 3 – Horizontal bar sub-category."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sub.index[::-1], sub.values[::-1], color=BLUE, height=0.6)
    ax.set_xlabel("Số bài viết", fontsize=10)
    ax.set_title("Phân bổ theo danh mục con (sub-category)", fontsize=12, color="#333", pad=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for bar, val in zip(bars, sub.values[::-1]):
        ax.text(bar.get_width() + 60, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color="#555")
    plt.tight_layout()
    _save(fig, out_dir, "collection_03_subcategory.png")
 
 
def _fig4_timeline(timeline, out_dir):
    """Hình 4 – Line chart timeline theo tháng."""
    x = range(len(timeline))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(list(x), timeline["count"], alpha=0.12, color=BLUE)
    ax.plot(list(x), timeline["count"], color=BLUE, linewidth=2,
            marker="o", markersize=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [m[2:] for m in timeline["month"]],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Số bài viết", fontsize=10)
    ax.set_title("Timeline thu thập dữ liệu — bài viết theo tháng (2024–2026)",
                 fontsize=12, color="#333", pad=10)
    # Đánh dấu Jan 2025
    jan25 = timeline["month"].tolist().index("2025-01") if "2025-01" in timeline["month"].tolist() else None
    if jan25 is not None:
        ax.axvline(x=jan25, color=AMBER, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.text(jan25 + 0.2, timeline["count"].max() * 0.90,
                "Jan 2025\n(tăng đột biến)", fontsize=8, color=AMBER)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, out_dir, "collection_04_timeline.png")
 
 

def _fig6_top_tags(top_tags, out_dir):
    """Hình 6 – Horizontal bar top tags."""
    tag_colors = [BLUE, PURPLE, GRAY, GREEN, PURPLE, GRAY, BLUE, BLUE, CORAL]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(top_tags.index[::-1], top_tags.values[::-1],
                   color=tag_colors[:len(top_tags)][::-1], height=0.6)
    ax.set_xlabel("Số bài viết", fontsize=10)
    ax.set_title("Top từ khóa (tags) nổi bật trong tập dữ liệu", fontsize=12, color="#333", pad=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for bar, val in zip(bars, top_tags.values[::-1]):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color="#555")
    plt.tight_layout()
    _save(fig, out_dir, "collection_06_top_tags.png")
 
 
def _fig7_content_length(labels_len, counts_len, out_dir):
    """Hình 7 – Histogram độ dài nội dung."""
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels_len, counts_len, color=BLUE, width=0.6)
    ax.set_xlabel("Độ dài nội dung (ký tự)", fontsize=10)
    ax.set_ylabel("Số bài viết", fontsize=10)
    ax.set_title("Phân bổ độ dài nội dung bài viết", fontsize=12, color="#333", pad=10)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, counts_len):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontsize=9, color="#555")
    plt.tight_layout()
    _save(fig, out_dir, "collection_07_content_length.png")
 
 
 
# ── Helper lưu file ─────────────────────────────────────────────────────────
def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")
 
 

# ═══════════════════════════════════════════════════════════════════════════════
# CHẠY TRỰC TIẾP
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_collection_viz()