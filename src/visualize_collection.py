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
    "#2CA02C",   # An ninh mạng
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
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# HÀM CHÍNH – gọi từ bên ngoài hoặc từ pipeline
# ═══════════════════════════════════════════════════════════════════════════════
 
def run_collection_viz(news_csv=NEWS_CSV, clustered_csv=CLUSTERED_CSV, out_dir=OUT_DIR):
    """
    Đọc dữ liệu và vẽ toàn bộ 7 hình Data Collection.
 
    Args:
        news_csv      (str): Đường dẫn tới news_data.csv
        clustered_csv (str): Đường dẫn tới clustered_news.csv
        out_dir       (str): Thư mục lưu ảnh (mặc định: notebooks/)
    """
    os.makedirs(out_dir, exist_ok=True)
 
    # ── Đọc dữ liệu ────────────────────────────────────────────────────────
    print("[visualize_collection] Đang đọc dữ liệu...")
    df = pd.read_csv(news_csv)
    df["published_time"] = pd.to_datetime(df["published_time"], utc=True, errors="coerce")
    df["year"]  = df["published_time"].dt.year
    df["month"] = df["published_time"].dt.to_period("M")
 
    df_cl = pd.read_csv(clustered_csv)
 
    # ── Chuẩn bị các giá trị dùng chung ────────────────────────────────────
    domain_counts = df["domain"].value_counts()
    dom_labels = ["VnExpress", "VietnamNet", "Thanh Niên"]
    dom_values = [
        domain_counts.get("vnexpress.net", 0),
        domain_counts.get("vietnamnet.vn",  0),
        domain_counts.get("thanhnien.vn",   0),
    ]
    dom_colors = [BLUE, GREEN, CORAL]
 
    sub = df["sub_category"].value_counts().head(8)
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timeline = (
            df[df["year"] >= 2024]
            .groupby(df["month"].astype(str))
            .size()
            .reset_index(name="count")
            .sort_values("month")
        )
 
    cluster_dist = df_cl["label"].value_counts()
    label_map = {
        "Phan_cung": "Phần cứng",
        "Tri_tue_nhan_tao": "AI",
        "An_ninh_mang": "An ninh mạng",
        "Mobile": "Mobile",
        "Software": "Software",
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
    _fig1_metric_cards(df, df_cl, out_dir)
    _fig2_domain_donut(dom_labels, dom_values, dom_colors, out_dir)
    _fig3_subcategory(sub, out_dir)
    _fig4_timeline(timeline, out_dir)
    _fig5_cluster_donut(cl_labels, cl_values, out_dir)
    _fig6_top_tags(top_tags, out_dir)
    _fig7_content_length(labels_len, counts_len, out_dir)
    _fig8_dashboard(dom_labels, dom_values, dom_colors,
                    sub, timeline,
                    cl_labels, cl_values,
                    labels_len, counts_len, out_dir)
 
    print(f"[visualize_collection] ✓ Đã lưu 8 hình vào: {out_dir}")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# CÁC HÀM VẼ TỪNG HÌNH
# ═══════════════════════════════════════════════════════════════════════════════
 
def _fig1_metric_cards(df, df_cl, out_dir):
    """Hình 1 – 4 metric cards tổng quan."""
    avg_len = int(df["main_content"].str.len().fillna(0).mean())
    metrics = [
        ("Tổng bài viết",  f"{len(df):,}",   "raw articles scraped"),
        ("Nguồn báo",      "3",               "vnexpress · vietnamnet\n· thanhnien"),
        ("Nhãn K-Means",   str(df_cl["label"].nunique()), "cụm phân loại"),
        ("Avg. content",   f"{avg_len:,}",    "ký tự / bài viết"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    fig.patch.set_facecolor("#F7F7F5")
    for ax, (title, val, sub) in zip(axes, metrics):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.80, title.upper(), ha="center", va="center",
                fontsize=9, color="#888", transform=ax.transAxes)
        ax.text(0.5, 0.48, val, ha="center", va="center",
                fontsize=28, color="#111", fontweight="500", transform=ax.transAxes)
        ax.text(0.5, 0.16, sub, ha="center", va="center",
                fontsize=9, color="#888", transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#ddd")
    plt.suptitle("Data Collection — Nhóm 6 Data Mining",
                 fontsize=13, y=1.04, color="#333")
    plt.tight_layout(pad=1.2)
    _save(fig, out_dir, "collection_01_metric_cards.png")
 
 
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
 
 
def _fig5_cluster_donut(labels, values, out_dir):
    """Hình 5 – Donut phân bổ nhãn K-Means."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = PALETTE[:5]
    wedges, _, autotexts = ax.pie(
        values, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.80,
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_color("white"); at.set_fontweight("500")
    ax.legend(
        [mpatches.Patch(color=c) for c in colors],
        [f"{l}  ({v:,})" for l, v in zip(labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.13),
        ncol=2, fontsize=9, frameon=False,
    )
    ax.set_title("Phân bổ nhãn sau phân cụm K-Means", fontsize=12, pad=14, color="#333")
    plt.tight_layout()
    _save(fig, out_dir, "collection_05_cluster_donut.png")
 
 
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
 
 
def _fig8_dashboard(dom_labels, dom_values, dom_colors,
                    sub, timeline,
                    cl_labels, cl_values,
                    labels_len, counts_len, out_dir):
    """Hình 8 – Dashboard tổng hợp 2×3."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Data Collection — Visualization & Insight  |  Nhóm 6 Data Mining",
        fontsize=14, fontweight="500", y=0.98, color="#222",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)
 
    # 8a: Domain donut
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(dom_values, colors=dom_colors, startangle=90,
            autopct="%1.0f%%", pctdistance=0.78,
            wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2))
    for t in ax1.texts: t.set_fontsize(9)
    ax1.set_title("Nguồn báo", fontsize=11, pad=8)
    ax1.legend([mpatches.Patch(color=c) for c in dom_colors], dom_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.14),
               ncol=3, fontsize=7.5, frameon=False)
 
    # 8b: Sub-category
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(sub.index[::-1], sub.values[::-1], color=BLUE, height=0.6)
    ax2.set_title("Sub-category", fontsize=11, pad=8)
    ax2.tick_params(axis="y", labelsize=7.5)
    ax2.set_axisbelow(True)
    ax2.spines["left"].set_visible(False)
 
    # 8c: Cluster donut
    ax3 = fig.add_subplot(gs[0, 2])
    colors3 = PALETTE[:5]
    ax3.pie(cl_values, colors=colors3, startangle=90,
            autopct="%1.0f%%", pctdistance=0.80,
            wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2))
    for t in ax3.texts: t.set_fontsize(8)
    ax3.set_title("Cụm K-Means", fontsize=11, pad=8)
    short_labels = [l.split("/")[0].strip().split("&")[0].strip() for l in cl_labels]
    ax3.legend([mpatches.Patch(color=c) for c in colors3], short_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=7, frameon=False)
 
    # 8d: Timeline (chiếm 2 cột)
    ax4 = fig.add_subplot(gs[1, :2])
    x = range(len(timeline))
    ax4.fill_between(list(x), timeline["count"], alpha=0.12, color=BLUE)
    ax4.plot(list(x), timeline["count"], color=BLUE, linewidth=1.8,
             marker="o", markersize=3)
    ax4.set_xticks(list(x))
    ax4.set_xticklabels([m[2:] for m in timeline["month"]],
                        rotation=45, ha="right", fontsize=7)
    ax4.set_title("Timeline thu thập (2024–2026)", fontsize=11, pad=8)
    jan25 = timeline["month"].tolist().index("2025-01") if "2025-01" in timeline["month"].tolist() else None
    if jan25:
        ax4.axvline(x=jan25, color=AMBER, linewidth=1, linestyle="--", alpha=0.8)
        ax4.text(jan25 + 0.2, timeline["count"].max() * 0.88, "Jan 2025", fontsize=7.5, color=AMBER)
    ax4.set_axisbelow(True)
 
    # 8e: Content length
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(labels_len, counts_len, color=BLUE, width=0.6)
    ax5.set_title("Độ dài nội dung", fontsize=11, pad=8)
    ax5.tick_params(axis="x", labelsize=7.5, rotation=30)
    ax5.set_axisbelow(True)
 
    out = os.path.join(out_dir, "collection_08_dashboard.png")
    plt.savefig(out, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ collection_08_dashboard.png")
 
 
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
 
