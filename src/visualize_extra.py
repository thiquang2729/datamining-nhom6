"""
Vẽ 2 hình bổ sung cho phần Data Collection – Visualization & Insight:
    1. Correlation Matrix: Nguồn báo vs Top Tags (heatmap đỏ–trắng)
    2. Scatter Plot – Word Count từng bài báo (với outlier đánh dấu đỏ)
 
Output: lưu vào  notebooks/
Chạy: python src/visualize_extra.py
"""
 
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
 
# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_CSV      = os.path.join(PROJECT_ROOT, "data", "news_data.csv")
OUT_DIR       = os.path.join(PROJECT_ROOT, "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)
 
 
def run_extra_viz(news_csv=NEWS_CSV, out_dir=OUT_DIR):
    print("[visualize_extra] Đang đọc dữ liệu...")
    df = pd.read_csv(news_csv)
 
    _fig_heatmap(df, out_dir)
    _fig_scatter(df, out_dir)
 
    print(f"[visualize_extra] ✓ Đã lưu 2 hình vào: {out_dir}")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 1 – Correlation Matrix: Nguồn báo vs Top Tags
# ═══════════════════════════════════════════════════════════════════════════════
 
def _fig_heatmap(df, out_dir):
    # ── Xây dựng ma trận ───────────────────────────────────────────────────
    tags_ex = df[["article_id", "domain", "tags"]].copy()
    tags_ex["tag"] = tags_ex["tags"].str.split(",")
    tags_ex = tags_ex.explode("tag")
    tags_ex["tag"] = tags_ex["tag"].str.strip().str.lower()
    tags_ex = tags_ex[tags_ex["tag"] != ""].drop_duplicates(["article_id", "tag"])
 
    top_tags = tags_ex["tag"].value_counts().head(9).index.tolist()
    filtered = tags_ex[tags_ex["tag"].isin(top_tags)]
    matrix = filtered.groupby(["domain", "tag"]).size().unstack(fill_value=0)
 
    # ── Nhãn hiển thị đẹp hơn ──────────────────────────────────────────────
    col_rename = {
        "ai":               "AI",
        "apple":            "Apple",
        "công nghệ":        "Công nghệ",
        "google":           "Google",
        "iphone":           "iPhone",
        "samsung":          "Samsung",
        "tin nóng":         "Tin nóng",
        "trung quốc":       "Trung Quốc",
        "trí tuệ nhân tạo": "Trí tuệ\nnhân tạo",
    }
    row_rename = {
        "thanhnien.vn":  "thanhnien",
        "vietnamnet.vn": "vietnamnet",
        "vnexpress.net": "vnexpress",
    }
    matrix.columns = [col_rename.get(c, c) for c in matrix.columns]
    matrix.index   = [row_rename.get(r, r) for r in matrix.index]
 
    data = matrix.values
    rows = matrix.index.tolist()
    cols = matrix.columns.tolist()
 
    # ── Vẽ ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
 
    # Colormap trắng → đỏ đậm (giống slide)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "white_red", ["#FFF9F5", "#FCDED6", "#E8715A", "#C0392B", "#8B0000"]
    )
 
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=data.max())
 
    # Lưới và nhãn
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, fontsize=11, ha="center")
    ax.set_yticklabels(rows, fontsize=11)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.set_xlabel("Tag (Từ khóa)", fontsize=12, labelpad=10, color="#333")
    ax.set_ylabel("Source (Nguồn)", fontsize=12, labelpad=10, color="#333")
 
    # Số trong ô
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            text_color = "white" if val > data.max() * 0.55 else "#333"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="500")
 
    # Viền ô
    for i in range(len(rows) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)
 
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Number of Articles", fontsize=9, color="#555")
 
    ax.set_title(
        "Correlation Matrix: Source vs Tag\n(Mối quan hệ giữa Nguồn báo - Từ khóa)",
        fontsize=13, pad=14, color="#222", fontweight="normal"
    )
 
    # Bỏ spines
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    plt.tight_layout()
    out = os.path.join(out_dir, "collection_09_correlation_matrix.png")
    fig.savefig(out, bbox_inches="tight", dpi=180, facecolor="white")
    plt.close()
    print(f"  ✓ collection_09_correlation_matrix.png")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 2 – Scatter Plot: Word Count từng bài báo + Outliers
# ═══════════════════════════════════════════════════════════════════════════════
 
def _fig_scatter(df, out_dir):
    # ── Tính word count ────────────────────────────────────────────────────
    df = df.copy()
    df["word_count"] = df["main_content"].str.split().str.len().fillna(0).astype(int)
 
    # IQR outlier detection (giống slide)
    Q1  = df["word_count"].quantile(0.25)
    Q3  = df["word_count"].quantile(0.75)
    IQR = Q3 - Q1
    lower = max(0, Q1 - 1.5 * IQR)
    upper = Q3 + 1.5 * IQR
 
    is_outlier  = (df["word_count"] < lower) | (df["word_count"] > upper)
    is_normal   = ~is_outlier
    n_outliers  = is_outlier.sum()
 
    x_all      = np.arange(len(df))
    y_normal   = df.loc[is_normal,  "word_count"].values
    x_normal   = x_all[is_normal.values]
    y_outlier  = df.loc[is_outlier, "word_count"].values
    x_outlier  = x_all[is_outlier.values]
 
    # ── Vẽ ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
 
    # Vùng normal (shading nhạt)
    ax.axhspan(lower, upper, alpha=0.12, color="#4CAF50", zorder=0)
 
    # Đường ngưỡng
    ax.axhline(lower, color="#E74C3C", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(upper, color="#E74C3C", linewidth=1.2, linestyle="--", alpha=0.7)
 
    # Điểm normal (xanh lá, nhỏ, trong suốt)
    ax.scatter(x_normal,  y_normal,
               c="#4CAF50", s=6, alpha=0.5, zorder=2, linewidths=0)
 
    # Điểm outlier (tam giác đỏ, to hơn)
    ax.scatter(x_outlier, y_outlier,
               c="#E74C3C", s=18, alpha=0.75, marker="^", zorder=3, linewidths=0)
 
    # Trục & nhãn
    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel("Word Count", fontsize=12, color="#333")
    ax.set_xlim(-200, len(df) + 200)
    ax.set_ylim(-500, df["word_count"].max() * 1.08)
    ax.tick_params(labelsize=10)
    ax.set_title("Scatter Plot – Từng bài báo", fontsize=13, pad=12,
                 color="#222", fontweight="normal")
 
    # Grid nhẹ
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#999")
    ax.set_axisbelow(True)
 
    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbb")
    ax.spines["bottom"].set_color("#bbb")
 
    # Legend (góc trên phải, giống slide)
    legend_handles = [
        mpatches.Patch(color="#4CAF50", label="Normal"),
        plt.scatter([], [], c="#E74C3C", marker="^", s=40,
                    label=f"Outliers ({n_outliers:,})"),
        plt.Line2D([0], [0], color="#E74C3C", linestyle="--",
                   linewidth=1.2, label=f"Lower: {int(lower)}"),
        plt.Line2D([0], [0], color="#E74C3C", linestyle="--",
                   linewidth=1.2, label=f"Upper: {int(upper)}"),
        mpatches.Patch(color="#4CAF50", alpha=0.2, label="Normal Zone"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="#ddd", handlelength=1.4)
 
    plt.tight_layout()
    out = os.path.join(out_dir, "collection_10_scatter_wordcount.png")
    fig.savefig(out, bbox_inches="tight", dpi=180, facecolor="white")
    plt.close()
    print(f"  ✓ collection_10_scatter_wordcount.png")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_extra_viz()
