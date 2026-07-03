"""
visualize_tech.py - Trực quan hóa dữ liệu bài viết Công nghệ đã thu thập.

Tạo 3 biểu đồ:
  1. Số lượng bài viết theo domain (bar chart)
  2. Phân bổ tỉ lệ theo domain (pie chart)
  3. Top 15 sub-categories phổ biến nhất (horizontal bar)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "news_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

def load_data():
    # Sử dụng pandas để đọc file CSV
    df = pd.read_csv(CSV_PATH)
    
    total = len(df)
    
    # Số lượng bài viết theo domain
    domain_counts_series = df['domain'].value_counts()
    domain_counts = [(domain, count) for domain, count in domain_counts_series.items()]
    
    # Top 15 sub-categories phổ biến nhất
    valid_subcats = df[df['sub_category'].notna() & (df['sub_category'] != '')]
    subcat_series = valid_subcats['sub_category'].value_counts().head(15)
    sub_cat_counts = [(subcat, count) for subcat, count in subcat_series.items()]
    
    return total, domain_counts, sub_cat_counts


def plot_domain_bar(domain_counts, total):
    """Biểu đồ cột: số bài theo domain."""
    domains = [d[0] for d in domain_counts]
    counts = [d[1] for d in domain_counts]
    
    colors = ['#4C72B0', '#55A868', '#C44E52']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(domains, counts, color=colors[:len(domains)], edgecolor='white', linewidth=1.5)
    
    # Thêm label lên mỗi cột
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Số lượng bài viết Công nghệ theo nguồn báo\n(Tổng: {total:,} bài)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Nguồn báo', fontsize=13)
    ax.set_ylabel('Số bài viết', fontsize=13)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tech_domain_counts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Đã lưu: tech_domain_counts.png")


def plot_domain_pie(domain_counts, total):
    """Biểu đồ tròn: phân bổ tỉ lệ."""
    domains = [d[0] for d in domain_counts]
    counts = [d[1] for d in domain_counts]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts, labels=domains, autopct='%1.1f%%',
        colors=colors[:len(domains)],
        textprops={'fontsize': 13},
        startangle=90, pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontweight('bold')
    
    ax.set_title(f'Phân bổ bài viết Công nghệ theo nguồn\n(Tổng: {total:,} bài)',
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tech_domain_pie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Đã lưu: tech_domain_pie.png")


def plot_subcategories(sub_cat_counts):
    """Biểu đồ ngang: top sub-categories."""
    if not sub_cat_counts:
        print("Không có dữ liệu sub_category để vẽ.")
        return
    
    cats = [c[0] for c in reversed(sub_cat_counts)]
    counts = [c[1] for c in reversed(sub_cat_counts)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(cats, counts, color='#4C72B0', edgecolor='white', linewidth=0.8)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=10)
    
    ax.set_title('Top 15 Sub-categories phổ biến nhất', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Số bài viết', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tech_subcategories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Đã lưu: tech_subcategories.png")


def main():
    if not CSV_PATH.exists():
        print("Chưa có file news_data.csv. Hãy đảm bảo main.py đã lưu dữ liệu ra CSV.")
        return
    
    total, domain_counts, sub_cat_counts = load_data()
    
    if total == 0:
        print("Database rỗng. Hãy chạy scraper trước.")
        return
    
    print(f"\n{'='*50}")
    print(f"  THỐNG KÊ DỮ LIỆU CÔNG NGHỆ")
    print(f"{'='*50}")
    print(f"  Tổng bài viết: {total:,}")
    for d, c in domain_counts:
        print(f"  - {d}: {c:,} bài ({c/total*100:.1f}%)")
    print(f"{'='*50}\n")
    
    plot_domain_bar(domain_counts, total)
    plot_domain_pie(domain_counts, total)
    plot_subcategories(sub_cat_counts)
    
    print("\nHoàn tất trực quan hóa!")


if __name__ == "__main__":
    main()
