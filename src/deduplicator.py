"""
Module Xử lý Trùng lặp (Deduplication).
Phát hiện và loại bỏ các bài viết bị lặp lại nhiều lần trong tập dữ liệu.

Phương pháp: Sử dụng pandas.DataFrame.duplicated() để so sánh trực tiếp
giá trị các cột, kiểm tra xem bài viết có bị lặp lại hay không.

Chức năng:
    - Phát hiện bài viết trùng lặp chính xác theo nội dung
    - Phát hiện bài viết trùng lặp theo tiêu đề
    - Tạo báo cáo thống kê trùng lặp
    - Trực quan hóa kết quả (dùng trong notebook)
"""

import pandas as pd
from logger import log_info, log_warning, log_success


def find_exact_duplicates(df, column='main_content'):
    """
    Phát hiện các bài viết có nội dung bị lặp lại bằng pandas.duplicated().

    So sánh trực tiếp giá trị cột để tìm các bản ghi trùng nhau,
    giữ lại bản ghi xuất hiện đầu tiên.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        column (str): Cột dùng để kiểm tra trùng lặp.

    Returns:
        tuple: (DataFrame, Series mask các bản ghi trùng lặp)
    """
    if column not in df.columns:
        log_warning(f"Cột '{column}' không tồn tại trong DataFrame.")
        return df, pd.Series([False] * len(df))

    log_info(f"Đang kiểm tra trùng lặp nội dung trên cột '{column}'...")

    # Tìm các bản ghi trùng lặp trực tiếp (giữ bản ghi đầu tiên)
    duplicated_mask = df.duplicated(subset=[column], keep='first')

    dup_count = duplicated_mask.sum()
    log_info(f"Trùng lặp nội dung: {dup_count:,} bản ghi bị lặp lại")

    return df, duplicated_mask



def remove_duplicates(df, column='main_content', keep='first'):
    """
    Loại bỏ các bài viết trùng lặp, giữ lại bản ghi đầu tiên.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        column (str): Cột dùng để kiểm tra trùng lặp.
        keep (str): Chiến lược giữ bản ghi ('first').

    Returns:
        pd.DataFrame: DataFrame đã loại bỏ trùng lặp.
    """
    before = len(df)

    # Tìm các bản ghi trùng lặp
    df, dup_mask = find_exact_duplicates(df, column)

    # Loại bỏ các bản ghi trùng
    df = df[~dup_mask]

    # Reset index
    df = df.reset_index(drop=True)

    removed = before - len(df)
    log_success(f"Loại bỏ trùng lặp: {removed:,} bản ghi (giữ bản ghi {keep})")

    return df


def generate_dedup_report(df_before, df_after, domain_column='domain'):
    """
    Tạo báo cáo thống kê về quá trình loại bỏ trùng lặp.

    Args:
        df_before (pd.DataFrame): DataFrame trước khi loại trùng.
        df_after (pd.DataFrame): DataFrame sau khi loại trùng.
        domain_column (str): Cột chứa tên domain nguồn.

    Returns:
        dict: Báo cáo thống kê chi tiết.
    """
    total_before = len(df_before)
    total_after = len(df_after)
    total_removed = total_before - total_after
    dup_rate = (total_removed / total_before * 100) if total_before > 0 else 0

    report = {
        'total_before': total_before,
        'total_after': total_after,
        'total_removed': total_removed,
        'duplicate_rate': round(dup_rate, 2),
    }

    # Thống kê theo domain
    if domain_column in df_before.columns and domain_column in df_after.columns:
        before_counts = df_before[domain_column].value_counts()
        after_counts = df_after[domain_column].value_counts()

        domain_stats = {}
        for domain in before_counts.index:
            b = before_counts.get(domain, 0)
            a = after_counts.get(domain, 0)
            domain_stats[domain] = {
                'before': int(b),
                'after': int(a),
                'removed': int(b - a),
                'rate': round((b - a) / b * 100, 2) if b > 0 else 0
            }
        report['domain_stats'] = domain_stats

    # In báo cáo
    log_info("="*50)
    log_info("BÁO CÁO TRÙNG LẶP")
    log_info("="*50)
    log_info(f"  Tổng trước:    {total_before:,}")
    log_info(f"  Tổng sau:      {total_after:,}")
    log_info(f"  Đã loại bỏ:    {total_removed:,}")
    log_info(f"  Tỷ lệ trùng:   {dup_rate:.2f}%")

    if 'domain_stats' in report:
        log_info("\n  Theo domain:")
        for domain, stats in sorted(report['domain_stats'].items(),
                                     key=lambda x: x[1]['removed'],
                                     reverse=True):
            log_info(f"    {domain}: {stats['before']} → {stats['after']} "
                     f"(loại {stats['removed']}, {stats['rate']}%)")

    return report


def visualize_dedup_report(df_before, df_after, domain_column='domain', save_dir=None):
    """
    Vẽ 3 biểu đồ riêng biệt trực quan hóa kết quả dedup.
    Lưu mỗi biểu đồ thành 1 file ảnh PNG riêng trong thư mục notebooks/.

    Args:
        df_before (pd.DataFrame): DataFrame trước khi loại trùng.
        df_after (pd.DataFrame): DataFrame sau khi loại trùng.
        domain_column (str): Cột chứa tên domain nguồn.
        save_dir (str): Thư mục lưu ảnh. Mặc định: notebooks/
    """
    import os
    import matplotlib
    matplotlib.use('Agg')  # Dùng backend không cần GUI
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.family'] = 'Segoe UI'
    sns.set_theme(style="whitegrid", palette="muted")

    # Xác định thư mục lưu ảnh
    if save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        save_dir = os.path.join(project_root, 'notebooks')

    os.makedirs(save_dir, exist_ok=True)

    total_removed = len(df_before) - len(df_after)
    total_kept = len(df_after)

    # ===================================================================
    # BIỂU ĐỒ 1: Bar chart số lượng bản ghi theo domain (trước/sau)
    # ===================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle('Số lượng bản ghi theo Domain (Trước/Sau loại trùng)', fontsize=14, fontweight='bold')

    if domain_column in df_before.columns:
        before_counts = df_before[domain_column].value_counts()
        after_counts = df_after[domain_column].value_counts()

        domains = before_counts.index[:10]
        before_vals = [before_counts.get(d, 0) for d in domains]
        after_vals = [after_counts.get(d, 0) for d in domains]

        x = range(len(domains))
        width = 0.35

        ax1.bar([i - width/2 for i in x], before_vals, width, label='Trước', color='#FF6B6B')
        ax1.bar([i + width/2 for i in x], after_vals, width, label='Sau', color='#4ECDC4')
        ax1.set_xlabel('Domain', fontsize=12)
        ax1.set_ylabel('Số bản ghi', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([d[:20] for d in domains], rotation=45, ha='right')
        ax1.legend(fontsize=11)

        # Thêm số liệu trên cột
        for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
            ax1.text(i - width/2, bv + 30, f'{bv:,}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, av + 30, f'{av:,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(save_dir, 'dedup_01_domain_bar.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    log_success(f"Biểu đồ 1 (Bar chart domain): {path1}")

    # ===================================================================
    # BIỂU ĐỒ 2: Pie chart tỷ lệ trùng lặp
    # ===================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig2.suptitle('Tỷ lệ Trùng lặp Dữ liệu', fontsize=14, fontweight='bold')

    # Hàm định dạng hiển thị cả phần trăm và số lượng bài viết
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n({val:,} bài)'
        return my_autopct

    ax2.pie(
        [total_kept, total_removed],
        labels=['Giữ lại', 'Trùng lặp'],
        autopct=make_autopct([total_kept, total_removed]),
        colors=['#4ECDC4', '#FF6B6B'],
        startangle=90,
        explode=(0, 0.05),
        textprops={'fontsize': 13}
    )

    plt.tight_layout()
    path2 = os.path.join(save_dir, 'dedup_02_pie_chart.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    log_success(f"Biểu đồ 2 (Pie chart): {path2}")

    # ===================================================================
    # BIỂU ĐỒ 3: So sánh trước/sau deduplication
    # ===================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    fig3.suptitle('So sánh Trước/Sau Loại trùng lặp', fontsize=14, fontweight='bold')

    categories = ['Trước khi\nloại trùng', 'Sau khi\nloại trùng', 'Đã loại bỏ']
    values = [len(df_before), len(df_after), total_removed]
    colors_bar = ['#45B7D1', '#4ECDC4', '#FF6B6B']

    bars = ax3.bar(categories, values, color=colors_bar, edgecolor='white', linewidth=1.5)
    ax3.set_ylabel('Số bản ghi', fontsize=12)

    # Thêm số liệu trên mỗi cột
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    path3 = os.path.join(save_dir, 'dedup_03_comparison.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    log_success(f"Biểu đồ 3 (So sánh): {path3}")


