"""
Module Làm sạch Dữ liệu (Data Cleaning).
Loại bỏ các bản ghi không hợp lệ khỏi tập dữ liệu thô.

Chức năng:
    - Loại bỏ dòng có dữ liệu rỗng (title, main_content, url)
    - Loại bỏ URL không hợp lệ
    - Lọc bài viết quá ngắn (< 50 từ) hoặc quá dài (> 10.000 từ)
"""

import re
import pandas as pd
from logger import log_info, log_warning, log_success


def remove_empty_rows(df, required_columns=None):
    """
    Xóa các dòng có giá trị rỗng ở các cột bắt buộc.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        required_columns (list): Danh sách các cột bắt buộc phải có giá trị.

    Returns:
        pd.DataFrame: DataFrame đã loại bỏ các dòng rỗng.
    """
    if required_columns is None:
        required_columns = ['title', 'main_content', 'url']

    before = len(df)

    # Xóa các dòng có giá trị NaN hoặc chuỗi rỗng ở các cột bắt buộc
    for col in required_columns:
        if col in df.columns:
            # Thay thế chuỗi rỗng và chỉ có khoảng trắng bằng NaN
            df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)

    df = df.dropna(subset=[c for c in required_columns if c in df.columns])

    removed = before - len(df)
    log_info(f"Loại bỏ dòng rỗng: {removed:,} bản ghi (cột: {required_columns})")

    return df


def validate_urls(df, url_column='url'):
    """
    Kiểm tra và loại bỏ các URL không hợp lệ.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        url_column (str): Tên cột chứa URL.

    Returns:
        pd.DataFrame: DataFrame đã loại bỏ các URL lỗi.
    """
    if url_column not in df.columns:
        log_warning(f"Cột '{url_column}' không tồn tại trong DataFrame.")
        return df

    before = len(df)

    # Regex kiểm tra URL hợp lệ: phải bắt đầu bằng http:// hoặc https://
    url_pattern = re.compile(
        r'^https?://'           # Bắt đầu bằng http:// hoặc https://
        r'[a-zA-Z0-9.-]+'      # Domain
        r'\.[a-zA-Z]{2,}'       # TLD (ít nhất 2 ký tự)
    )

    # Lọc các URL hợp lệ
    valid_mask = df[url_column].astype(str).apply(
        lambda x: bool(url_pattern.match(x))
    )
    df = df[valid_mask]

    removed = before - len(df)
    log_info(f"Loại bỏ URL lỗi: {removed:,} bản ghi")

    return df


def filter_by_length(df, content_column='main_content', min_words=50, max_words=10000):
    """
    Lọc bài viết theo độ dài nội dung (số từ).

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        content_column (str): Tên cột chứa nội dung bài viết.
        min_words (int): Số từ tối thiểu (mặc định: 50).
        max_words (int): Số từ tối đa (mặc định: 10.000).

    Returns:
        pd.DataFrame: DataFrame đã lọc theo độ dài.
    """
    if content_column not in df.columns:
        log_warning(f"Cột '{content_column}' không tồn tại trong DataFrame.")
        return df

    before = len(df)

    # Đếm số từ trong nội dung
    word_counts = df[content_column].astype(str).apply(lambda x: len(x.split()))

    # Lọc bài quá ngắn
    too_short = (word_counts < min_words).sum()
    # Lọc bài quá dài
    too_long = (word_counts > max_words).sum()

    # Giữ lại bài viết trong ngưỡng
    mask = (word_counts >= min_words) & (word_counts <= max_words)
    df = df[mask]

    removed = before - len(df)
    log_info(f"Lọc theo độ dài ({min_words}-{max_words} từ): "
             f"loại {removed:,} bản ghi (quá ngắn: {too_short:,}, quá dài: {too_long:,})")

    return df


def clean_data(df, config=None):
    """
    Hàm tổng hợp chạy toàn bộ quy trình làm sạch dữ liệu.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu thô.
        config (dict, optional): Cấu hình tùy chỉnh.

    Returns:
        tuple: (DataFrame đã làm sạch, dict thống kê chi tiết)
    """
    if config is None:
        config = {}

    required_columns = config.get('required_columns', ['title', 'main_content', 'url'])
    min_words = config.get('min_words', 50)
    max_words = config.get('max_words', 10000)

    total_before = len(df)
    stats = {'total_before': total_before}

    log_info(f"Bắt đầu làm sạch dữ liệu ({total_before:,} bản ghi)...")

    # Bước 1: Loại bỏ dòng rỗng
    before_step = len(df)
    df = remove_empty_rows(df, required_columns)
    stats['empty_removed'] = before_step - len(df)

    # Bước 2: Loại bỏ URL lỗi
    before_step = len(df)
    df = validate_urls(df)
    stats['url_invalid'] = before_step - len(df)

    # Bước 3: Lọc theo độ dài
    before_step = len(df)
    # Đếm chi tiết trước khi lọc
    word_counts = df['main_content'].astype(str).apply(lambda x: len(x.split()))
    stats['too_short'] = int((word_counts < min_words).sum())
    stats['too_long'] = int((word_counts > max_words).sum())
    df = filter_by_length(df, min_words=min_words, max_words=max_words)

    # Reset index sau khi lọc
    df = df.reset_index(drop=True)

    stats['total_after'] = len(df)
    stats['total_removed'] = total_before - len(df)

    log_success(f"Làm sạch hoàn tất: còn lại {len(df):,} bản ghi")

    return df, stats


def visualize_cleaning_report(stats, save_dir=None):
    """
    Vẽ 3 biểu đồ riêng biệt trực quan hóa kết quả làm sạch dữ liệu.
    Lưu mỗi biểu đồ thành 1 file ảnh PNG riêng trong thư mục notebooks/.

    Args:
        stats (dict): Thống kê từ clean_data().
        save_dir (str): Thư mục lưu ảnh. Mặc định: notebooks/
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.family'] = 'Segoe UI'
    sns.set_theme(style="whitegrid", palette="muted")

    if save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        save_dir = os.path.join(project_root, 'notebooks')

    os.makedirs(save_dir, exist_ok=True)

    # ===================================================================
    # BIỂU ĐỒ 1: Bar chart chi tiết từng loại lỗi đã loại bỏ
    # ===================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle('Chi tiết các loại lỗi đã loại bỏ (Bước Làm sạch)', fontsize=14, fontweight='bold')

    categories = ['Dòng rỗng\n(NaN/empty)', 'URL lỗi', 'Quá ngắn\n(<50 từ)', 'Quá dài\n(>10000 từ)']
    values = [
        stats.get('empty_removed', 0),
        stats.get('url_invalid', 0),
        stats.get('too_short', 0),
        stats.get('too_long', 0)
    ]
    colors = ['#FF6B6B', '#FFB347', '#87CEEB', '#DDA0DD']

    bars = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Số bản ghi bị loại', fontsize=12)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    path1 = os.path.join(save_dir, 'cleaning_01_error_types.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    log_success(f"Biểu đồ 1 (Chi tiết lỗi): {path1}")

    # ===================================================================
    # BIỂU ĐỒ 2: Pie chart tỷ lệ giữ lại vs loại bỏ
    # ===================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig2.suptitle('Tỷ lệ Giữ lại vs Loại bỏ (Bước Làm sạch)', fontsize=14, fontweight='bold')

    total_kept = stats.get('total_after', 0)
    total_removed = stats.get('total_removed', 0)

    # Hàm định dạng hiển thị cả phần trăm và số lượng bài viết
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n({val:,} bài)'
        return my_autopct

    ax2.pie(
        [total_kept, total_removed],
        labels=['Giữ lại', 'Loại bỏ'],
        autopct=make_autopct([total_kept, total_removed]),
        colors=['#4ECDC4', '#FF6B6B'],
        startangle=90,
        explode=(0, 0.05),
        textprops={'fontsize': 13}
    )

    plt.tight_layout()
    path2 = os.path.join(save_dir, 'cleaning_02_pie_chart.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    log_success(f"Biểu đồ 2 (Pie chart): {path2}")

    # ===================================================================
    # BIỂU ĐỒ 3: So sánh trước/sau làm sạch
    # ===================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    fig3.suptitle('So sánh Trước/Sau Làm sạch Dữ liệu', fontsize=14, fontweight='bold')

    categories = ['Trước khi\nlàm sạch', 'Sau khi\nlàm sạch', 'Đã loại bỏ']
    values = [stats.get('total_before', 0), total_kept, total_removed]
    colors_bar = ['#45B7D1', '#4ECDC4', '#FF6B6B']

    bars = ax3.bar(categories, values, color=colors_bar, edgecolor='white', linewidth=1.5)
    ax3.set_ylabel('Số bản ghi', fontsize=12)

    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    path3 = os.path.join(save_dir, 'cleaning_03_comparison.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    log_success(f"Biểu đồ 3 (So sánh): {path3}")

