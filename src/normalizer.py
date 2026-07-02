"""
Module Chuẩn hóa Văn bản (Text Normalizer).
Chuẩn hóa Unicode, loại bỏ HTML tags, ký tự đặc biệt, và dữ liệu nhiễu.

Chức năng:
    - Chuẩn hóa Unicode NFC cho tiếng Việt
    - Xóa HTML tags và giải mã HTML entities
    - Loại bỏ emoji, ký tự đặc biệt
    - Xóa URL nhúng, dữ liệu base64
    - Chuẩn hóa khoảng trắng
"""

import re
import html
import unicodedata
import pandas as pd
from logger import log_info, log_success, log_warning


def normalize_unicode(text):
    """
    Chuẩn hóa Unicode NFC cho văn bản tiếng Việt.
    Đảm bảo các ký tự có dấu được biểu diễn nhất quán.

    Args:
        text (str): Văn bản cần chuẩn hóa.

    Returns:
        str: Văn bản đã chuẩn hóa Unicode NFC.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    return unicodedata.normalize('NFC', text)


def remove_html_tags(text):
    """
    Loại bỏ toàn bộ thẻ HTML và giải mã HTML entities.

    Args:
        text (str): Văn bản có thể chứa HTML.

    Returns:
        str: Văn bản đã loại bỏ HTML.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Giải mã HTML entities (vd: &amp; → &, &lt; → <)
    text = html.unescape(text)

    # Xóa toàn bộ thẻ HTML
    text = re.sub(r'<[^>]+>', '', text)

    return text


def remove_urls_in_text(text):
    """
    Loại bỏ các URL nhúng trong nội dung bài viết.

    Args:
        text (str): Văn bản có thể chứa URL.

    Returns:
        str: Văn bản đã loại bỏ URL nhúng.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Xóa URL http/https
    text = re.sub(r'https?://\S+', '', text)

    # Xóa dữ liệu base64
    text = re.sub(r'data:[a-zA-Z/]+;base64,[A-Za-z0-9+/=]+', '', text)

    return text


def remove_special_characters(text):
    """
    Xóa toàn bộ ký tự đặc biệt và dấu câu, chỉ giữ lại chữ cái (bao gồm chữ tiếng Việt có dấu),
    số và khoảng trắng.

    Args:
        text (str): Văn bản cần xử lý (đã chuyển thành chữ thường).

    Returns:
        str: Văn bản đã loại bỏ toàn bộ ký tự đặc biệt và dấu câu.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Xóa emoji và các biểu tượng Unicode đặc biệt
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Chỉ giữ lại: chữ cái tiếng Anh/tiếng Việt thường, số, và khoảng trắng
    text = re.sub(
        r'[^a-z0-9\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
        ' ',
        text
    )

    return text


def normalize_whitespace(text):
    """
    Chuẩn hóa khoảng trắng: loại bỏ multiple spaces, trim đầu/cuối.

    Args:
        text (str): Văn bản cần chuẩn hóa.

    Returns:
        str: Văn bản đã chuẩn hóa khoảng trắng.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Thay thế nhiều khoảng trắng liên tiếp bằng 1 khoảng trắng
    text = re.sub(r'\s+', ' ', text)

    # Trim đầu và cuối
    text = text.strip()

    return text


def normalize_text(text):
    """
    Pipeline chuẩn hóa đầy đủ cho một đoạn văn bản.
    Áp dụng tất cả các bước chuẩn hóa theo thứ tự: chuyển chữ thường,
    chuẩn hóa Unicode, loại bỏ HTML, xóa URL/base64, xóa ký tự đặc biệt/dấu câu,
    chuẩn hóa khoảng trắng.

    Args:
        text (str): Văn bản thô cần chuẩn hóa.

    Returns:
        str: Văn bản đã chuẩn hóa hoàn toàn (chữ thường, sạch ký tự đặc biệt).
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Chuyển văn bản thành chữ thường
    text = text.lower()

    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = remove_urls_in_text(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    return text


def clean_base64_columns(df, columns=None):
    """
    Xóa dữ liệu base64/ảnh nhúng trong các cột chỉ định.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        columns (list): Danh sách cột cần xử lý.

    Returns:
        pd.DataFrame: DataFrame đã xóa dữ liệu base64.
    """
    if columns is None:
        columns = ['thumbnail_url', 'images']

    for col in columns:
        if col in df.columns:
            # Thay thế chuỗi base64 bằng chuỗi rỗng
            df[col] = df[col].fillna('').astype(str).apply(
                lambda x: '' if x.startswith('data:image') else x
            )
            log_info(f"Đã xóa dữ liệu base64 trong cột '{col}'")

    return df


def normalize_dataframe(df, text_columns=None, config=None):
    """
    Chuẩn hóa toàn bộ DataFrame.
    Áp dụng normalize_text cho các cột văn bản chính, chuyển các cột văn bản khác sang chữ thường,
    và lọc giữ lại chỉ 8 trường cụ thể.

    Trả về thêm dict thống kê để phục vụ trực quan hóa.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        text_columns (list): Danh sách cột cần chuẩn hóa văn bản.
        config (dict, optional): Cấu hình tùy chỉnh.

    Returns:
        tuple: (pd.DataFrame đã chuẩn hóa, dict thống kê)
    """
    if text_columns is None:
        text_columns = ['title', 'description', 'main_content']

    log_info(f"Bắt đầu chuẩn hóa văn bản ({len(df):,} bản ghi)...")

    # Thu thập thống kê trước khi chuẩn hóa
    stats = {
        'total_records': len(df),
        'html_tags': 0,
        'urls_removed': 0,
        'base64_removed': 0,
        'emoji_removed': 0,
        'special_chars_removed': 0,
        'total_chars_before': 0,
        'total_chars_after': 0,
    }

    # Đếm các loại nhiễu trước khi chuẩn hóa
    for col in text_columns:
        if col in df.columns:
            col_text = df[col].astype(str)
            stats['total_chars_before'] += col_text.str.len().sum()

            # Đếm HTML tags
            stats['html_tags'] += int(col_text.str.count(r'<[^>]+>').sum())

            # Đếm URL nhúng
            stats['urls_removed'] += int(col_text.str.count(r'https?://\S+').sum())

            # Đếm base64
            stats['base64_removed'] += int(col_text.str.count(r'data:[a-zA-Z/]+;base64,').sum())

            # Đếm emoji
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF"
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE
            )
            stats['emoji_removed'] += int(col_text.apply(
                lambda x: len(emoji_pattern.findall(x))
            ).sum())

    # Chuẩn hóa các cột văn bản chính
    for col in text_columns:
        if col in df.columns:
            log_info(f"  Đang chuẩn hóa cột '{col}'...")
            df[col] = df[col].apply(normalize_text)
            log_info(f"  ✔ Cột '{col}' đã chuẩn hóa xong")

    # Đếm tổng ký tự sau chuẩn hóa
    for col in text_columns:
        if col in df.columns:
            stats['total_chars_after'] += int(df[col].astype(str).str.len().sum())

    stats['special_chars_removed'] = max(0, stats['total_chars_before'] - stats['total_chars_after']
                                          - stats['html_tags'] * 5  # ước lượng trung bình 5 ký tự/tag
                                          - stats['urls_removed'] * 30  # ước lượng trung bình 30 ký tự/URL
                                          )

    # Chuyển các trường văn bản khác được giữ lại thành chữ thường
    other_text_cols = ['tags', 'category', 'sub_category']
    for col in other_text_cols:
        if col in df.columns:
            log_info(f"  Đang chuyển cột '{col}' thành chữ thường...")
            df[col] = df[col].astype(str).str.lower()
            log_info(f"  ✔ Cột '{col}' đã chuyển sang chữ thường")

    log_success(f"Chuẩn hóa hoàn tất: {len(df):,} bản ghi")

    return df, stats


def visualize_normalize_report(stats, save_dir=None):
    """
    Vẽ 3 biểu đồ trực quan hóa kết quả chuẩn hóa văn bản.
    Lưu mỗi biểu đồ thành 1 file ảnh PNG riêng trong thư mục notebooks/.

    Biểu đồ:
        1. Bar chart: Số lượng từng loại nhiễu đã loại bỏ
        2. Bar chart: So sánh tổng ký tự trước/sau chuẩn hóa
        3. Pie chart: Tỷ lệ phân bổ các loại nhiễu

    Args:
        stats (dict): Thống kê từ normalize_dataframe().
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
    # BIỂU ĐỒ 1: Bar chart - Số lượng từng loại nhiễu đã loại bỏ
    # ===================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle('Chi tiết các loại nhiễu đã loại bỏ (Bước Chuẩn hóa)',
                  fontsize=14, fontweight='bold')

    categories = [
        'HTML Tags',
        'URL nhúng',
        'Dữ liệu\nBase64',
        'Emoji',
    ]
    values = [
        stats.get('html_tags', 0),
        stats.get('urls_removed', 0),
        stats.get('base64_removed', 0),
        stats.get('emoji_removed', 0),
    ]
    colors = ['#E74C3C', '#3498DB', '#9B59B6', '#F39C12']

    bars = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Số lượng đã loại bỏ', fontsize=12)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(values) * 0.01,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    path1 = os.path.join(save_dir, 'normalize_01_noise_types.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    log_success(f"Biểu đồ Normalize 1 (Loại nhiễu): {path1}")

    # ===================================================================
    # BIỂU ĐỒ 2: Bar chart - So sánh tổng ký tự trước/sau chuẩn hóa
    # ===================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.suptitle('So sánh Tổng ký tự Trước/Sau Chuẩn hóa', fontsize=14, fontweight='bold')

    chars_before = stats.get('total_chars_before', 0)
    chars_after = stats.get('total_chars_after', 0)
    chars_removed = chars_before - chars_after

    cat = ['Trước\nchuẩn hóa', 'Sau\nchuẩn hóa', 'Đã loại bỏ']
    vals = [chars_before, chars_after, chars_removed]
    colors_bar = ['#45B7D1', '#4ECDC4', '#FF6B6B']

    bars = ax2.bar(cat, vals, color=colors_bar, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Tổng số ký tự', fontsize=12)

    for bar, val in zip(bars, vals):
        if val >= 1_000_000:
            label = f'{val/1_000_000:.1f}M'
        elif val >= 1_000:
            label = f'{val/1_000:.0f}K'
        else:
            label = f'{val:,}'
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(vals) * 0.01,
                 label, ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    path2 = os.path.join(save_dir, 'normalize_02_chars_comparison.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    log_success(f"Biểu đồ Normalize 2 (Ký tự trước/sau): {path2}")

    # ===================================================================
    # BIỂU ĐỒ 3: Pie chart - Tỷ lệ phân bổ các loại nhiễu
    # ===================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    fig3.suptitle('Tỷ lệ phân bổ các loại nhiễu đã loại bỏ', fontsize=14, fontweight='bold')

    noise_labels = ['HTML Tags', 'URL nhúng', 'Base64', 'Emoji']
    noise_values = [
        stats.get('html_tags', 0),
        stats.get('urls_removed', 0),
        stats.get('base64_removed', 0),
        stats.get('emoji_removed', 0),
    ]
    noise_colors = ['#E74C3C', '#3498DB', '#9B59B6', '#F39C12']

    # Lọc bỏ các loại có giá trị 0
    filtered = [(l, v, c) for l, v, c in zip(noise_labels, noise_values, noise_colors) if v > 0]

    if filtered:
        f_labels, f_values, f_colors = zip(*filtered)

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f'{pct:.1f}%\n({val:,})'
            return my_autopct

        ax3.pie(
            f_values,
            labels=f_labels,
            autopct=make_autopct(f_values),
            colors=f_colors,
            startangle=90,
            textprops={'fontsize': 12}
        )
    else:
        ax3.text(0.5, 0.5, 'Không có nhiễu phát hiện',
                 ha='center', va='center', fontsize=14, transform=ax3.transAxes)

    plt.tight_layout()
    path3 = os.path.join(save_dir, 'normalize_03_noise_distribution.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    log_success(f"Biểu đồ Normalize 3 (Phân bổ nhiễu): {path3}")

