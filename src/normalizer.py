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
    Xóa ký tự đặc biệt, emoji, giữ lại tiếng Việt, số, và dấu câu cơ bản.

    Args:
        text (str): Văn bản cần xử lý.

    Returns:
        str: Văn bản đã loại bỏ ký tự đặc biệt.
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

    # Giữ lại: chữ cái (bao gồm tiếng Việt có dấu), số, dấu câu cơ bản, khoảng trắng
    # Các ký tự tiếng Việt nằm trong Unicode range bao gồm dấu
    text = re.sub(
        r'[^\w\s.,!?;:\-–—\'\"()\[\]{}àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ'
        r'ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
        r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆ'
        r'ÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]',
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
    Áp dụng tất cả các bước chuẩn hóa theo thứ tự.

    Args:
        text (str): Văn bản thô cần chuẩn hóa.

    Returns:
        str: Văn bản đã chuẩn hóa hoàn toàn.
    """
    if not isinstance(text, str) or not text.strip():
        return text

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
            df[col] = df[col].astype(str).apply(
                lambda x: '' if x.startswith('data:image') else x
            )
            log_info(f"Đã xóa dữ liệu base64 trong cột '{col}'")

    return df


def normalize_dataframe(df, text_columns=None, config=None):
    """
    Chuẩn hóa toàn bộ DataFrame.
    Áp dụng normalize_text cho các cột văn bản chính.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        text_columns (list): Danh sách cột cần chuẩn hóa văn bản.
        config (dict, optional): Cấu hình tùy chỉnh.

    Returns:
        pd.DataFrame: DataFrame đã chuẩn hóa.
    """
    if text_columns is None:
        text_columns = ['title', 'description', 'main_content']

    log_info(f"Bắt đầu chuẩn hóa văn bản ({len(df):,} bản ghi)...")

    # Chuẩn hóa các cột văn bản chính
    for col in text_columns:
        if col in df.columns:
            log_info(f"  Đang chuẩn hóa cột '{col}'...")
            df[col] = df[col].apply(normalize_text)
            log_info(f"  ✔ Cột '{col}' đã chuẩn hóa xong")

    # Xóa dữ liệu base64 trong cột ảnh
    df = clean_base64_columns(df)

    log_success(f"Chuẩn hóa hoàn tất: {len(df):,} bản ghi")

    return df
