"""
Module Tích hợp Dữ liệu (Data Integration).
Đọc dữ liệu từ 3 nguồn báo (VnExpress, Thanh Niên, Vietnamnet), 
ghép lại thành một DataFrame duy nhất và chỉ giữ lại 7 trường quan trọng.
"""

import os
import pandas as pd
from logger import log_info, log_success, log_warning, log_error


def integrate_data(source_paths, keep_columns, output_path=None, encoding='utf-8-sig'):
    """
    Tích hợp dữ liệu từ nhiều nguồn CSV riêng biệt thành một DataFrame duy nhất.
    Chỉ giữ lại các cột cần thiết.

    Args:
        source_paths (list): Danh sách đường dẫn tới các file CSV nguồn.
        keep_columns (list): Danh sách các cột cần giữ lại.
        output_path (str, optional): Đường dẫn lưu kết quả tích hợp.
        encoding (str): Encoding khi đọc/ghi file CSV.

    Returns:
        pd.DataFrame: DataFrame sau khi tích hợp và lọc cột.
    """
    log_info(f"Bắt đầu tích hợp dữ liệu từ {len(source_paths)} nguồn...")
    dataframes = []

    for path in source_paths:
        if not os.path.exists(path):
            log_warning(f"Không tìm thấy file nguồn: {path}. Bỏ qua file này.")
            continue

        try:
            df_source = pd.read_csv(path, encoding=encoding)
            log_info(f"  Đã đọc {len(df_source):,} bản ghi từ: {path}")
            dataframes.append(df_source)
        except Exception as e:
            log_error(f"  Lỗi khi đọc file {path}: {e}")

    if not dataframes:
        raise ValueError("Không có dữ liệu nguồn nào được đọc thành công để tích hợp!")

    # Ghép tất cả các DataFrame
    integrated_df = pd.concat(dataframes, ignore_index=True)
    total_raw = len(integrated_df)
    log_info(f"Tổng số bản ghi sau khi ghép thô: {total_raw:,} bản ghi")

    # Lọc chỉ giữ lại các cột được yêu cầu
    keep_present = [col for col in keep_columns if col in integrated_df.columns]
    missing_cols = [col for col in keep_columns if col not in integrated_df.columns]

    if missing_cols:
        log_warning(f"Các cột yêu cầu không tồn tại trong dữ liệu thô: {missing_cols}")

    log_info(f"Đang lọc DataFrame chỉ giữ lại các cột: {keep_present}")
    integrated_df = integrated_df[keep_present]

    # Điền giá trị rỗng cho các cột bị thiếu nhưng bắt buộc phải có cấu trúc
    for col in missing_cols:
        integrated_df[col] = pd.NA

    # Lưu xuống file nếu có yêu cầu
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        integrated_df.to_csv(output_path, index=False, encoding=encoding)
        log_success(f"Đã lưu kết quả tích hợp ({len(integrated_df):,} bản ghi) → {output_path}")

    return integrated_df
