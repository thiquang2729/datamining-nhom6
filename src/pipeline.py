"""
Pipeline Xử lý Dữ liệu Tin tức Công nghệ.
Kết nối tất cả module thành một luồng xử lý xuyên suốt từ đầu đến cuối.

Pipeline gồm các bước:
    1. Đọc dữ liệu thô (news_data.csv)
    2. Làm sạch dữ liệu (cleaning.py)
    3. Chuẩn hóa văn bản (normalizer.py)
    4. Xử lý trùng lặp (deduplicator.py)
    5-8. Placeholder cho các thành viên khác (chờ tích hợp)

Output:
    - data/cleaned_news.csv: Dữ liệu đã làm sạch
"""

import os
import sys
import json
import time
import pandas as pd

# Thêm thư mục src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleaning import clean_data, visualize_cleaning_report
from normalizer import normalize_dataframe
from deduplicator import remove_duplicates, generate_dedup_report, visualize_dedup_report
from logger import (
    log_info, log_success, log_error, log_warning,
    log_step_end, log_summary, StepTimer
)


class DataPipeline:
    """Pipeline xử lý dữ liệu tin tức công nghệ."""

    def __init__(self, config_path=None):
        """
        Khởi tạo pipeline với cấu hình.

        Args:
            config_path (str): Đường dẫn đến file config.json.
        """
        # Tìm config.json ở thư mục gốc dự án
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, 'config.json')

        self.config = self._load_config(config_path)
        self.df = None
        self.step_stats = []

        # Đường dẫn dựa trên thư mục gốc dự án
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _load_config(self, config_path):
        """Đọc file cấu hình JSON."""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            log_info(f"Đã tải cấu hình từ: {config_path}")
            return config
        else:
            log_warning(f"Không tìm thấy file cấu hình: {config_path}. Dùng cấu hình mặc định.")
            return self._default_config()

    def _default_config(self):
        """Trả về cấu hình mặc định."""
        return {
            "input_path": "data/news_data.csv",
            "output_cleaned_path": "data/cleaned_news.csv",
            "output_processed_path": "data/processed_news.csv",
            "cleaning": {
                "required_columns": ["title", "main_content", "url"],
                "min_words": 50,
                "max_words": 10000
            },
            "deduplication": {
                "method": "md5",
                "column": "main_content",
                "keep": "first"
            },
            "encoding": "utf-8-sig"
        }

    def _get_path(self, relative_path):
        """Chuyển đường dẫn tương đối thành đường dẫn tuyệt đối."""
        return os.path.join(self.project_root, relative_path)

    def _save_checkpoint(self, step_name):
        """Lưu checkpoint sau mỗi bước."""
        if self.df is not None:
            checkpoint_dir = self._get_path('data/checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'{step_name}.csv')
            self.df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
            log_info(f"Đã lưu checkpoint: {checkpoint_path}")

    # ===================================================================
    # BƯỚC 1: ĐỌC DỮ LIỆU THÔ
    # ===================================================================
    def step_load_data(self):
        """Bước 1: Đọc dữ liệu thô từ CSV."""
        input_path = self._get_path(self.config['input_path'])
        encoding = self.config.get('encoding', 'utf-8-sig')

        with StepTimer("Bước 1: Đọc dữ liệu thô") as timer:
            try:
                self.df = pd.read_csv(input_path, encoding=encoding)
                before = len(self.df)
                log_success(f"Đã đọc {before:,} bản ghi từ {input_path}")
                log_info(f"Các cột: {list(self.df.columns)}")
                self.step_stats.append({
                    'step': 'load_data',
                    'before': before,
                    'after': before,
                    'time': timer.elapsed
                })
            except Exception as e:
                log_error(f"Lỗi khi đọc dữ liệu: {e}")
                raise

        return self

    # ===================================================================
    # BƯỚC 2: LÀM SẠCH DỮ LIỆU (Thành viên 2)
    # ===================================================================
    def step_clean(self):
        """Bước 2: Làm sạch dữ liệu - loại bỏ dòng rỗng, URL lỗi, bài quá ngắn/dài."""
        if self.df is None:
            log_error("Chưa có dữ liệu! Hãy chạy step_load_data() trước.")
            return self

        before = len(self.df)
        cleaning_config = self.config.get('cleaning', {})

        with StepTimer("Bước 2: Làm sạch dữ liệu", before) as timer:
            self.df, cleaning_stats = clean_data(self.df, cleaning_config)
            
            # Vẽ và lưu 3 biểu đồ báo cáo làm sạch dữ liệu vào thư mục notebooks/
            visualize_cleaning_report(cleaning_stats)
            
            after = len(self.df)
            log_step_end("Bước 2: Làm sạch", before, after, timer.elapsed)
            self.step_stats.append({
                'step': 'clean',
                'before': before,
                'after': after,
                'time': timer.elapsed
            })

        self._save_checkpoint('02_cleaned')
        return self

    # ===================================================================
    # BƯỚC 3: CHUẨN HÓA VĂN BẢN (Thành viên 2)
    # ===================================================================
    def step_normalize(self):
        """Bước 3: Chuẩn hóa Unicode, loại bỏ HTML, ký tự đặc biệt."""
        if self.df is None:
            log_error("Chưa có dữ liệu! Hãy chạy step_load_data() trước.")
            return self

        before = len(self.df)

        with StepTimer("Bước 3: Chuẩn hóa văn bản", before) as timer:
            self.df = normalize_dataframe(self.df)
            after = len(self.df)
            log_step_end("Bước 3: Chuẩn hóa", before, after, timer.elapsed)
            self.step_stats.append({
                'step': 'normalize',
                'before': before,
                'after': after,
                'time': timer.elapsed
            })

        self._save_checkpoint('03_normalized')
        return self

    # ===================================================================
    # BƯỚC 4: XỬ LÝ TRÙNG LẶP (Thành viên 2)
    # ===================================================================
    def step_deduplicate(self):
        """Bước 4: Loại bỏ bài viết trùng lặp bằng MD5 hash."""
        if self.df is None:
            log_error("Chưa có dữ liệu! Hãy chạy step_load_data() trước.")
            return self

        before = len(self.df)
        dedup_config = self.config.get('deduplication', {})
        column = dedup_config.get('column', 'main_content')
        keep = dedup_config.get('keep', 'first')

        # Lưu lại bản copy trước khi dedup để tạo báo cáo
        df_before_dedup = self.df.copy()

        with StepTimer("Bước 4: Xử lý trùng lặp", before) as timer:
            self.df = remove_duplicates(self.df, column=column, keep=keep)
            after = len(self.df)

            # Tạo báo cáo trùng lặp
            generate_dedup_report(df_before_dedup, self.df)

            # Tạo biểu đồ trực quan hóa và lưu ra file ảnh
            visualize_dedup_report(df_before_dedup, self.df)

            log_step_end("Bước 4: Loại trùng lặp", before, after, timer.elapsed)
            self.step_stats.append({
                'step': 'deduplicate',
                'before': before,
                'after': after,
                'time': timer.elapsed
            })

        self._save_checkpoint('04_deduplicated')
        return self

    # ===================================================================
    # BƯỚC 5-8: PLACEHOLDER (Chờ code các thành viên khác)
    # ===================================================================
    def step_preprocess_nlp(self):
        """Bước 5: Tiền xử lý NLP (Thành viên 3 - PLACEHOLDER)."""
        log_warning("Bước 5: Tiền xử lý NLP - PLACEHOLDER (chờ code Thành viên 3)")
        # TODO: Tích hợp code preprocessing.py của Thành viên 3
        # from preprocessing import preprocess_nlp
        # self.df = preprocess_nlp(self.df)
        return self

    def step_clustering(self):
        """Bước 6: Phân cụm & Gán nhãn (Thành viên 4 - PLACEHOLDER)."""
        log_warning("Bước 6: Phân cụm & Gán nhãn - PLACEHOLDER (chờ code Thành viên 4)")
        # TODO: Tích hợp code modeling.py của Thành viên 4
        # from modeling import cluster_and_label
        # self.df = cluster_and_label(self.df)
        return self

    def step_train_model(self):
        """Bước 7: Huấn luyện mô hình (Thành viên 5 - PLACEHOLDER)."""
        log_warning("Bước 7: Huấn luyện mô hình - PLACEHOLDER (chờ code Thành viên 5)")
        # TODO: Tích hợp code của Thành viên 5
        return self

    def step_export(self):
        """Bước 8: Xuất dữ liệu & Báo cáo (Thành viên 6 - PLACEHOLDER)."""
        log_warning("Bước 8: Xuất dữ liệu - PLACEHOLDER (chờ code Thành viên 6)")
        # TODO: Tích hợp code của Thành viên 6
        return self

    # ===================================================================
    # LƯU KẾT QUẢ
    # ===================================================================
    def save_output(self):
        """Lưu dữ liệu đã xử lý ra file CSV."""
        if self.df is None:
            log_error("Không có dữ liệu để lưu!")
            return self

        output_path = self._get_path(self.config['output_cleaned_path'])

        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        log_success(f"Đã lưu {len(self.df):,} bản ghi → {output_path}")

        return self

    # ===================================================================
    # CHẠY PIPELINE
    # ===================================================================
    def run_all(self):
        """
        Chạy toàn bộ pipeline từ đầu đến cuối.
        Hiện tại chỉ chạy bước 1-4 (phần của Thành viên 2).
        """
        total_start = time.time()

        log_info("🚀 BẮT ĐẦU PIPELINE XỬ LÝ DỮ LIỆU TIN TỨC CÔNG NGHỆ")
        log_info("="*60)

        try:
            # Các bước của Thành viên 2
            self.step_load_data()
            total_before = len(self.df)

            self.step_clean()
            self.step_normalize()
            self.step_deduplicate()

            # Lưu kết quả
            self.save_output()

            # Tổng kết
            total_after = len(self.df)
            total_time = time.time() - total_start
            log_summary(total_before, total_after, total_time)

        except Exception as e:
            log_error(f"Pipeline thất bại: {e}")
            raise

        return self

    def run_step(self, step_name):
        """
        Chạy một bước cụ thể.

        Args:
            step_name (str): Tên bước cần chạy.
        """
        steps = {
            'load_data': self.step_load_data,
            'clean': self.step_clean,
            'normalize': self.step_normalize,
            'deduplicate': self.step_deduplicate,
            'preprocess_nlp': self.step_preprocess_nlp,
            'clustering': self.step_clustering,
            'train_model': self.step_train_model,
            'export': self.step_export,
        }

        if step_name not in steps:
            log_error(f"Bước '{step_name}' không hợp lệ. "
                      f"Các bước có sẵn: {list(steps.keys())}")
            return self

        steps[step_name]()
        return self


# ===================================================================
# CHẠY TRỰC TIẾP
# ===================================================================
if __name__ == '__main__':
    pipeline = DataPipeline()
    pipeline.run_all()
