"""
Pipeline Xử lý Dữ liệu Tin tức Công nghệ.
Kết nối tất cả module thành một luồng xử lý xuyên suốt từ đầu đến cuối.

Pipeline gồm các bước:
    1. Đọc dữ liệu thô (news_data.csv)
    2. Làm sạch dữ liệu (cleaning.py)
    3. Chuẩn hóa văn bản (normalizer.py)
    4. Xử lý trùng lặp (deduplicator.py)
    5. Tiền xử lý NLP và trích xuất đặc trưng TF-IDF (preprocessing.py)
    6-8. Placeholder cho các thành viên khác (chờ tích hợp)

Output:
    - data/cleaned_news.csv: Dữ liệu đã làm sạch
    - data/processed_news.csv: Dữ liệu đã thêm cột processed_content
    - data/tfidf_features.pkl: Ma trận đặc trưng TF-IDF
    - models/vectorizer.pkl: Vectorizer đã fit
"""

import os
import sys
import json
import time
import pandas as pd

# Thêm thư mục src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleaning import clean_data, visualize_cleaning_report
from normalizer import normalize_dataframe, visualize_normalize_report
from deduplicator import remove_duplicates, generate_dedup_report, visualize_dedup_report
from preprocessing import preprocess_nlp
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
                "column": "main_content",
                "keep": "first"
            },
            "preprocessing": {
                "stopword_path": "data/vietnamese-stopwords.txt",
                "content_column": "main_content",
                "processed_column": "processed_content",
                "tfidf_matrix_path": "data/tfidf_features.pkl",
                "vectorizer_path": "models/vectorizer.pkl",
                "n_components": None,
                "max_features": None
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
            
            # Vẽ và lưu biểu đồ báo cáo làm sạch dữ liệu vào thư mục notebooks/
            visualize_cleaning_report(cleaning_stats, df_cleaned=self.df)
            
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
            self.df, normalize_stats = normalize_dataframe(self.df)

            # Vẽ và lưu 3 biểu đồ báo cáo chuẩn hóa văn bản vào thư mục notebooks/
            visualize_normalize_report(normalize_stats)

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
        """Bước 4: Loại bỏ bài viết trùng lặp (kiểm tra bài viết lặp lại)."""
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
    # BƯỚC 5: TIỀN XỬ LÝ NLP
    # ===================================================================
    def step_preprocess_nlp(self):
        """Bước 5: Tiền xử lý NLP và trích xuất đặc trưng TF-IDF."""
        if self.df is None:
            log_error("Chưa có dữ liệu! Hãy chạy các bước trước khi tiền xử lý NLP.")
            return self

        before = len(self.df)
        preprocessing_config = self.config.get('preprocessing', {})
        stopword_path = self._get_path(
            preprocessing_config.get('stopword_path', 'data/vietnamese-stopwords.txt')
        )
        text_column = preprocessing_config.get('content_column', 'main_content')
        processed_column = preprocessing_config.get('processed_column', 'processed_content')
        output_matrix_path = self._get_path(
            preprocessing_config.get('tfidf_matrix_path', 'data/tfidf_features.pkl')
        )
        output_vectorizer_path = self._get_path(
            preprocessing_config.get('vectorizer_path', 'models/vectorizer.pkl')
        )
        n_components = preprocessing_config.get('n_components')
        max_features = preprocessing_config.get('max_features')

        with StepTimer("Bước 5: Tiền xử lý NLP", before) as timer:
            self.df, X_features = preprocess_nlp(
                self.df,
                stopword_path=stopword_path,
                text_column=text_column,
                processed_column=processed_column,
                output_matrix_path=output_matrix_path,
                output_vectorizer_path=output_vectorizer_path,
                n_components=n_components,
                max_features=max_features,
            )

            after = len(self.df)
            log_success(f"Đã tạo ma trận TF-IDF: {X_features.shape}")
            log_step_end("Bước 5: Tiền xử lý NLP", before, after, timer.elapsed)
            self.step_stats.append({
                'step': 'preprocess_nlp',
                'before': before,
                'after': after,
                'time': timer.elapsed
            })

        self._save_checkpoint('05_preprocessed')
        return self

    # ===================================================================
    # BƯỚC 6-8: PLACEHOLDER (Chờ code các thành viên khác)
    # ===================================================================
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
    def save_output(self, output_config_key='output_cleaned_path'):
        """Lưu dữ liệu đã xử lý ra file CSV."""
        if self.df is None:
            log_error("Không có dữ liệu để lưu!")
            return self

        output_path = self._get_path(
            self.config.get(output_config_key, self.config['output_cleaned_path'])
        )

        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        log_success(f"Đã lưu {len(self.df):,} bản ghi → {output_path}")

        return self

    # ===================================================================
    # TRỰC QUAN HÓA PIPELINE TỔNG THỂ
    # ===================================================================
    def _visualize_pipeline_funnel(self, save_dir=None):
        """
        Vẽ biểu đồ Waterfall/Funnel thể hiện số bản ghi giảm dần qua từng bước pipeline.
        Lưu ra file ảnh PNG trong thư mục notebooks/.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams['font.family'] = 'Segoe UI'
        sns.set_theme(style="whitegrid", palette="muted")

        if save_dir is None:
            save_dir = os.path.join(self.project_root, 'notebooks')
        os.makedirs(save_dir, exist_ok=True)

        if not self.step_stats:
            log_warning("Không có thống kê pipeline để vẽ biểu đồ.")
            return

        # Thu thập dữ liệu từ step_stats
        step_names = []
        step_values = []
        step_name_map = {
            'load_data': 'Dữ liệu thô',
            'clean': 'Làm sạch',
            'normalize': 'Chuẩn hóa',
            'deduplicate': 'Loại trùng',
            'preprocess_nlp': 'NLP',
        }

        for stat in self.step_stats:
            name = step_name_map.get(stat['step'], stat['step'])
            if stat['step'] == 'load_data':
                step_names.append(name)
                step_values.append(stat['after'])
            else:
                step_names.append(name)
                step_values.append(stat['after'])

        # Vẽ waterfall chart
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Pipeline Xử lý Dữ liệu — Số bản ghi qua từng bước',
                     fontsize=14, fontweight='bold')

        colors = []
        for i, val in enumerate(step_values):
            if i == 0:
                colors.append('#45B7D1')  # Xanh dương - dữ liệu gốc
            elif i == len(step_values) - 1:
                colors.append('#4ECDC4')  # Xanh lá - kết quả cuối
            else:
                colors.append('#FFA07A')  # Cam nhạt - các bước trung gian

        bars = ax.bar(step_names, step_values, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Số bản ghi', fontsize=12)

        # Thêm số liệu và delta trên mỗi cột
        for i, (bar, val) in enumerate(zip(bars, step_values)):
            # Số bản ghi
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(step_values) * 0.01,
                    f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

            # Delta so với bước trước
            if i > 0:
                delta = step_values[i] - step_values[i - 1]
                if delta != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() * 0.5,
                            f'{delta:,}', ha='center', va='center',
                            fontsize=9, color='#C0392B', fontweight='bold')

        # Vẽ đường nối giữa các cột
        for i in range(len(step_values) - 1):
            ax.plot([i + 0.4, i + 0.6], [step_values[i], step_values[i + 1]],
                    color='#999', linewidth=1.5, linestyle='--', alpha=0.6)

        plt.tight_layout()
        path = os.path.join(save_dir, 'pipeline_funnel.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log_success(f"Biểu đồ Pipeline Funnel: {path}")

    # ===================================================================
    # CHẠY PIPELINE
    # ===================================================================
    def run_all(self):
        """
        Chạy toàn bộ pipeline từ đầu đến cuối.
        Chạy các bước đã tích hợp: làm sạch, chuẩn hóa, loại trùng và NLP.
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

            # Lưu dữ liệu sau làm sạch/dedup, sau đó tiếp tục bước NLP
            self.save_output('output_cleaned_path')

            self.step_preprocess_nlp()
            self.save_output('output_processed_path')

            # Vẽ biểu đồ pipeline funnel tổng thể
            self._visualize_pipeline_funnel()

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
    # CHẠY THEO THÀNH VIÊN
    # ===================================================================
    def run_tv1(self):
        """
        Chạy phần của Thành viên 1 (Ngô Hoàng Anh): Thu thập & EDA.
        Đọc dữ liệu thô và vẽ biểu đồ thống kê cơ bản.
        """
        total_start = time.time()
        log_info("🔹 CHẠY PHẦN THÀNH VIÊN 1: Data Collection & EDA")
        log_info("=" * 60)

        try:
            self.step_load_data()

            # Gọi visualize_collection nếu có
            try:
                from visualize_collection import run_collection_viz
                run_collection_viz()
            except ImportError:
                log_warning("Không tìm thấy module visualize_collection.")
            try:
                from visualize_extra import run_extra_viz
                run_extra_viz()
            except ImportError:
                log_warning("Không tìm thấy module visualize_extra.")

            total_time = time.time() - total_start
            log_success(f"Phần TV1 hoàn tất trong {total_time:.1f}s")
        except Exception as e:
            log_error(f"Phần TV1 thất bại: {e}")
            raise

        return self

    def run_tv2(self):
        """
        Chạy phần của Thành viên 2 (Lê Quang Thi): Làm sạch, Chuẩn hóa, Loại trùng.
        Bước 1-4: load → clean → normalize → deduplicate → lưu cleaned_news.csv
        """
        total_start = time.time()
        log_info("🔹 CHẠY PHẦN THÀNH VIÊN 2: Cleaning & Normalization & Dedup")
        log_info("=" * 60)

        try:
            self.step_load_data()
            total_before = len(self.df)

            self.step_clean()
            self.step_normalize()
            self.step_deduplicate()

            self.save_output('output_cleaned_path')

            # Vẽ biểu đồ pipeline funnel
            self._visualize_pipeline_funnel()

            total_after = len(self.df)
            total_time = time.time() - total_start
            log_summary(total_before, total_after, total_time)
        except Exception as e:
            log_error(f"Phần TV2 thất bại: {e}")
            raise

        return self

    def run_tv3(self):
        """
        Chạy phần của Thành viên 3 (Tôn Hoàng Nhớ): Tiền xử lý NLP & TF-IDF.
        Đọc dữ liệu đã làm sạch → NLP → lưu processed_news.csv + tfidf_features.pkl
        """
        total_start = time.time()
        log_info("🔹 CHẠY PHẦN THÀNH VIÊN 3: NLP Preprocessing & TF-IDF")
        log_info("=" * 60)

        try:
            # Đọc dữ liệu đã làm sạch (output của TV2)
            cleaned_path = self._get_path(self.config.get('output_cleaned_path', 'data/cleaned_news.csv'))
            encoding = self.config.get('encoding', 'utf-8-sig')

            if os.path.exists(cleaned_path):
                self.df = pd.read_csv(cleaned_path, encoding=encoding)
                log_info(f"Đã đọc {len(self.df):,} bản ghi từ {cleaned_path}")
                self.step_stats.append({
                    'step': 'load_data',
                    'before': len(self.df),
                    'after': len(self.df),
                    'time': 0
                })
            else:
                log_error(f"Không tìm thấy file cleaned: {cleaned_path}. Hãy chạy TV2 trước!")
                return self

            self.step_preprocess_nlp()
            self.save_output('output_processed_path')

            total_time = time.time() - total_start
            log_success(f"Phần TV3 hoàn tất trong {total_time:.1f}s")
        except Exception as e:
            log_error(f"Phần TV3 thất bại: {e}")
            raise

        return self

    def run_tv4(self):
        """
        Chạy phần của Thành viên 4 (Nguyễn Văn Trường): Clustering & Gán nhãn.
        Đọc dữ liệu đã xử lý NLP → phân cụm → gán nhãn.
        """
        total_start = time.time()
        log_info("🔹 CHẠY PHẦN THÀNH VIÊN 4: Clustering & Labeling")
        log_info("=" * 60)

        try:
            # Đọc dữ liệu đã xử lý NLP (output của TV3)
            processed_path = self._get_path(self.config.get('output_processed_path', 'data/processed_news.csv'))
            encoding = self.config.get('encoding', 'utf-8-sig')

            if os.path.exists(processed_path):
                self.df = pd.read_csv(processed_path, encoding=encoding)
                log_info(f"Đã đọc {len(self.df):,} bản ghi từ {processed_path}")
            else:
                log_error(f"Không tìm thấy file processed: {processed_path}. Hãy chạy TV3 trước!")
                return self

            self.step_clustering()

            total_time = time.time() - total_start
            log_success(f"Phần TV4 hoàn tất trong {total_time:.1f}s")
        except Exception as e:
            log_error(f"Phần TV4 thất bại: {e}")
            raise

        return self

    def run_tv5(self):
        """
        Chạy phần của Thành viên 5: Deep Learning & Tuning.
        Huấn luyện mô hình phân loại.
        """
        total_start = time.time()
        log_info("🔹 CHẠY PHẦN THÀNH VIÊN 5: Deep Learning & Model Training")
        log_info("=" * 60)

        try:
            self.step_train_model()

            total_time = time.time() - total_start
            log_success(f"Phần TV5 hoàn tất trong {total_time:.1f}s")
        except Exception as e:
            log_error(f"Phần TV5 thất bại: {e}")
            raise

        return self

    def run_tv6(self):
        """
        Chạy phần của Thành viên 6 (Khánh Huyền): Đánh giá & Xuất dữ liệu.
        Xuất kết quả cuối cùng và báo cáo.
        """
        total_start = time.time()
        log_info("🔹 CHẠY PHẦN THÀNH VIÊN 6: Export & Reporting")
        log_info("=" * 60)

        try:
            self.step_export()

            total_time = time.time() - total_start
            log_success(f"Phần TV6 hoàn tất trong {total_time:.1f}s")
        except Exception as e:
            log_error(f"Phần TV6 thất bại: {e}")
            raise

        return self


# ===================================================================
# CHẠY TRỰC TIẾP
# ===================================================================
def print_usage():
    """In hướng dẫn sử dụng."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        PIPELINE XỬ LÝ DỮ LIỆU TIN TỨC CÔNG NGHỆ          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Cách dùng: python pipeline.py [option]                      ║
║                                                              ║
║  Các option:                                                 ║
║    all   - Chạy toàn bộ pipeline (mặc định)                  ║
║    tv1   - Thành viên 1: Thu thập dữ liệu & EDA             ║
║    tv2   - Thành viên 2: Làm sạch, Chuẩn hóa, Loại trùng   ║
║    tv3   - Thành viên 3: Tiền xử lý NLP & TF-IDF            ║
║    tv4   - Thành viên 4: Phân cụm & Gán nhãn                ║
║    tv5   - Thành viên 5: Deep Learning & Tuning              ║
║    tv6   - Thành viên 6: Xuất dữ liệu & Báo cáo            ║
║                                                              ║
║  Ví dụ:                                                      ║
║    python pipeline.py all    → Chạy tất cả                   ║
║    python pipeline.py tv2    → Chỉ chạy phần TV2             ║
║    python pipeline.py tv3    → Chỉ chạy phần TV3             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    # Nếu có argument dòng lệnh → chạy trực tiếp
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(
            description='Pipeline Xử lý Dữ liệu Tin tức Công nghệ'
        )
        parser.add_argument(
            'mode',
            choices=['all', 'tv1', 'tv2', 'tv3', 'tv4', 'tv5', 'tv6'],
            help='Chế độ chạy: all (toàn bộ) hoặc tv1-tv6 (theo thành viên)'
        )
        args = parser.parse_args()

        pipeline = DataPipeline()
        mode_map = {
            'all': pipeline.run_all,
            'tv1': pipeline.run_tv1,
            'tv2': pipeline.run_tv2,
            'tv3': pipeline.run_tv3,
            'tv4': pipeline.run_tv4,
            'tv5': pipeline.run_tv5,
            'tv6': pipeline.run_tv6,
        }
        mode_map[args.mode]()

    else:
        # Không có argument → hiện menu tương tác
        print("""
╔══════════════════════════════════════════════════════════════╗
║        PIPELINE XỬ LÝ DỮ LIỆU TIN TỨC CÔNG NGHỆ          ║
║                     Nhóm 6 - Data Mining                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [0]  Chạy toàn bộ pipeline (all)                            ║
║                                                              ║
║  [1]  TV1 - Ngô Hoàng Anh:    Thu thập dữ liệu & EDA       ║
║  [2]  TV2 - Lê Quang Thi:     Làm sạch, Chuẩn hóa, Dedup   ║
║  [3]  TV3 - Tôn Hoàng Nhớ:    Tiền xử lý NLP & TF-IDF      ║
║  [4]  TV4 - Nguyễn Văn Trường: Phân cụm & Gán nhãn          ║
║  [5]  TV5 - (Thành viên 5):   Deep Learning & Tuning         ║
║  [6]  TV6 - Khánh Huyền:      Xuất dữ liệu & Báo cáo       ║
║                                                              ║
║  [q]  Thoát                                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

        choice = input("👉 Chọn số (0-6) hoặc 'q' để thoát: ").strip()

        if choice == 'q':
            print("Đã thoát.")
            sys.exit(0)

        pipeline = DataPipeline()
        choice_map = {
            '0': pipeline.run_all,
            '1': pipeline.run_tv1,
            '2': pipeline.run_tv2,
            '3': pipeline.run_tv3,
            '4': pipeline.run_tv4,
            '5': pipeline.run_tv5,
            '6': pipeline.run_tv6,
        }

        if choice in choice_map:
            choice_map[choice]()
        else:
            print(f"❌ Lựa chọn '{choice}' không hợp lệ. Vui lòng chọn 0-6 hoặc 'q'.")
            sys.exit(1)

