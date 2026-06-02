"""
Module Logger có màu sắc cho Pipeline xử lý dữ liệu.
Sử dụng colorama để hiển thị log có màu sắc trên terminal.

Màu sắc:
    - 🟢 SUCCESS (xanh lá): Bước hoàn thành thành công
    - 🔵 INFO (xanh dương): Thông tin chung
    - 🟡 WARNING (vàng): Cảnh báo
    - 🔴 ERROR (đỏ): Lỗi nghiêm trọng
    - 🟣 STEP (tím/magenta): Bắt đầu/kết thúc một bước trong pipeline
"""

import sys
import time
from colorama import init, Fore, Style

# Fix encoding cho Windows console (tránh lỗi UnicodeEncodeError với tiếng Việt)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Khởi tạo colorama cho Windows
init(autoreset=True)


def log_info(message):
    """Log thông tin chung (màu xanh dương)."""
    print(f"{Fore.CYAN}[INFO] {message}{Style.RESET_ALL}")


def log_success(message):
    """Log thành công (màu xanh lá)."""
    print(f"{Fore.GREEN}[SUCCESS] ✔ {message}{Style.RESET_ALL}")


def log_warning(message):
    """Log cảnh báo (màu vàng)."""
    print(f"{Fore.YELLOW}[WARNING] ⚠ {message}{Style.RESET_ALL}")


def log_error(message):
    """Log lỗi (màu đỏ)."""
    print(f"{Fore.RED}[ERROR] ✘ {message}{Style.RESET_ALL}")


def log_step_start(step_name, record_count=None):
    """
    Log bắt đầu một bước trong pipeline (màu tím).

    Args:
        step_name (str): Tên bước đang thực hiện.
        record_count (int, optional): Số lượng bản ghi đầu vào.
    """
    count_str = f" | Đầu vào: {record_count:,} bản ghi" if record_count else ""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}[STEP] ▶ {step_name}{count_str}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")


def log_step_end(step_name, before_count, after_count, elapsed_time):
    """
    Log kết thúc một bước trong pipeline (màu xanh lá) kèm thống kê.

    Args:
        step_name (str): Tên bước đã hoàn thành.
        before_count (int): Số bản ghi trước khi xử lý.
        after_count (int): Số bản ghi sau khi xử lý.
        elapsed_time (float): Thời gian thực hiện (giây).
    """
    removed = before_count - after_count
    pct = (removed / before_count * 100) if before_count > 0 else 0
    print(f"\n{Fore.GREEN}[DONE] ✔ {step_name} hoàn thành!")
    print(f"{Fore.GREEN}  ├─ Trước: {before_count:,} bản ghi")
    print(f"{Fore.GREEN}  ├─ Sau:   {after_count:,} bản ghi")
    print(f"{Fore.GREEN}  ├─ Đã loại bỏ: {removed:,} ({pct:.1f}%)")
    print(f"{Fore.GREEN}  └─ Thời gian: {elapsed_time:.2f}s")
    print(f"{Fore.GREEN}{'-'*60}{Style.RESET_ALL}")


def log_summary(total_before, total_after, total_time):
    """
    Log tổng kết toàn bộ pipeline.

    Args:
        total_before (int): Tổng bản ghi ban đầu.
        total_after (int): Tổng bản ghi còn lại.
        total_time (float): Tổng thời gian chạy pipeline (giây).
    """
    removed = total_before - total_after
    pct = (removed / total_before * 100) if total_before > 0 else 0
    print(f"\n{Fore.CYAN}{'#'*60}")
    print(f"{Fore.CYAN}  TỔNG KẾT PIPELINE")
    print(f"{Fore.CYAN}{'#'*60}")
    print(f"{Fore.CYAN}  ├─ Tổng bản ghi ban đầu : {total_before:,}")
    print(f"{Fore.CYAN}  ├─ Tổng bản ghi còn lại : {total_after:,}")
    print(f"{Fore.CYAN}  ├─ Tổng đã loại bỏ      : {removed:,} ({pct:.1f}%)")
    print(f"{Fore.CYAN}  └─ Tổng thời gian       : {total_time:.2f}s")
    print(f"{Fore.CYAN}{'#'*60}{Style.RESET_ALL}")


class StepTimer:
    """Context manager để đo thời gian cho mỗi bước pipeline."""

    def __init__(self, step_name, record_count=None):
        self.step_name = step_name
        self.record_count = record_count
        self.start_time = None

    def __enter__(self):
        log_step_start(self.step_name, self.record_count)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            log_error(f"{self.step_name} thất bại sau {elapsed:.2f}s: {exc_val}")
        return False  # Không chặn exception

    @property
    def elapsed(self):
        """Trả về thời gian đã trôi qua (giây)."""
        if self.start_time:
            return time.time() - self.start_time
        return 0
