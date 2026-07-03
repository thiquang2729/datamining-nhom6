"""
main.py - Scrape bài viết Công nghệ từ 3 nguồn: VnExpress, VietnamNet, Thanh Niên.

Đọc danh sách URL từ file (mặc định: data/tech_urls.txt), scrape nội dung,
lưu vào file data/news_data.csv liên tục (chạy đến đâu lưu đến đó) và sử dụng đa luồng.
"""
import sys
import random
import threading
import csv
import os
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from scrapers import (
    VnExpressScraper,
    VietNamNetScraper,
    ThanhNienScraper,
    BaseScraper,
)
from models import Article

# Ghi đè mã hóa console cho Windows để in tiếng Việt không bị lỗi
sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------
# Mapping domain → scraper instance (chỉ 3 nguồn Công nghệ)
# ---------------------------------------------------------------
SCRAPER_MAP: Dict[str, BaseScraper] = {
    "vnexpress.net": VnExpressScraper(),
    "vietnamnet.vn": VietNamNetScraper(),
    "thanhnien.vn":  ThanhNienScraper(),
}

# Thread-safe counters
_lock = threading.Lock()
ok_count = 0
fail_count = 0

# Đường dẫn tuyệt đối đến thư mục data/ (nằm ngoài src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
CSV_FILE = str(_DATA_DIR / "news_data.csv")

def load_urls_from_file(file_path: Path) -> List[str]:
    """Read a plain‑text file (one URL per line) and return a list."""
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_domain(url: str) -> str:
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc
    return netloc.replace("www.", "")


def init_csv():
    """Khởi tạo file CSV và ghi header nếu chưa có."""
    file_exists = os.path.exists(CSV_FILE)
    # Nếu file chưa tồn tại hoặc user muốn ghi đè từ đầu, mở mode "w". 
    # Ở đây dùng mode "w" để xoá data cũ, tạo data mới tinh.
    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        # Lấy fieldnames từ mô hình Article
        fieldnames = list(Article.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def process_url(url: str, total_urls: int) -> None:
    global ok_count, fail_count

    domain = get_domain(url)
    scraper = SCRAPER_MAP.get(domain)

    if not scraper:
        with _lock:
            fail_count += 1
            print(f"[{ok_count + fail_count}/{total_urls}] [WARN] Không có scraper cho domain '{domain}': {url}")
        return

    article = scraper.scrape(url)

    with _lock:
        if article:
            # Ép category = "Công nghệ" cho tất cả bài viết
            article.category = "Công nghệ"
            
            # Ghi trực tiếp vào CSV (lưu dần)
            with open(CSV_FILE, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(Article.__dataclass_fields__.keys()))
                writer.writerow(asdict(article))
                
            ok_count += 1
            title_short = (article.title[:60] + '...') if article.title and len(article.title) > 60 else article.title
            print(f"[{ok_count + fail_count}/{total_urls}] [OK] {domain} | {title_short}")
        else:
            fail_count += 1
            print(f"[{ok_count + fail_count}/{total_urls}] [FAIL] Không trích xuất được: {url}")


def main(url_file: str = None):
    global ok_count, fail_count

    if url_file is None:
        path = _DATA_DIR / "tech_urls.txt"
    else:
        path = Path(url_file)
    if not path.is_file():
        print(f"Không tìm thấy file '{path}'.")
        print("Hãy chạy 'python collect_tech.py' trước để thu thập URL.")
        sys.exit(1)

    urls = load_urls_from_file(path)

    if not urls:
        print("Không có URL nào để xử lý. Thoát.")
        sys.exit(0)

    # Shuffle để phân tán tải đều cho các domain (tránh bị block)
    random.shuffle(urls)

    total = len(urls)
    print(f"\n{'='*60}")
    print(f"  SCRAPE BÀI VIẾT CÔNG NGHỆ VÀ LƯU DẦN VÀO CSV")
    print(f"  Tổng số URL: {total:,}")
    print(f"  Chế độ     : Đa luồng (15 threads) - Ghi CSV Realtime")
    print(f"{'='*60}\n")

    # Khởi tạo CSV (tạo mới và ghi header)
    init_csv()

    # Chạy đa luồng
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(process_url, url, total) for url in urls]
        for _ in as_completed(futures):
            pass

    print(f"\n{'='*60}")
    print(f"  ✅ Thành công : {ok_count:,} / {total:,}")
    print(f"  ❌ Thất bại   : {fail_count:,} / {total:,}")
    print(f"  Tỉ lệ OK      : {ok_count/total*100:.1f}%")
    print(f"  Dữ liệu lưu   : {CSV_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
