"""
collect_tech.py - Thu thập URL bài viết Công nghệ từ 3 nguồn báo.

Nguồn: VnExpress, VietnamNet, Thanh Niên
Mục tiêu: ~20,000 URL, cân bằng giữa 3 domain.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import time
import logging
from typing import Set
from pathlib import Path
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
OUTPUT_FILE = str(_DATA_DIR / "tech_urls.txt")
TARGET_TOTAL = 20_000
TARGET_PER_DOMAIN = 7_200  # Thu thập dư một chút để cân bằng ở mức ~6,667

UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
]

def _get(url, timeout=15):
    return requests.get(url, headers={"User-Agent": random.choice(UA)}, timeout=timeout)

def crawl_pages(base_fmt, domain_prefix, suffix, max_pages, existing):
    """Crawl paginated category pages and return article URLs."""
    urls = set()
    empty = 0
    for page in range(1, max_pages + 1):
        if len(urls) + len(existing) >= TARGET_PER_DOMAIN:
            break
        page_url = base_fmt.format(page=page)
        try:
            r = _get(page_url)
            if r.status_code == 404:
                break
            if r.status_code != 200:
                logger.warning(f"  [{domain_prefix}] Trang {page} trả về status {r.status_code}, thử qua trang tiếp theo...")
                empty += 1
                if empty >= 15:
                    break
                time.sleep(0.5)
                continue

            soup = BeautifulSoup(r.text, 'html.parser')
            n = 0
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/') and not href.startswith('//'):
                    href = domain_prefix + href
                if (href.startswith(domain_prefix)
                    and href.endswith(suffix)
                    and "-" in href.split("/")[-1]
                    and href not in urls
                    and href not in existing):
                    urls.add(href)
                    n += 1
            if n == 0:
                empty += 1
                if empty >= 15:
                    break
            else:
                empty = 0
            
            if page % 100 == 0:
                logger.info(f"  p{page}: +{len(urls):,} URLs")
            time.sleep(0.08)
        except Exception as e:
            logger.warning(f"  Lỗi khi crawl trang {page}: {e}")
            time.sleep(0.5)
    return urls

# ---------------------------------------------------------------------------
def collect_vnexpress(target):
    urls = set()
    # RSS
    for rss in ["https://vnexpress.net/rss/so-hoa.rss", "https://vnexpress.net/rss/khoa-hoc.rss"]:
        try:
            r = _get(rss)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, 'xml')
                for item in soup.find_all('item'):
                    link = item.find('link')
                    if link and link.text:
                        u = link.text.strip()
                        if u.startswith("https://vnexpress.net/") and u.endswith(".html"):
                            urls.add(u)
        except Exception:
            pass
    logger.info(f"  [RSS] → {len(urls)} URLs")

    # Crawl so-hoa + khoa-hoc category pages
    for cat in ["so-hoa", "khoa-hoc"]:
        logger.info(f"  [CRAWL] {cat}...")
        cat_urls = crawl_pages(
            f"https://vnexpress.net/{cat}-p{{page}}",
            "https://vnexpress.net", ".html", 50, urls
        )
        urls.update(cat_urls)
        logger.info(f"  [CRAWL] {cat} → +{len(cat_urls):,} (tổng: {len(urls):,})")

    # Crawl sub-categories
    subcats = ["so-hoa/cong-nghe", "so-hoa/san-pham", "khoa-hoc/cong-nghe-moi", "khoa-hoc/vu-tru", "khoa-hoc/tin-tuc"]
    for sc in subcats:
        logger.info(f"  [CRAWL SUB] {sc}...")
        sc_urls = crawl_pages(
            f"https://vnexpress.net/{sc}-p{{page}}",
            "https://vnexpress.net", ".html", 25, urls
        )
        urls.update(sc_urls)
        logger.info(f"  [CRAWL SUB] {sc} → +{len(sc_urls):,} (tổng: {len(urls):,})")

    # Extensive search query keyword list
    queries = [
        "công nghệ", "AI", "trí tuệ nhân tạo", "smartphone", "điện thoại", 
        "máy tính", "bảo mật", "phần mềm", "game", "chip", 
        "vi xử lý", "robot", "mạng xã hội", "viễn thông", "khoa học",
        "vũ trụ", "thiết bị", "bán dẫn", "mạng di động", "chuyển đổi số",
        "samsung", "iphone", "chatgpt", "openai", "google", "apple",
        "xiaomi", "oppo", "microsoft", "intel", "amd", "nvidia", "meta",
        "máy tính bảng", "đồng hồ thông minh", "tai nghe", "hacker", "an ninh mạng",
        "dữ liệu", "vệ tinh", "tên lửa", "elon musk", "spacex", "xe điện",
        "cáp quang", "wifi", "router", "sim", "đường truyền", "tốc độ",
        "mã độc", "virus", "lừa đảo", "quyền riêng tư", "mật khẩu", "xác thực",
        "nhận diện", "thuật toán", "tự động hóa", "vr", "ar", "blockchain",
        "tiền số", "bitcoin", "ethereum", "web3", "nft", "metaverse", "5g", "6g",
        "doanh thu", "cổ phiếu", "sa thải", "tuyển dụng", "nhân sự", "ceo",
        "viettel", "vinaphone", "mobifone", "fpt", "vnpt", "phát minh",
        "sáng chế", "bằng sáng chế", "sở hữu trí tuệ", "độc quyền", "luật an ninh mạng"
    ]
    logger.info(f"  [SEARCH] Bắt đầu tìm kiếm với {len(queries)} từ khóa...")
    for q in queries:
        if len(urls) >= target:
            break
        empty = 0
        q_added = 0
        for page in range(1, 101):
            if len(urls) >= target:
                break
            search_url = f"https://timkiem.vnexpress.net/?q={q}&cate_code=khoa-hoc-cong-nghe&page={page}"
            try:
                r = _get(search_url)
                if r.status_code == 404:
                    break
                if r.status_code != 200:
                    empty += 1
                    if empty >= 10:
                        break
                    time.sleep(0.5)
                    continue

                soup = BeautifulSoup(r.text, 'html.parser')
                n = 0
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if (href.startswith("https://vnexpress.net/")
                        and href.endswith(".html")
                        and "-" in href.split("/")[-1]
                        and href not in urls):
                        urls.add(href)
                        n += 1
                        q_added += 1
                if n == 0:
                    empty += 1
                    if empty >= 10:
                        break
                else:
                    empty = 0
                time.sleep(0.08)
            except Exception:
                time.sleep(0.3)
        if q_added > 0:
            logger.info(f"    Từ khóa '{q}' → thêm được +{q_added:,} (tổng: {len(urls):,})")

    logger.info(f"  [DONE] vnexpress.net → {len(urls):,} URLs")
    return urls

# ---------------------------------------------------------------------------
def collect_vietnamnet(target):
    urls = set()
    # RSS
    try:
        r = _get("https://vietnamnet.vn/rss/cong-nghe.rss")
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'xml')
            for item in soup.find_all('item'):
                link = item.find('link')
                if link and link.text:
                    u = link.text.strip()
                    if u.startswith("https://vietnamnet.vn/") and u.endswith(".html"):
                        urls.add(u)
    except Exception:
        pass
    logger.info(f"  [RSS] → {len(urls)} URLs")

    # Crawl cong-nghe
    logger.info(f"  [CRAWL] cong-nghe...")
    cat_urls = crawl_pages(
        "https://vietnamnet.vn/cong-nghe-page{page}",
        "https://vietnamnet.vn", ".html", 800, urls
    )
    urls.update(cat_urls)
    logger.info(f"  [DONE] vietnamnet.vn → {len(urls):,} URLs")
    return urls

# ---------------------------------------------------------------------------
def collect_thanhnien(target):
    urls = set()
    # RSS
    rss_feeds = [
        "https://thanhnien.vn/rss/cong-nghe.rss",
        "https://thanhnien.vn/rss/cong-nghe/tin-tuc-cong-nghe.rss",
        "https://thanhnien.vn/rss/cong-nghe/san-pham.rss",
        "https://thanhnien.vn/rss/cong-nghe/game.rss",
        "https://thanhnien.vn/rss/cong-nghe/thu-thuat.rss",
        "https://thanhnien.vn/rss/cong-nghe/blockchain.rss",
        "https://thanhnien.vn/rss/cong-nghe/xu-huong-chuyen-doi-so.rss",
    ]
    for rss in rss_feeds:
        try:
            r = _get(rss)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, 'xml')
                for item in soup.find_all('item'):
                    link = item.find('link')
                    if link and link.text:
                        u = link.text.strip()
                        if u.startswith("https://thanhnien.vn/") and u.endswith(".htm"):
                            urls.add(u)
        except Exception:
            pass
    logger.info(f"  [RSS] → {len(urls)} URLs")

    # Crawl timelinelist cho zoneId=185315 (Công nghệ)
    logger.info(f"  [CRAWL] timelinelist (zone 185315)...")
    cat_urls = crawl_pages(
        "https://thanhnien.vn/timelinelist/185315/{page}.htm",
        "https://thanhnien.vn", ".htm", 500, urls
    )
    urls.update(cat_urls)
    logger.info(f"  [DONE] thanhnien.vn → {len(urls):,} URLs")
    return urls

# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("  THU THẬP URL CÔNG NGHỆ TỪ 3 NGUỒN BÁO (BẢN SIÊU RỘNG)")
    logger.info(f"  Mục tiêu: ~{TARGET_TOTAL:,} URLs (~6,667/domain)")
    logger.info("=" * 60)

    results = {}
    for name, fn in [("vnexpress.net", collect_vnexpress),
                      ("vietnamnet.vn", collect_vietnamnet),
                      ("thanhnien.vn",  collect_thanhnien)]:
        logger.info(f"\n--- {name} ---")
        results[name] = fn(TARGET_PER_DOMAIN)

    # Cân bằng
    counts = {d: len(u) for d, u in results.items()}
    min_count = min(counts.values())
    per_domain = min(min_count, 6667)

    logger.info(f"\n{'='*60}")
    logger.info(f"  KẾT QUẢ THU THẬP VÀ CÂN BẰNG DỮ LIỆU")
    logger.info(f"{'='*60}")
    for d, c in counts.items():
        logger.info(f"  {d}: {c:,} URLs thu thập được")
    logger.info(f"  Lấy cân bằng: {per_domain:,} URLs/domain")

    final = []
    for domain, all_urls in results.items():
        sampled = random.sample(sorted(all_urls), min(per_domain, len(all_urls)))
        final.extend(sampled)
    random.shuffle(final)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for u in final:
            f.write(u + "\n")

    logger.info(f"\n  Đã lưu {len(final):,} URLs → {OUTPUT_FILE}")
    for d in ["vnexpress.net", "vietnamnet.vn", "thanhnien.vn"]:
        c = sum(1 for u in final if d in u)
        logger.info(f"    {d}: {c:,}")

if __name__ == "__main__":
    main()
