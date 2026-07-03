"""Base scraper and site‑specific implementations.

Only a minimal set of selectors is used; they can be refined after
inspecting the actual page markup.
"""

import json
import re
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from models import Article

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

def random_user_agent() -> str:
    return random.choice(USER_AGENTS)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseScraper(ABC):
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch_html(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page with random UA and a tiny random delay.
        Returns ``BeautifulSoup`` or ``None`` on failure.
        """
        time.sleep(random.uniform(0.1, 0.4))
        headers = {"User-Agent": random_user_agent()}
        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[BaseScraper] fetch error {url}: {e}")
            return None

    @abstractmethod
    def parse_article(self, soup: BeautifulSoup, url: str) -> Article:
        """Extract 22 fields from the page. Must be overridden."""
        pass

    def scrape(self, url: str) -> Optional[Article]:
        soup = self.fetch_html(url)
        if not soup:
            return None
        try:
            article = self.parse_article(soup, url)
            article.scraped_time = datetime.utcnow().isoformat()
            return article
        except Exception as e:
            print(f"[BaseScraper] parse error {url}: {e}")
            return None

# ---------------------------------------------------------------------------
# Site‑specific scrapers – simple, robust fallbacks are used.
# ---------------------------------------------------------------------------
class VnExpressScraper(BaseScraper):
    def parse_article(self, soup: BeautifulSoup, url: str) -> Article:
        fname = url.split('/')[-1]
        m = re.search(r'-(\d+)\.(html|htm)$', fname)
        article_id = m.group(1) if m else fname
        domain = "vnexpress.net"
        # Title
        title_tag = soup.select_one("h1.title-detail, h1.title_news_detail")
        title = title_tag.get_text(strip=True) if title_tag else None
        # Description / summary
        desc_tag = soup.select_one("p.description, meta[name='description']")
        description = desc_tag.get_text(strip=True) if desc_tag else None
        # Main content – collect paragraphs inside article body
        paragraphs = soup.select("article .detail-content p, .fck_detail p")
        main_content = " ".join(p.get_text(strip=True) for p in paragraphs) if paragraphs else None
        # Author name (if present)
        author_tag = soup.select_one("span.author, a.author_name")
        author_name = author_tag.get_text(strip=True) if author_tag else None
        # Tags – collect meta keywords or visible tag list
        tags_tag = soup.select_one("meta[name='keywords']")
        tags = tags_tag["content"].strip() if tags_tag and tags_tag.get("content") else None
        # Images & videos
        images = [img["src"] for img in soup.select("article img, .detail-content img") if img.get("src")]
        videos = [v["src"] for v in soup.select("article video source, .detail-content video source") if v.get("src")]
        # Published time – try common meta property
        pub_tag = soup.select_one("meta[property='article:published_time'], time[datetime]")
        published_time = pub_tag["content"] if pub_tag and pub_tag.get("content") else None
        # Category – attempt to infer from breadcrumb or URL
        category = None
        sub_category = None
        
        # Meta section is very clean
        meta_sec = soup.find("meta", {"name": "its_section"})
        meta_subsec = soup.find("meta", {"name": "its_subsection"})
        
        if meta_sec and meta_sec.get("content"):
            category = meta_sec["content"].strip().capitalize()
        if meta_subsec and meta_subsec.get("content"):
            # its_subsection is comma-separated, e.g. "giải trí, phim, chuyện màn ảnh"
            subs = [s.strip().capitalize() for s in meta_subsec["content"].split(",") if s.strip()]
            if len(subs) > 1:
                sub_category = subs[1]
                
        # Fallback to breadcrumb
        if not category:
            breadcrumb = soup.select_one("ul.breadcrumb a, nav.breadcrumb a, div.breadcrumb a")
            if breadcrumb:
                category = breadcrumb.get_text(strip=True)
                
        if not sub_category:
            crumbs = soup.select("ul.breadcrumb a, nav.breadcrumb a, div.breadcrumb a")
            if len(crumbs) > 1:
                sub_category = crumbs[1].get_text(strip=True)
                
        # Views / comments – not always available, keep None
        views_count = None
        comments_count = None
        # Media URLs – thumbnail (first image)
        thumbnail_url = images[0] if images else None
        return Article(
            article_id=article_id,
            url=url,
            domain=domain,
            category_url=None,
            author_url=None,
            amp_url=None,
            reference_url=None,
            short_url=None,
            title=title,
            description=description,
            main_content=main_content,
            author_name=author_name,
            tags=tags,
            thumbnail_url=thumbnail_url,
            images=json.dumps(images) if images else None,
            videos=json.dumps(videos) if videos else None,
            published_time=published_time,
            scraped_time=datetime.utcnow().isoformat(),
            category=category,
            sub_category=sub_category,
            views_count=views_count,
            comments_count=comments_count,
        )


class TuoiTreScraper(BaseScraper):
    def parse_article(self, soup: BeautifulSoup, url: str) -> Article:
        fname = url.split('/')[-1]
        m = re.search(r'-(\d+)\.(html|htm)$', fname)
        article_id = m.group(1) if m else fname
        domain = "tuoitre.vn"
        
        # Fixed Selector for Tuổi Trẻ Title
        title_tag = soup.select_one("h1.detail-title, h1.article-title, h1.title, h1.title-detail")
        title = title_tag.get_text(strip=True) if title_tag else None
        
        description = None  # Tuổi Trẻ rarely provides a meta description
        
        # Fixed Selector for Tuổi Trẻ Content
        paragraphs = soup.select(".detail-content p, .content-detail p, div.detail-cmain p, article .content-detail p, .article-body p")
        main_content = " ".join(p.get_text(strip=True) for p in paragraphs) if paragraphs else None
        
        # Author
        author_tag = soup.select_one("a.author-name, span.author")
        author_name = author_tag.get_text(strip=True) if author_tag else None
        
        # Tags – meta keywords
        tags_tag = soup.select_one("meta[name='keywords']")
        tags = tags_tag["content"].strip() if tags_tag and tags_tag.get("content") else None
        images = [img["src"] for img in soup.select("article img, .content-detail img, .detail-content img") if img.get("src")]
        videos = []  # Not common on Tuổi Trẻ articles
        thumbnail_url = images[0] if images else None
        
        # Published time
        pub_tag = soup.select_one("meta[property='article:published_time'], time")
        published_time = pub_tag["content"] if pub_tag and pub_tag.get("content") else None
        
        # Category inference from URL path
        parts = url.replace("https://tuoitre.vn/", "").split('/')
        category = parts[0] if parts else None
        sub_category = parts[1] if len(parts) > 1 else None
        
        return Article(
            article_id=article_id,
            url=url,
            domain=domain,
            category_url=None,
            author_url=None,
            amp_url=None,
            reference_url=None,
            short_url=None,
            title=title,
            description=description,
            main_content=main_content,
            author_name=author_name,
            tags=tags,
            thumbnail_url=thumbnail_url,
            images=json.dumps(images) if images else None,
            videos=json.dumps(videos) if videos else None,
            published_time=published_time,
            scraped_time=datetime.utcnow().isoformat(),
            category=category,
            sub_category=sub_category,
            views_count=None,
            comments_count=None,
        )

class VietNamNetScraper(BaseScraper):
    def parse_article(self, soup: BeautifulSoup, url: str) -> Article:
        fname = url.split('/')[-1]
        m = re.search(r'-(\d+)\.(html|htm)$', fname)
        article_id = m.group(1) if m else fname
        domain = "vietnamnet.vn"
        
        # Fixed Selector for VietNamNet Title
        title_tag = soup.select_one("h1.content-detail-title, h1.title, h1.title-news")
        title = title_tag.get_text(strip=True) if title_tag else None
        
        description_tag = soup.select_one("meta[name='description']")
        description = description_tag["content"].strip() if description_tag and description_tag.get("content") else None
        
        # Fixed Selector for VietNamNet Content
        paragraphs = soup.select(".main-content p, .maincontent p, article p, .article-body p")
        main_content = " ".join(p.get_text(strip=True) for p in paragraphs) if paragraphs else None
        author_tag = soup.select_one("span.author, a.author")
        author_name = author_tag.get_text(strip=True) if author_tag else None
        tags_tag = soup.select_one("meta[name='keywords']")
        tags = tags_tag["content"].strip() if tags_tag and tags_tag.get("content") else None
        images = [img["src"] for img in soup.select("article img, .article-body img") if img.get("src")]
        videos = []
        thumbnail_url = images[0] if images else None
        pub_tag = soup.select_one("meta[property='article:published_time'], time[datetime]")
        published_time = pub_tag["content"] if pub_tag and pub_tag.get("content") else None
        
        # Category from JSON-LD or breadcrumb fallback
        category = None
        sub_category = None
        
        # Try JSON-LD BreadcrumbList
        for s in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(s.string)
                if data.get('@type') == 'BreadcrumbList' and 'itemListElement' in data:
                    items = [item['name'] for item in data['itemListElement']]
                    # Format: ['Trang chủ', 'Thời sự', 'Dân sinh']
                    # We pick index 1 as Category, index 2 as Subcategory
                    val_items = [i for i in items if i != 'Trang chủ']
                    if val_items:
                        category = val_items[0].strip()
                        if len(val_items) > 1:
                            sub_category = val_items[1].strip()
                    break
            except Exception:
                pass
                
        if not category:
            crumbs = soup.select("nav.breadcrumb a, div.breadcrumb a, ul.breadcrumb a")
            if crumbs:
                category = crumbs[0].get_text(strip=True)
            if len(crumbs) > 1:
                sub_category = crumbs[1].get_text(strip=True)
                
        return Article(
            article_id=article_id,
            url=url,
            domain=domain,
            category_url=None,
            author_url=None,
            amp_url=None,
            reference_url=None,
            short_url=None,
            title=title,
            description=description,
            main_content=main_content,
            author_name=author_name,
            tags=tags,
            thumbnail_url=thumbnail_url,
            images=json.dumps(images) if images else None,
            videos=json.dumps(videos) if videos else None,
            published_time=published_time,
            scraped_time=datetime.utcnow().isoformat(),
            category=category,
            sub_category=sub_category,
            views_count=None,
            comments_count=None,
        )

class ThanhNienScraper(BaseScraper):
    def parse_article(self, soup: BeautifulSoup, url: str) -> Article:
        # Thanh Nien article ID is the trailing number before .htm, e.g. -1851105141.htm
        fname = url.split('/')[-1]
        m = re.search(r'-(\d+)\.(html|htm)$', fname)
        article_id = m.group(1) if m else fname
        domain = "thanhnien.vn"
        # Title: actual class is 'detail-title'
        title_tag = soup.select_one("h1.detail-title, h1.title, h1.entry-title")
        title = title_tag.get_text(strip=True) if title_tag else None
        description_tag = soup.select_one("meta[name='description']")
        description = description_tag["content"].strip() if description_tag and description_tag.get("content") else None
        # Content: Thanh Nien wraps body text in div.detail-content (not plain <p> tags)
        content_div = soup.select_one("div.detail-content, div.detail-cmain")
        if content_div:
            # Remove nested 'related article' boxes to avoid noise
            for tag in content_div.select("div.detail-related, div.box-tinlienquan"):
                tag.decompose()
            main_content = content_div.get_text(separator=" ", strip=True) or None
        else:
            main_content = None
        # Author: div with 'author' in class name
        author_tag = soup.select_one("div[class*='author'] a, div[class*='author'] span, span.author-name")
        author_name = author_tag.get_text(strip=True) if author_tag else None
        tags_tag = soup.select_one("meta[name='keywords']")
        tags = tags_tag["content"].strip() if tags_tag and tags_tag.get("content") else None
        images = [img["src"] for img in soup.select("div.detail-content img, div.detail-cmain img") if img.get("src")]
        videos = []
        thumbnail_url = images[0] if images else None
        pub_tag = soup.select_one("meta[property='article:published_time']")
        published_time = pub_tag["content"] if pub_tag and pub_tag.get("content") else None
        
        # Category: use meta article:section or category tags
        category = None
        sub_category = None
        
        meta_sec = soup.find("meta", {"property": "article:section"})
        if meta_sec and meta_sec.get("content"):
            category = meta_sec["content"].strip()
            
        if not category:
            category_tag = soup.select_one("div.detail__category a, nav.breadcrumb a, a.detail__cate, div.detail-cate a, div.detail-menu ul li a")
            category = category_tag.get_text(strip=True) if category_tag else None
            
        crumbs = soup.select("nav.breadcrumb a, div.detail-menu ul li a")
        if len(crumbs) > 1:
            sub_category = crumbs[1].get_text(strip=True)
            
        return Article(
            article_id=article_id,
            url=url,
            domain=domain,
            category_url=None,
            author_url=None,
            amp_url=None,
            reference_url=None,
            short_url=None,
            title=title,
            description=description,
            main_content=main_content,
            author_name=author_name,
            tags=tags,
            thumbnail_url=thumbnail_url,
            images=json.dumps(images) if images else None,
            videos=json.dumps(videos) if videos else None,
            published_time=published_time,
            scraped_time=datetime.utcnow().isoformat(),
            category=category,
            sub_category=sub_category,
            views_count=None,
            comments_count=None,
        )

class VTCNewsScraper(BaseScraper):
    def parse_article(self, soup: BeautifulSoup, url: str) -> Article:
        fname = url.split('/')[-1]
        m = re.search(r'(\d+)\.(html|htm)$', fname)
        article_id = m.group(1) if m else fname
        domain = "vtcnews.vn"
        title_tag = soup.select_one("h1.title, h1.article-title")
        title = title_tag.get_text(strip=True) if title_tag else None
        description_tag = soup.select_one("meta[name='description']")
        description = description_tag["content"].strip() if description_tag and description_tag.get("content") else None
        paragraphs = soup.select("article p, .article-content p")
        main_content = " ".join(p.get_text(strip=True) for p in paragraphs) if paragraphs else None
        author_tag = soup.select_one("span.author, a.author")
        author_name = author_tag.get_text(strip=True) if author_tag else None
        tags_tag = soup.select_one("meta[name='keywords']")
        tags = tags_tag["content"].strip() if tags_tag and tags_tag.get("content") else None
        images = [img["src"] for img in soup.select("article img, .article-content img") if img.get("src")]
        videos = []
        thumbnail_url = images[0] if images else None
        pub_tag = soup.select_one("meta[property='article:published_time'], time")
        published_time = pub_tag["content"] if pub_tag and pub_tag.get("content") else None
        # Category inference from URL
        parts = url.replace("https://vtcnews.vn/", "").split('/')
        category = parts[0] if parts else None
        sub_category = parts[1] if len(parts) > 1 else None
        return Article(
            article_id=article_id,
            url=url,
            domain=domain,
            category_url=None,
            author_url=None,
            amp_url=None,
            reference_url=None,
            short_url=None,
            title=title,
            description=description,
            main_content=main_content,
            author_name=author_name,
            tags=tags,
            thumbnail_url=thumbnail_url,
            images=json.dumps(images) if images else None,
            videos=json.dumps(videos) if videos else None,
            published_time=published_time,
            scraped_time=datetime.utcnow().isoformat(),
            category=category,
            sub_category=sub_category,
            views_count=None,
            comments_count=None,
        )
