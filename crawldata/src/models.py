"""Data model for a news article.

Defines a dataclass matching the 22‑field schema required for storage.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Article:
    # ID / URL fields
    article_id: str
    url: str
    domain: str
    category_url: Optional[str]
    author_url: Optional[str]
    amp_url: Optional[str]
    reference_url: Optional[str]
    short_url: Optional[str]

    # Content fields
    title: Optional[str]
    description: Optional[str]
    main_content: Optional[str]
    author_name: Optional[str]
    tags: Optional[str]  # comma‑separated string

    # Media fields
    thumbnail_url: Optional[str]
    images: Optional[str]   # JSON list string
    videos: Optional[str]   # JSON list string

    # Metadata fields
    published_time: Optional[str]
    scraped_time: str  # ISO‑8601
    category: Optional[str]
    sub_category: Optional[str]
    views_count: Optional[int]
    comments_count: Optional[int]
    
    # Raw extracted extra fields (JSON string)
    extra_data: Optional[str] = None
