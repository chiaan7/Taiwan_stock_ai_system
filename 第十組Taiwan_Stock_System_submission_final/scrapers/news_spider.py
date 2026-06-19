from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import hashlib
import html
import re
import time
from typing import Callable, Iterable
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


POSITIVE_WORDS = (
    "買超",
    "成長",
    "利多",
    "創高",
    "旺",
    "擴產",
    "看好",
    "升級",
    "強勁",
    "正向",
    "優於預期",
)
NEGATIVE_WORDS = (
    "賣超",
    "衰退",
    "利空",
    "下修",
    "疲弱",
    "風險",
    "降評",
    "虧損",
    "壓力",
    "負向",
    "不如預期",
)
SKIP_TITLE_KEYWORDS = ("[公告]", "已被刪除", "Re: [公告]")


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    summary: str = ""
    published_at: str | None = None
    display_date: str = ""
    sentiment: str = "中立"
    content: str = ""
    keyword: str = ""
    stock_id: str = ""
    crawl_time: str = ""
    list_score: int | None = None
    push_count: int | None = None
    boo_count: int | None = None
    arrow_count: int | None = None
    reply_count: int | None = None


_PTT_SESSION: requests.Session | None = None


def _read_text(url: str, timeout: int = 15, retries: int = 3, backoff: float = 0.8) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        },
    )
    last_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        try:
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="ignore")
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    if last_error:
        raise last_error
    return ""


def _ptt_session() -> requests.Session:
    global _PTT_SESSION
    if _PTT_SESSION is not None:
        return _PTT_SESSION

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        }
    )
    try:
        session.post("https://www.ptt.cc/ask/over18", data={"yes": "yes"}, timeout=10)
    except Exception:
        pass
    _PTT_SESSION = session
    return session


def _read_ptt_text(url: str, timeout: int = 15, retries: int = 3) -> str:
    last_error: Exception | None = None
    session = _ptt_session()
    for attempt in range(retries):
        try:
            response = session.get(url, cookies={"over18": "1"}, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except Exception as exc:
            last_error = exc
            time.sleep(1.2 + attempt)
    if last_error:
        raise last_error
    return ""


def classify_sentiment(text: str) -> str:
    positive = sum(text.count(word) for word in POSITIVE_WORDS)
    negative = sum(text.count(word) for word in NEGATIVE_WORDS)
    if positive > negative:
        return "正向"
    if negative > positive:
        return "負向"
    return "中立"


def fetch_yahoo_news(
    stock_id: str,
    stock_name: str = "",
    limit: int = 50,
    months: int = 3,
    fetch_content: bool = True,
    errors: list[dict[str, str]] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[NewsItem]:
    cutoff = datetime.now() - timedelta(days=months * 31)
    keyword = stock_name or str(stock_id)
    items: list[NewsItem] = []

    rss_urls = [f"https://tw.stock.yahoo.com/rss?keyword={quote(keyword)}"]
    if stock_id:
        rss_urls.insert(0, f"https://tw.stock.yahoo.com/rss?s={quote(str(stock_id))}")
    for url in rss_urls:
        items.extend(
            _fetch_yahoo_rss(
                url,
                keyword,
                cutoff,
                limit - len(items),
                fetch_content,
                errors,
                progress,
                len(items),
                limit,
            )
        )
        if len(items) >= limit:
            return _dedupe_items(items)[:limit]

    html_urls = [f"https://tw.stock.yahoo.com/search?keyword={quote(keyword)}"]
    if stock_id:
        html_urls = [
            f"https://tw.stock.yahoo.com/quote/{quote(str(stock_id))}/news",
            f"https://tw.stock.yahoo.com/q/h?s={quote(str(stock_id))}",
            *html_urls,
        ]
    for url in html_urls:
        items.extend(
            _fetch_yahoo_news_html(
                url,
                keyword,
                cutoff,
                limit - len(items),
                fetch_content,
                errors,
                progress,
                len(items),
                limit,
            )
        )
        if len(items) >= limit:
            break

    return _dedupe_items(items)[:limit]


def fetch_google_news(
    keyword: str,
    limit: int = 50,
    days: int = 7,
    fetch_content: bool = True,
    errors: list[dict[str, str]] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[NewsItem]:
    """Fetch Google News search results and then extract title/content from each target."""
    query = quote(f"{keyword} when:{days}d")
    url = f"https://news.google.com/search?q={query}&hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant"
    try:
        page_html = _read_text(url)
    except Exception as exc:
        _record_error(errors, "google-news", url, exc)
        return []

    soup = BeautifulSoup(page_html, "html.parser")
    links = soup.select("a.WwrzSb")
    if not links:
        links = soup.select('a[href*="./read/"], a[href*="/read/"]')

    items: list[NewsItem] = []
    seen_urls: set[str] = set()
    for link in links:
        href = link.get("href", "")
        if not href:
            continue
        news_url = urljoin("https://news.google.com/", href)
        if news_url in seen_urls:
            continue
        seen_urls.add(news_url)

        article_node = link.find_parent("article")
        fallback_title = _extract_google_listing_title(article_node) if article_node else ""
        fallback_source = _extract_google_listing_source(article_node) if article_node else "Google News"
        fallback_date = _extract_google_listing_date(article_node) if article_node else None
        article_title, article_content, final_url = _fetch_title_and_content(
            news_url,
            fallback_title=fallback_title or link.get("aria-label", "") or link.get_text(" ", strip=True),
            errors=None,
            source="google-news-article",
        )
        if not article_title and not article_content:
            continue
        text_for_sentiment = f"{article_title} {article_content}"
        items.append(
            NewsItem(
                title=article_title or clean_text(keyword),
                url=final_url or news_url,
                source=fallback_source or "Google News",
                summary=_summarize_text(article_content),
                published_at=fallback_date,
                sentiment=classify_sentiment(text_for_sentiment),
                content=article_content,
                keyword=keyword,
            )
        )
        progress and progress(len(items), limit, f"Google news {len(items)}/{limit}")
        if len(items) >= limit:
            break
    return _dedupe_items(items)[:limit]


def _extract_google_listing_title(article_node) -> str:
    if not article_node:
        return ""
    for selector in ("a.JtKRv", "h3 a", "h4 a", "a[aria-label]"):
        node = article_node.select_one(selector)
        if node:
            title = clean_text(node.get_text(" ", strip=True) or node.get("aria-label", ""))
            if title:
                return title
    return ""


def _extract_google_listing_source(article_node) -> str:
    if not article_node:
        return "Google News"
    for selector in ("div.vr1PYe", ".wEwyrc", "[data-n-tid]"):
        node = article_node.select_one(selector)
        if node:
            source = clean_text(node.get_text(" ", strip=True))
            if source:
                return f"Google News / {source}"
    return "Google News"


def _extract_google_listing_date(article_node) -> str | None:
    if not article_node:
        return None
    node = article_node.select_one("time")
    if not node:
        return None
    raw_datetime = node.get("datetime", "")
    if raw_datetime:
        try:
            return datetime.fromisoformat(raw_datetime.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            pass
    text = clean_text(node.get_text(" ", strip=True))
    return text or None


def fetch_yahoo_news_scroll(
    stock_id: str,
    stock_name: str = "",
    limit: int = 50,
    months: int = 3,
    scroll_rounds: int = 20,
    fetch_content: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    errors: list[dict[str, str]] | None = None,
) -> list[NewsItem]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _record_error(errors, "yahoo-scroll", "playwright_import", exc)
        return []

    cutoff = datetime.now() - timedelta(days=months * 31)
    keyword = stock_name or str(stock_id)
    url = f"https://tw.stock.yahoo.com/quote/{quote(str(stock_id))}/news"
    items: list[NewsItem] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(locale="zh-TW")
            page.goto(url, wait_until="networkidle", timeout=30000)
            seen_urls: set[str] = set()
            stagnant_rounds = 0

            for step in range(1, scroll_rounds + 1):
                progress and progress(step, scroll_rounds, f"Yahoo scroll {step}/{scroll_rounds}")
                before_count = len(seen_urls)
                items.extend(
                    _extract_yahoo_items_from_html(
                        page.content(),
                        keyword,
                        cutoff,
                        limit - len(items),
                        fetch_content,
                        seen_urls,
                        errors,
                    )
                )
                progress and progress(len(items), limit, f"Yahoo news {len(items)}/{limit}")
                if len(items) >= limit:
                    break
                if len(seen_urls) == before_count:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0
                if stagnant_rounds >= 4:
                    break
                page.mouse.wheel(0, 2800)
                page.wait_for_timeout(900)

            browser.close()
    except Exception as exc:
        _record_error(errors, "yahoo-scroll", url, exc)

    return _dedupe_items(items)[:limit]


def fetch_ptt_stock_posts(
    keyword: str,
    stock_id: str = "",
    max_pages: int = 80,
    days: int = 93,
    fetch_content: bool = True,
    content_search: bool = True,
    sleep_seconds: float = 0.15,
    progress: Callable[[int, int, str], None] | None = None,
    errors: list[dict[str, str]] | None = None,
    include_replies: bool = False,
) -> list[NewsItem]:
    """Collect recent PTT Stock board posts matching stock name/id in title or content."""
    url = "https://www.ptt.cc/bbs/Stock/index.html"
    cutoff = datetime.now() - timedelta(days=days)
    terms = _ptt_search_terms(keyword, stock_id)
    items: list[NewsItem] = []

    for page_number in range(1, max_pages + 1):
        progress and progress(page_number, max_pages, f"PTT page {page_number}/{max_pages}")
        try:
            page_html = _read_ptt_text(url)
        except Exception as exc:
            _record_error(errors, "ptt-list", url, exc)
            break

        page_posts = _parse_ptt_listing(page_html)
        oldest_seen: datetime | None = None

        for post in page_posts:
            published = _parse_ptt_date(post["date"])
            if published and (oldest_seen is None or published < oldest_seen):
                oldest_seen = published
            if published and published < cutoff:
                continue

            title = post["title"]
            if _should_skip_ptt_title(title, include_replies=include_replies):
                continue
            href = str(post.get("href", ""))
            if not href:
                continue

            title_matches = _matches_terms(title, terms)
            full_url = "https://www.ptt.cc" + href if href.startswith("/") else href
            content = ""
            push_count = boo_count = arrow_count = reply_count = None
            should_fetch_detail = fetch_content and (content_search or title_matches or not terms)
            if should_fetch_detail:
                content, push_count, boo_count, arrow_count = _fetch_ptt_article_detail(full_url, errors=errors)
                reply_count = push_count + boo_count + arrow_count if push_count is not None else None

            text_for_sentiment = f"{title} {content}"
            if fetch_content and not content:
                continue
            if terms and not _matches_terms(text_for_sentiment, terms):
                continue

            items.append(
                NewsItem(
                    title=title,
                    url=full_url,
                    source="PTT Stock",
                    summary=_summarize_text(content),
                    published_at=published.strftime("%Y-%m-%d") if published else None,
                    display_date=_format_ptt_display_date(post["date"]),
                    sentiment=classify_sentiment(text_for_sentiment),
                    content=content,
                    keyword=keyword,
                    stock_id=stock_id,
                    list_score=post["list_score"],
                    push_count=push_count,
                    boo_count=boo_count,
                    arrow_count=arrow_count,
                    reply_count=reply_count,
                )
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)

        if oldest_seen and oldest_seen < cutoff:
            break

        prev_url = _find_ptt_prev_url(page_html)
        if not prev_url:
            break
        url = "https://www.ptt.cc" + prev_url
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return _dedupe_items(items)


def build_rag_documents(
    stock_id: str,
    stock_name: str,
    news_items: Iterable[NewsItem],
    ptt_items: Iterable[NewsItem],
) -> list[dict[str, str | int]]:
    documents: list[dict[str, str | int]] = []
    for item in [*news_items, *ptt_items]:
        content = clean_text(item.content)
        title = clean_text(item.title)
        text = clean_text(f"{title}\n{content}") if content else title
        if not text:
            continue
        source_type = "ptt" if item.source == "PTT Stock" else "news"
        document_seed = item.url or f"{source_type}|{stock_id}|{title}"
        document_id = f"{source_type}:{hashlib.sha1(document_seed.encode('utf-8')).hexdigest()[:12]}"
        doc: dict[str, str | int] = {
            "document_id": document_id,
            "source_type": source_type,
            "stock_id": stock_id,
            "stock_name": stock_name,
            "source": item.source,
            "title": title,
            "url": item.url,
            "summary": item.summary or _summarize_text(item.content),
            "published_at": item.published_at or "",
            "crawl_time": item.crawl_time or datetime.now().isoformat(timespec="seconds"),
            "display_date": item.display_date,
            "sentiment": item.sentiment,
            "category": "社群討論" if source_type == "ptt" else "財經新聞",
            "content": content or title,
            "text": text,
        }
        if item.source == "PTT Stock":
            doc.update(
                {
                    "list_score": item.list_score if item.list_score is not None else 0,
                    "push_count": item.push_count if item.push_count is not None else 0,
                    "boo_count": item.boo_count if item.boo_count is not None else 0,
                    "arrow_count": item.arrow_count if item.arrow_count is not None else 0,
                    "reply_count": item.reply_count if item.reply_count is not None else 0,
                }
            )
        documents.append(doc)
    return documents


def summarize_sentiment(items: Iterable[NewsItem]) -> dict[str, int | str]:
    counts = {"正向": 0, "中立": 0, "負向": 0}
    for item in items:
        counts[item.sentiment] = counts.get(item.sentiment, 0) + 1
    label = max(counts, key=counts.get) if sum(counts.values()) else "資料不足"
    return {"label": label, **counts}


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _fetch_yahoo_rss(
    url: str,
    keyword: str,
    cutoff: datetime,
    limit: int,
    fetch_content: bool,
    errors: list[dict[str, str]] | None,
    progress: Callable[[int, int, str], None] | None,
    start_count: int,
    total_limit: int,
) -> list[NewsItem]:
    if limit <= 0:
        return []
    try:
        feed = _read_text(url)
        root = ET.fromstring(feed)
    except Exception as exc:
        _record_error(errors, "yahoo-rss", url, exc)
        return []

    items: list[NewsItem] = []
    for element in root.findall(".//item"):
        title = clean_text(element.findtext("title", ""))
        link = clean_text(element.findtext("link", ""))
        description = clean_text(element.findtext("description", ""))
        pub_date = _parse_rss_date(element.findtext("pubDate", ""))
        if pub_date and pub_date < cutoff:
            continue
        if keyword and keyword not in f"{title} {description}":
            continue
        content = _fetch_article_text(link, errors=errors, source="yahoo-article") if fetch_content and link else description
        text_for_sentiment = f"{title} {description} {content}"
        items.append(
            NewsItem(
                title=title,
                url=link,
                source="Yahoo 股市",
                summary=_summarize_text(content or description),
                published_at=pub_date.strftime("%Y-%m-%d") if pub_date else None,
                sentiment=classify_sentiment(text_for_sentiment),
                content=content or description,
                keyword=keyword,
            )
        )
        progress and progress(start_count + len(items), total_limit, f"Yahoo news {start_count + len(items)}/{total_limit}")
        if len(items) >= limit:
            break
    return items


def _fetch_yahoo_news_html(
    url: str,
    keyword: str,
    cutoff: datetime,
    limit: int,
    fetch_content: bool,
    errors: list[dict[str, str]] | None,
    progress: Callable[[int, int, str], None] | None,
    start_count: int,
    total_limit: int,
) -> list[NewsItem]:
    if limit <= 0:
        return []
    try:
        page_html = _read_text(url)
    except Exception as exc:
        _record_error(errors, "yahoo-html", url, exc)
        return []

    return _extract_yahoo_items_from_html(
        page_html,
        keyword,
        cutoff,
        limit,
        fetch_content,
        set(),
        errors,
        progress,
        start_count,
        total_limit,
    )


def _extract_yahoo_items_from_html(
    page_html: str,
    keyword: str,
    cutoff: datetime,
    limit: int,
    fetch_content: bool,
    seen_urls: set[str],
    errors: list[dict[str, str]] | None,
    progress: Callable[[int, int, str], None] | None = None,
    start_count: int = 0,
    total_limit: int | None = None,
) -> list[NewsItem]:
    items: list[NewsItem] = []
    for match in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page_html, re.S):
        href, raw_title = match.groups()
        title = clean_text(raw_title)
        if not title or len(title) < 6:
            continue
        if keyword and keyword not in title:
            continue
        if href.startswith("/"):
            href = "https://tw.stock.yahoo.com" + href
        if "tw.stock.yahoo.com" not in href:
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)
        content = _fetch_article_text(href, errors=errors, source="yahoo-article") if fetch_content else ""
        published = _extract_date_from_text(content)
        if published and published < cutoff:
            continue
        items.append(
            NewsItem(
                title=title,
                url=href,
                source="Yahoo 股市",
                summary=_summarize_text(content),
                published_at=published.strftime("%Y-%m-%d") if published else None,
                sentiment=classify_sentiment(f"{title} {content}"),
                content=content,
                keyword=keyword,
            )
        )
        if progress:
            total = total_limit or limit
            progress(start_count + len(items), total, f"Yahoo news {start_count + len(items)}/{total}")
        if len(items) >= limit:
            break
    return items


def _fetch_title_and_content(
    url: str,
    fallback_title: str = "",
    errors: list[dict[str, str]] | None = None,
    source: str = "article",
) -> tuple[str, str, str]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                    ),
                    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                },
            )
            with urlopen(request, timeout=15) as response:
                final_url = response.geturl()
                charset = response.headers.get_content_charset() or "utf-8"
                page_html = response.read().decode(charset, errors="ignore")
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
    else:
        if last_error:
            _record_error(errors, source, url, last_error)
        return clean_text(fallback_title), "", url

    soup = BeautifulSoup(page_html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)
    if not title:
        og_title = soup.select_one('meta[property="og:title"], meta[name="title"]')
        title = clean_text(og_title.get("content", "")) if og_title else ""
    content = _extract_article_content_from_html(page_html)
    return title or clean_text(fallback_title), content, final_url


def _extract_article_content_from_html(page_html: str) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    for trash in soup.select("script, style, noscript, nav, header, footer, aside"):
        trash.decompose()

    article = soup.find("article")
    if article:
        text = clean_text(article.get_text(" ", strip=True))
        if len(text) >= 80:
            return text

    meta = soup.select_one('meta[name="description"], meta[property="og:description"]')
    description = clean_text(meta.get("content", "")) if meta else ""
    paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    paragraph_text = clean_text(" ".join(p for p in paragraphs if len(p) >= 20))
    return paragraph_text or description


def _fetch_article_text(
    url: str,
    errors: list[dict[str, str]] | None = None,
    source: str = "article",
) -> str:
    _, content, _ = _fetch_title_and_content(url, errors=errors, source=source)
    return content


def _parse_ptt_listing(page_html: str) -> list[dict[str, str | int | None]]:
    soup = BeautifulSoup(page_html, "html.parser")
    posts: list[dict[str, str | int | None]] = []
    for rent in soup.select(".r-ent"):
        title_link = rent.select_one(".title a")
        title_node = rent.select_one(".title")
        if title_link:
            title = title_link.get_text(strip=True)
            href = title_link.get("href", "")
        else:
            title = title_node.get_text(" ", strip=True) if title_node else ""
            href = ""
        if not title:
            continue
        date_node = rent.select_one(".date")
        score_node = rent.select_one(".nrec span")
        posts.append(
            {
                "title": clean_text(title),
                "href": href,
                "date": clean_text(date_node.get_text(strip=True) if date_node else ""),
                "list_score": _parse_ptt_score(score_node.get_text(strip=True) if score_node else ""),
            }
        )
    return posts


def _fetch_ptt_article_detail(
    url: str,
    errors: list[dict[str, str]] | None = None,
) -> tuple[str, int | None, int | None, int | None]:
    try:
        page_html = _read_ptt_text(url)
    except Exception as exc:
        _record_error(errors, "ptt-article", url, exc)
        return "", None, None, None

    soup = BeautifulSoup(page_html, "html.parser")
    pushes = soup.select("div.push")
    push_count = boo_count = arrow_count = 0
    for push in pushes:
        tag = push.select_one("span.push-tag")
        if not tag:
            continue
        tag_text = tag.get_text(strip=True)
        if tag_text == "推":
            push_count += 1
        elif tag_text == "噓":
            boo_count += 1
        elif tag_text == "→":
            arrow_count += 1

    main = soup.find(id="main-content")
    if not main:
        return "", push_count, boo_count, arrow_count
    for trash in main.select("div.push, span.f2, div.article-metaline, div.article-metaline-right"):
        trash.decompose()
    content = clean_text(main.get_text(" ", strip=True))
    return content, push_count, boo_count, arrow_count


def _should_skip_ptt_title(title: str, include_replies: bool) -> bool:
    if not include_replies and title.startswith("Re:"):
        return True
    return any(keyword in title for keyword in SKIP_TITLE_KEYWORDS)


def _ptt_search_terms(keyword: str, stock_id: str) -> list[str]:
    aliases = {
        "2330": ["台積電", "台積", "TSMC"],
        "2317": ["鴻海", "富士康"],
        "2454": ["聯發科", "發哥", "MTK"],
        "2303": ["聯電", "UMC"],
        "2308": ["台達電", "台達"],
    }
    terms = {keyword.strip(), stock_id.strip()}
    terms.update(aliases.get(stock_id.strip(), []))
    if keyword.strip() == "台積電":
        terms.add("台積")
    if keyword.strip() == "聯發科":
        terms.add("發哥")
    return [term for term in terms if term]


def _matches_terms(text: str, terms: list[str]) -> bool:
    if not terms:
        return True
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def _parse_ptt_score(value: str) -> int | None:
    value = clean_text(value)
    if not value:
        return None
    if value == "爆":
        return 99
    if value == "XX":
        return -99
    if value.startswith("X"):
        try:
            return -int(value.lstrip("X") or "0") * 10
        except ValueError:
            return None
    try:
        return int(value)
    except ValueError:
        return None


def _find_ptt_prev_url(page_html: str) -> str | None:
    soup = BeautifulSoup(page_html, "html.parser")
    numbered_links: list[tuple[int, str]] = []
    for link in soup.select("a.btn.wide"):
        href = link.get("href", "")
        text = link.get_text(strip=True)
        if "上頁" in text:
            return href
        match = re.search(r"/bbs/Stock/index(\d+)\.html", href, re.I)
        if match:
            numbered_links.append((int(match.group(1)), href))
    if numbered_links:
        return max(numbered_links, key=lambda item: item[0])[1]
    return None


def _parse_ptt_date(value: str) -> datetime | None:
    value = clean_text(value)
    try:
        parsed = datetime.strptime(f"{datetime.now().year}/{value}", "%Y/%m/%d")
        if parsed > datetime.now() + timedelta(days=1):
            parsed = parsed.replace(year=parsed.year - 1)
        return parsed
    except Exception:
        return None


def _format_ptt_display_date(value: str) -> str:
    value = clean_text(value)
    match = re.match(r"(\d{1,2})/(\d{1,2})", value)
    if not match:
        return value
    return f"{int(match.group(1))} 月 {int(match.group(2))} 日"


def _parse_rss_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.replace(tzinfo=None)
    except Exception:
        return None


def _extract_date_from_text(value: str) -> datetime | None:
    match = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", value)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except Exception:
        return None


def _dedupe_items(items: Iterable[NewsItem]) -> list[NewsItem]:
    seen_keys: dict[str, int] = {}
    deduped: list[NewsItem] = []
    for item in items:
        keys = _dedupe_keys(item)
        if not keys:
            continue
        existing_index = next((seen_keys[key] for key in keys if key in seen_keys), None)
        if existing_index is not None:
            if _item_quality(item) > _item_quality(deduped[existing_index]):
                deduped[existing_index] = item
            continue
        for key in keys:
            seen_keys[key] = len(deduped)
        deduped.append(item)
    return deduped


def _dedupe_keys(item: NewsItem) -> list[str]:
    keys: list[str] = []
    title = clean_text(item.title).lower()
    title = re.sub(r"\s+", "", title)
    if title:
        keys.append(f"title:{title}")
    if item.url:
        keys.append(f"url:{item.url}")
    return keys


def _item_quality(item: NewsItem) -> int:
    return (20 if item.url else 0) + min(len(item.content or ""), 10000)


def _summarize_text(value: str, max_chars: int = 180) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _record_error(
    errors: list[dict[str, str]] | None,
    source: str,
    target: str,
    exc: Exception,
) -> None:
    if errors is None:
        return
    errors.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "target": target,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
