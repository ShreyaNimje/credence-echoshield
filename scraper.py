import trafilatura
from bs4 import BeautifulSoup
import requests

def extract_article_content(url: str) -> dict:
    """
    Downloads webpage and strips boilerplate using trafilatura's internal fetcher,
    with a browser-spoofed requests fallback.
    """
    try:
        # 1. Trafilatura's built-in robust fetcher (bypasses many standard bot traps)
        downloaded = trafilatura.fetch_url(url)
        
        # 2. Fallback to requests with full browser headers if fetch_url fails
        if not downloaded:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
            }
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            downloaded = resp.text

        # 3. Clean boilerplate (ads, menus, footers)
        clean_text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )

        # 4. Fallback title and text extraction via BeautifulSoup
        soup = BeautifulSoup(downloaded, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "Extracted Article"

        if not clean_text or len(clean_text.split()) < 30:
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            article_body = soup.find("article") or soup.find("main") or soup.find("body")
            paragraphs = [p.get_text().strip() for p in article_body.find_all("p") if len(p.get_text().strip()) > 25]
            clean_text = " ".join(paragraphs)

        return {
            "success": True,
            "title": title,
            "text": clean_text.strip(),
            "url": url,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "title": "",
            "text": "",
            "url": url,
            "error": str(e)
        }