import asyncio, random, nltk, db, httpx
from typing import Dict, Any, Optional, List
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from pydantic import BaseModel
from datetime import datetime

class ScraperResponse(BaseModel):
    title: Optional[str]
    authors: Optional[List[str]]
    publication: Optional[str]
    published_at: Optional[datetime]
    text: Optional[str]

# Ensure NLTK resources
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# Browser-like user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/118.0",
]

# --- Step 1: Reliable HTTPX fetch with retries ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch_with_retries(client: httpx.AsyncClient, url: str) -> str:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    response = await client.get(url, headers=headers, follow_redirects=True)
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code}",
            request=response.request,
            response=response,
        )
    return response.text


# --- Step 2: Playwright fallback for stubborn sites ---
async def fetch_with_playwright(url: str) -> str:
    try:
        import sys
        from playwright.async_api import async_playwright

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        print(f"⚙️  Falling back to Playwright for {url}")
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            html = await page.content()
            await browser.close()
            return html

    except Exception as e:
        raise RuntimeError(f"Playwright fallback failed: {e}")


# --- Step 3: Main scraper integrating both ---
async def scrape_article(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        try:
            html = await fetch_with_retries(client, url)

        except RetryError as e:
            # Extract the final inner exception (e.g. HTTPStatusError)
            last_exc = e.last_attempt.exception()
            if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 403:
                try:
                    html = await fetch_with_playwright(url)
                except Exception as e2:
                    return {"error": f"Playwright fallback failed: {e2}", "url": url}
            else:
                return {
                    "error": f"Fetch failed after retries: {type(last_exc).__name__}: {last_exc}",
                    "url": url,
                }

        except httpx.HTTPStatusError as e:
            # (Non-retry path, just in case)
            if e.response.status_code == 403:
                try:
                    html = await fetch_with_playwright(url)
                except Exception as e2:
                    return {"error": f"Playwright fallback failed: {e2}", "url": url}
            else:
                return {"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}", "url": url}

        except Exception as e:
            return {"error": f"Unexpected error fetching {url}: {e}"}

    def _parse_html(html: str) -> Dict[str, Any]:
        art = Article(url)
        art.set_html(html)
        art.parse()
        # try:
        #     art.nlp()
        # except Exception:
        #     pass

        text = str(art.text or "").strip()

        result = ScraperResponse(
            title=art.title,
            authors=art.authors,
            publication=art.source_url or None,
            published_at=getattr(art, "publish_date", None),
            text=text,
        )

        return result.model_dump()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _parse_html, html)
