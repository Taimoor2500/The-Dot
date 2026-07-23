"""Fetch and extract main article text from a URL."""

import trafilatura
from trafilatura.settings import use_config

# trafilatura's default config retries failed connections with a backoff factor
# of DOWNLOAD_TIMEOUT/2 seconds (30/2=15s), so a single blocked/dead site can
# cost 30-40s. We don't need any specific article that badly -- GDELT gives us
# many sources -- so fail fast instead.
_FAST_CONFIG = use_config()
_FAST_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "8")
_FAST_CONFIG.set("DEFAULT", "MAX_REDIRECTS", "1")


def fetch_article_text(url: str) -> str | None:
    """Download the page at `url` and extract its main body text.

    Returns None if the page can't be fetched or no extractable content is found
    (paywalled/blocked pages, dead links) -- callers should skip those articles.
    """
    downloaded = trafilatura.fetch_url(url, config=_FAST_CONFIG)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded, favor_precision=True)
