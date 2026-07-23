"""SQLite-backed buffer for raw fetched articles."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path("data/articles.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url TEXT PRIMARY KEY,
    title TEXT,
    domain TEXT,
    language TEXT,
    seendate TEXT,
    body_text TEXT,
    fetched_at TEXT NOT NULL,
    extracted_at TEXT
);

CREATE TABLE IF NOT EXISTS extraction_failures (
    url TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    error TEXT,
    raw_response TEXT
);
"""


@dataclass
class Article:
    url: str
    title: str
    domain: str | None
    language: str | None
    seendate: str | None
    body_text: str


@contextmanager
def connect(db_path: Path = DEFAULT_DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_article(conn: sqlite3.Connection, article: Article) -> None:
    conn.execute(
        """
        INSERT INTO articles (url, title, domain, language, seendate, body_text, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title,
            body_text=excluded.body_text,
            fetched_at=excluded.fetched_at
        """,
        (
            article.url,
            article.title,
            article.domain,
            article.language,
            article.seendate,
            article.body_text,
            datetime.utcnow().isoformat(),
        ),
    )


def unextracted_articles(conn: sqlite3.Connection, limit: int = 100) -> list[Article]:
    rows = conn.execute(
        """
        SELECT url, title, domain, language, seendate, body_text
        FROM articles
        WHERE extracted_at IS NULL AND body_text IS NOT NULL
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [Article(*row) for row in rows]


def mark_extracted(conn: sqlite3.Connection, url: str) -> None:
    conn.execute(
        "UPDATE articles SET extracted_at = ? WHERE url = ?",
        (datetime.utcnow().isoformat(), url),
    )


def log_extraction_failure(
    conn: sqlite3.Connection, url: str, error: str, raw_response: str | None
) -> None:
    conn.execute(
        "INSERT INTO extraction_failures (url, attempted_at, error, raw_response) VALUES (?, ?, ?, ?)",
        (url, datetime.utcnow().isoformat(), error, raw_response),
    )
