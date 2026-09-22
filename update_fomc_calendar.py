"""
update_fomc_calendar.py — maintenance tool that fetches the official
Federal Reserve FOMC meeting calendar and keeps hybrid_engine.py's
FOMC_DATES set in sync with it.

NOT part of the live trading path: hybrid_engine.py never imports this
file (no network dependency in the live --execute/--manage/--screen path),
and this file never imports hybrid_engine.py's broker/state-file machinery
— it only reads and rewrites the FOMC_DATES literal block in
hybrid_engine.py's own source text, as plain text surgery, then verifies
the result is still valid Python before keeping the change.

WHY THIS EXISTS: FOMC_DATES is a live macro-blackout gate — hybrid_engine.py
blocks new entries 1:45-3:30 PM ET on each listed date. A hand-maintained
list silently stops protecting once it runs out, and the Fed's own site
marks meeting dates more than ~1 year out as "tentative until confirmed at
the meeting immediately preceding it" (so a future date CAN move). This
script re-derives the list from the source of truth instead of trusting
memory or a stale snapshot.

WHAT COUNTS AS A BLACKOUT DATE: the SECOND day of each two-day FOMC
meeting (statement + press conference at ~2:00/2:30 PM ET) — the same
convention hybrid_engine.py's FOMC_DATES already uses. A single-day
"notation vote" entry (a procedural vote with no statement or press
conference — e.g. August 2025's "22 (notation vote)" row) is deliberately
EXCLUDED: it carries none of the market-moving announcement this gate
exists to black out, and its markup doesn't parse as a day-day range.

Usage:
    python3 update_fomc_calendar.py                  # fetch, diff, write if changed
    python3 update_fomc_calendar.py --years 2026 2027 2028
    python3 update_fomc_calendar.py --dry-run         # fetch + diff only, never writes
    python3 update_fomc_calendar.py --html-file page.html   # parse a local copy (debugging,
                                                              # skips the network fetch)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import BASE, LOGS

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HYBRID_ENGINE_FILE = BASE / "hybrid_engine.py"
DEFAULT_YEARS = {2026, 2027}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}

_FULL_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]
MONTH_LOOKUP: dict[str, int] = {}
for _i, _name in enumerate(_FULL_MONTHS, start=1):
    MONTH_LOOKUP[_name] = _i
    MONTH_LOOKUP[_name[:3]] = _i          # "Apr/May", "Jan/Feb", "Oct/Nov" use 3-letter codes

_DATE_RANGE_RE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\*?\s*$")
_YEAR_HEADER_RE = re.compile(r"(\d{4})\s+FOMC Meetings")
_FOMC_DATES_BLOCK_RE = re.compile(
    r"(FOMC_DATES: set\[dt\.date\] = \{\n)(.*?)(\n\})", re.DOTALL)
_DATE_LITERAL_RE = re.compile(r"dt\.date\((\d+),\s*(\d+),\s*(\d+)\)")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
             logging.FileHandler(LOGS / "fomc_calendar_update.log")],
)
logger = logging.getLogger("update_fomc_calendar")


# ---------------------------------------------------------------- fetch
def fetch_calendar_html(url: str = FOMC_CALENDAR_URL, retries: int = 4,
                        backoff_base: float = 1.5, timeout: float = 15.0) -> str:
    """GET the Fed's calendar page with clean retries (exponential backoff).
    Raises RuntimeError, chained from the last underlying exception, if
    every attempt fails — this tool runs a handful of times a year, not on
    a tight schedule, so failing loudly beats guessing."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            if not resp.text or "FOMC" not in resp.text:
                raise ValueError("response did not look like the FOMC calendar page")
            return resp.text
        except Exception as e:                                      # noqa: BLE001
            last_exc = e
            wait = backoff_base ** attempt
            msg = f"fetch attempt {attempt}/{retries} failed ({type(e).__name__}: {e})"
            if attempt < retries:
                logger.warning("%s — retrying in %.1fs", msg, wait)
                time.sleep(wait)
            else:
                logger.error(msg)
    assert last_exc is not None
    raise RuntimeError(f"could not fetch {url} after {retries} attempts") from last_exc


# ---------------------------------------------------------------- parse
def _second_meeting_day(month_label: str, date_range: str, year: int) -> dt.date | None:
    """
    `month_label` is e.g. "January" or "Apr/May" (a meeting straddling a
    month boundary, using 3-letter codes); `date_range` is e.g. "27-28",
    "17-18*", or a non-range entry like "22 (notation vote)" (deliberately
    rejected — see module docstring). Returns the SECOND day's calendar
    date, or None if this entry isn't a real two-day meeting.
    """
    m = _DATE_RANGE_RE.match(date_range)
    if not m:
        return None
    end_day = int(m.group(2))

    months = [mo.strip().lower() for mo in month_label.split("/")]
    start_month = MONTH_LOOKUP.get(months[0])
    end_month = MONTH_LOOKUP.get(months[-1])
    if start_month is None or end_month is None:
        return None

    end_year = year
    if end_month < start_month:            # Dec/Jan-style wrap (none seen yet; defensive)
        end_year += 1
    try:
        return dt.date(end_year, end_month, end_day)
    except ValueError:
        return None


def parse_fomc_dates(html: str, years: set[int]) -> dict[int, list[dt.date]]:
    """
    Parses every real two-day FOMC meeting for each year in `years` from
    the Fed's calendar page markup: a
    `<h4><a id="...">YYYY FOMC Meetings</a></h4>` section header, followed
    (in document order, not necessarily direct siblings) by
    `fomc-meeting__month` / `fomc-meeting__date` div pairs, one per
    meeting, until the next such header. Returns
    {year: [sorted second-meeting-day dates]} — a year present in `years`
    but absent from the result means its section wasn't found on the page
    at all (different from a section that WAS found but is legitimately
    empty, which can't happen for a real FOMC year).
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[int, list[dt.date]] = {}

    year_headers: list[tuple[int, object]] = []
    for h in soup.find_all("h4"):
        a = h.find("a", id=True)
        if a is None:
            continue
        m = _YEAR_HEADER_RE.match(a.get_text(strip=True))
        if m:
            year_headers.append((int(m.group(1)), h))

    for year, header in year_headers:
        if year not in years:
            continue
        found: list[dt.date] = []
        node = header
        while True:
            node = node.find_next(["h4", "div"])
            if node is None or node.name == "h4":
                break
            classes = node.get("class") or []
            if "fomc-meeting__month" in classes:
                month_label = node.get_text(strip=True)
                date_div = node.find_next(class_="fomc-meeting__date")
                if date_div is None:
                    continue
                d = _second_meeting_day(month_label, date_div.get_text(strip=True), year)
                if d is not None:
                    found.append(d)
        out[year] = sorted(set(found))
    return out


# ---------------------------------------------------------------- render + merge
def _render_fomc_dates_block(dates: set[dt.date]) -> str:
    """Reproduces the existing FOMC_DATES literal's exact style: 4-per-line,
    4-space indent, `dt.date(Y, M, D)` calls — so a diff of hybrid_engine.py
    after this tool runs shows only real content changes."""
    ordered = sorted(dates)
    lines = []
    for i in range(0, len(ordered), 4):
        chunk = ordered[i:i + 4]
        line = "    " + " ".join(f"dt.date({d.year}, {d.month}, {d.day})," for d in chunk)
        lines.append(line)
    return "\n".join(lines)


def merge_fomc_dates(existing: set[dt.date], parsed: dict[int, list[dt.date]],
                     years: set[int]) -> tuple[set[dt.date], list[int]]:
    """
    Combines `existing` (whatever hybrid_engine.py currently has) with
    freshly `parsed` dates for `years`. A year that was requested but is
    MISSING from `parsed` (page structure changed, section not found, etc.)
    is left exactly as it was in `existing`, rather than silently wiped —
    returned in the second element as `unparsed_years` so the caller can
    warn loudly instead of quietly deleting a year's blackout coverage.
    """
    unparsed_years = [y for y in years if y not in parsed]
    kept = {d for d in existing if d.year not in years}
    updated_scope: set[dt.date] = set()
    for year in years:
        if year in parsed:
            updated_scope |= set(parsed[year])
        else:
            updated_scope |= {d for d in existing if d.year == year}
    return kept | updated_scope, unparsed_years


# ---------------------------------------------------------------- file surgery
def load_current_fomc_dates(source: str) -> set[dt.date]:
    m = _FOMC_DATES_BLOCK_RE.search(source)
    if m is None:
        raise RuntimeError(f"could not find the FOMC_DATES block in {HYBRID_ENGINE_FILE}")
    block = m.group(2)
    return {dt.date(int(y), int(mo), int(d)) for y, mo, d in _DATE_LITERAL_RE.findall(block)}


def write_fomc_dates(source: str, new_dates: set[dt.date]) -> str:
    m = _FOMC_DATES_BLOCK_RE.search(source)
    if m is None:
        raise RuntimeError(f"could not find the FOMC_DATES block in {HYBRID_ENGINE_FILE}")
    rendered = _render_fomc_dates_block(new_dates)
    return source[:m.start()] + m.group(1) + rendered + m.group(3) + source[m.end():]


def apply_update(new_dates: set[dt.date], dry_run: bool = False) -> bool:
    """
    Rewrites hybrid_engine.py's FOMC_DATES block to `new_dates`, atomically
    and with a timestamped backup, ONLY if the content actually changed and
    the resulting file still parses as valid Python — this is live
    production source code, so a failed sanity check aborts the write
    entirely rather than leaving a broken file behind. Returns True if a
    write happened.
    """
    source = HYBRID_ENGINE_FILE.read_text()
    current = load_current_fomc_dates(source)

    added = sorted(new_dates - current)
    removed = sorted(current - new_dates)
    if not added and not removed:
        logger.info("FOMC_DATES: no changes needed (%d dates already up to date)",
                    len(current))
        return False

    logger.info("FOMC_DATES: %d date(s) added, %d date(s) removed", len(added), len(removed))
    for d in added:
        logger.info("  + %s", d.isoformat())
    for d in removed:
        logger.info("  - %s", d.isoformat())

    if dry_run:
        logger.info("--dry-run: not writing (would update hybrid_engine.py)")
        return False

    new_source = write_fomc_dates(source, new_dates)

    try:
        ast.parse(new_source)
    except SyntaxError as e:
        logger.error("Refusing to write: rendered hybrid_engine.py would not parse "
                     "as valid Python (%s) — file left untouched", e)
        raise

    backup_path = HYBRID_ENGINE_FILE.with_suffix(
        f".py.bak.{dt.datetime.now():%Y%m%dT%H%M%S}")
    backup_path.write_text(source)

    fd, tmp = tempfile.mkstemp(dir=HYBRID_ENGINE_FILE.parent,
                               prefix=".hybrid_engine.", suffix=".py.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(new_source)
        os.replace(tmp, HYBRID_ENGINE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("FOMC_DATES: hybrid_engine.py updated (backup: %s)", backup_path.name)
    return True


# ---------------------------------------------------------------- CLI
def run(years: set[int], dry_run: bool, html_file: str | None) -> int:
    if html_file:
        logger.info("Reading local HTML file %s (skipping network fetch)", html_file)
        html = Path(html_file).read_text()
    else:
        logger.info("Fetching %s ...", FOMC_CALENDAR_URL)
        html = fetch_calendar_html()

    parsed = parse_fomc_dates(html, years)
    for year in sorted(years):
        n = len(parsed.get(year, []))
        logger.info("Parsed %d meeting date(s) for %d%s", n, year,
                   "" if year in parsed else " (SECTION NOT FOUND ON PAGE)")

    source = HYBRID_ENGINE_FILE.read_text()
    existing = load_current_fomc_dates(source)
    new_dates, unparsed_years = merge_fomc_dates(existing, parsed, years)

    if unparsed_years:
        logger.warning("Years requested but not found on the page: %s — their existing "
                       "FOMC_DATES entries were left untouched, not wiped", unparsed_years)

    try:
        apply_update(new_dates, dry_run=dry_run)
    except Exception:                                                 # noqa: BLE001
        logger.exception("FOMC_DATES update failed")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sync hybrid_engine.py's FOMC_DATES with the official Fed calendar")
    ap.add_argument("--years", type=int, nargs="+", default=sorted(DEFAULT_YEARS),
                    help=f"years to fetch/sync (default: {sorted(DEFAULT_YEARS)})")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + diff only; never writes hybrid_engine.py")
    ap.add_argument("--html-file", default=None,
                    help="parse a local HTML file instead of fetching (debugging)")
    args = ap.parse_args()
    return run(set(args.years), args.dry_run, args.html_file)


if __name__ == "__main__":
    sys.exit(main())
