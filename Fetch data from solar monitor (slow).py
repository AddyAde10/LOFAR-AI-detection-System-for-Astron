
import requests
from datetime import datetime, timedelta
import re
import json
import time
import random
import os

# -------------------------------
# Configuration
# -------------------------------
START_DATE = datetime(2013, 3, 1)
END_DATE   = datetime(2024, 8, 31)
BATCH_YEARS = [(2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024)]
OUTPUT_FILE = 'rsp_events.json'
TEMP_DIR    = 'temp_batches'
MIN_SLEEP   = 1.0   # seconds
MAX_SLEEP   = 3.0   # seconds

# -------------------------------
# Prepare regex
# -------------------------------
PART_RE = re.compile(
    r'^(?P<event>\d+)\s+'
    r'(?P<begin>\S+)\s+(?P<max>\S+)\s+(?P<end>\S+)\s+'
    r'(?P<obs>\S+)\s+(?P<Q>\S+)\s+'
    r'RSP\s+'
    r'(?P<locfrq>\S+)\s+'
    r'(?P<part>(?:II|IV|V)(?:/\d+)?)\b'
)

# -------------------------------
# Helper functions
# -------------------------------
def daterange(start_date, end_date):
    """Yield every date between start_date and end_date (inclusive)."""
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

def build_url(d: datetime) -> str:
    """
    Build the URL for the NOAA NGDC daily report for date d.
    Format: https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/
            daily_reports/solar_event_reports/YYYY/MM/YYYYMMDDevents.txt
    """
    return (
        f"https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
        f"daily_reports/solar_event_reports/{d.year:04d}/{d.month:02d}/"
        f"{d.year:04d}{d.month:02d}{d.day:02d}events.txt"
    )


def fetch_rsp_entries_for_date(session: requests.Session, d: datetime):
    """Fetch the text file for date d using session and return matched RSP entries."""
    url = build_url(d)
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return []
    entries = []
    for line in r.text.splitlines():
        m = PART_RE.match(line)
        if m:
            entries.append({
                'date':   d.strftime('%Y-%m-%d'),
                'event':  m.group('event'),
                'begin':  m.group('begin'),
                'max':    m.group('max'),
                'end':    m.group('end'),
                'obs':    m.group('obs'),
                'Q':      m.group('Q'),
                'locfrq': m.group('locfrq'),
                'part':   m.group('part'),
                'raw':    line.strip(),
            })
    return entries


def save_batch(batch_entries, batch_name):
    """Save a batch’s entries to a temp JSON (so we can resume)."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"{batch_name}.json")
    with open(path, 'w') as f:
        json.dump(batch_entries, f, indent=2)
    print(f"[{batch_name}]  Saved {len(batch_entries)} entries to {path}")

def load_batch(batch_name):
    """Load a batch if it exists (to resume)."""
    path = os.path.join(TEMP_DIR, f"{batch_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"[{batch_name}]  Loaded {len(data)} existing entries from {path}")
        return data
    return []

# -------------------------------
# Main routine
# -------------------------------
def main():
    # Prepare session with retries + backoff
    session = requests.Session()
    session.headers.update({'User-Agent': 'BurstHarvester/1.0'})
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    all_entries = []

    # Process in yearly batches
    for (y0, y1) in BATCH_YEARS:
        batch_name = f"{y0}-{y1}"
        batch_entries = load_batch(batch_name)
        start = max(START_DATE, datetime(y0, 1, 1))
        end   = min(END_DATE,   datetime(y1, 12, 31))

        processed_dates = {e['date'] for e in batch_entries}
        for single_date in daterange(start, end):
            ds = single_date.strftime('%Y-%m-%d')
            if ds in processed_dates:
                continue

            entries = fetch_rsp_entries_for_date(session, single_date)
            if entries:
                batch_entries.extend(entries)
                for e in entries:
                    print(f"{e['date']}  {e['raw']}")

            time.sleep(MIN_SLEEP + random.random() * (MAX_SLEEP - MIN_SLEEP))

        save_batch(batch_entries, batch_name)
        all_entries.extend(batch_entries)
        print(f"Completed batch {batch_name}, sleeping 60s before next batch…")
        time.sleep(60)

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nAll done: saved total {len(all_entries)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()