import requests
from datetime import datetime, timedelta
import re
import json
import time

def daterange(start_date, end_date):
    """Yield every date between start_date and end_date (inclusive)."""
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

# Regex to match RSP lines with desired particulars
PART_RE = re.compile(
    r'^(?P<event>\S+)\s+'
    r'(?P<begin>\S+)\s+(?P<max>\S+)\s+(?P<end>\S+)\s+'
    r'(?P<obs>\S+)\s+(?P<Q>\S+)\s+'
    r'RSP\s+'
    r'(?P<locfrq>\S+)\s+'
    r'(?P<particulars>(?:II|IV|V)(?:/\d+)?)\s*(?P<reg>\S*)\s*$'
)
def fetch_rsp_entries_for_date(d):
    """Fetch the text file for date d and return matched RSP entries."""
    url = (
        f"https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"           #### Used alt URL becoz solarmonitor.org blocks ip address due to multiple requests
        f"daily_reports/solar_event_reports/{d.year:04d}/{d.month:02d}/"
        f"{d.year:04d}{d.month:02d}{d.day:02d}events.txt"
    )
    try:
        r = requests.get(url, timeout=10)
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
                'particulars': m.group('particulars'),
                'reg': m.group('reg'),
                'raw':    line.strip(),
            })
    return entries

def main():
    start = datetime(2013, 3, 1)    # Start date for LOFAR catalog
    end   = datetime(2024, 8, 31)   # End date for LOFAR catalog
    all_entries = []

    for single_date in daterange(start, end):
        print("Fetching data for {}...".format(single_date.strftime("%Y-%m-%d")))
        entries = fetch_rsp_entries_for_date(single_date)
        if entries:
            all_entries.extend(entries)
            for e in entries:
                print("{date}  {raw}".format(date=e["date"], raw=e["raw"]))
        else:
            print("No entries found for {}".format(single_date.strftime("%Y-%m-%d")))

    # Save to JSON file
    with open('rsp_events.json', 'w') as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nSaved {len(all_entries)} entries to rsp_events.json")

if __name__ == "__main__":
    main()

