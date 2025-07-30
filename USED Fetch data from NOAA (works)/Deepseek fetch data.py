import requests
from datetime import datetime, timedelta
import json
import re

def daterange(start_date, end_date):
    """Yield every date between start_date and end_date (inclusive)."""
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

def parse_events(content, current_date):
    """Parse event data using fixed-width column positions."""
    lines = content.splitlines()
    header_index = None
    header_line = None
    
    # Find the header line
    for i, line in enumerate(lines):
        if line.startswith("#Event"):
            header_line = line
            header_index = i
            break
    
    if header_index is None:
        return []
    
    # Define expected columns in order of appearance
    columns = ["Event", "Begin", "Max", "End", "Obs", "Q", "Type", "Loc/Frq", "Particulars", "Reg#"]
    col_positions = []
    
    # Find start positions of each column
    for col in columns:
        pos = header_line.find(col)
        if pos == -1:
            # Fallback if standard columns not found
            return []
        col_positions.append(pos)
    
    # Add end position for last column
    col_positions.append(None)
    
    events = []
    # Process data lines (skip header and separator line)
    for line in lines[header_index+2:]:
        if not line.strip() or line.startswith('#'):
            continue
            
        # Extract fields using column positions
        fields = {}
        for i in range(len(columns)):
            start = col_positions[i]
            end = col_positions[i+1] if i < len(columns) - 1 else None
            
            # Handle field extraction with bounds checking
            if end is None:
                field_value = line[start:].strip()
            else:
                field_value = line[start:end].strip()
            fields[columns[i]] = field_value
        
        # Check if it's an RSP event with desired Particulars pattern
        if fields.get("Type") == "RSP":
            particulars = fields.get("Particulars", "")
            if re.match(r'^(I{1,3}|IV|V)(?:/|\b)', particulars):
                event_data = {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "event": fields["Event"],
                    "begin": fields["Begin"],
                    "max": fields["Max"],
                    "end": fields["End"],
                    "obs": fields["Obs"],
                    "q": fields["Q"],
                    "loc_frq": fields["Loc/Frq"],
                    "particulars": particulars,
                    "reg": fields.get("Reg#", ""),
                    "raw_line": line.strip()
                }
                events.append(event_data)
                
    return events

def fetch_events_for_date(date):
    """Fetch events for a specific date."""
    url = (
        f"https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/"
        f"daily_reports/solar_event_reports/{date.year}/{date.month:02d}/"
        f"{date.year}{date.month:02d}{date.day:02d}events.txt"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None

def main():
    start_date = datetime(2022, 5, 2)
    end_date = datetime(2024, 8, 31)
    all_events = []
    
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        print(f"Processing {date_str}...")
        
        content = fetch_events_for_date(single_date)
        if content is None:
            print(f"  - Failed to fetch data for {date_str}")
            continue
        
        events = parse_events(content, single_date)
        if events:
            print(f"  - Found {len(events)} matching events")
            all_events.extend(events)
        else:
            print("  - No matching events found")
    
    # Save results to JSON
    output_file = "solar_events_20220502_20240831.json"
    with open(output_file, 'w') as f:
        json.dump(all_events, f, indent=2)
    
    print(f"\nSaved {len(all_events)} events to {output_file}")

if __name__ == "__main__":
    main()