import json

# Load the input JSON
with open('solar_events_20220502_20240831.json', 'r') as f:
    data = json.load(f)

updated = []

for entry in data:
    begin = entry.get("begin", "")
    max_ = entry.get("max", "")
    end = entry.get("end", "")

    # Rule 1: Keep only first 4 characters of begin
    if len(begin) >= 4:
        entry["begin"] = begin[:4]

    # Rule 2: Prepend last character of max to end
    if max_ and end:
        entry["end"] = max_[-1] + end

    updated.append(entry)

# Save modified output
with open("solar_events_cleaned.json", "w") as f:
    json.dump(updated, f, indent=2)

print(f"Processed {len(updated)} events and saved to solar_events_cleaned.json")