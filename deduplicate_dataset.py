import json
import hashlib

# Input and output file paths
input_file = "Data/fine-tuning-dataset.jsonl"
output_file = "Data/deduplicated-dataset.jsonl"

print(f"Analyzing {input_file} for duplicates...")

# Lists to store entries and their hashes
entries = []
entry_hashes = set()
duplicates = 0
total_entries = 0

# Process the file line by line
with open(input_file, 'r', encoding='utf-8') as f:
    for line_number, line in enumerate(f, 1):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('//'):
            continue
            
        total_entries += 1
        
        # Try to parse the JSON
        try:
            entry = json.loads(line)
            
            # Create a hash based on the input-output pair content
            # This will help identify semantic duplicates even if they have minor differences in formatting
            content_key = f"{entry.get('input', '')}-{entry.get('output', '')}"
            content_hash = hashlib.md5(content_key.encode()).hexdigest()
            
            if content_hash not in entry_hashes:
                # This is a new unique entry
                entries.append(entry)
                entry_hashes.add(content_hash)
            else:
                duplicates += 1
                print(f"Duplicate found at line {line_number}: {entry.get('input', '')[:50]}...")
                
        except json.JSONDecodeError as e:
            print(f"Line {line_number}: Invalid JSON - {e}")

# Write the deduplicated data to the output file
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in entries:
        json_line = json.dumps(entry)
        f.write(json_line + '\n')

print(f"\nDeduplication complete!")
print(f"Total entries processed: {total_entries}")
print(f"Duplicates removed: {duplicates}")
print(f"Unique entries saved: {len(entries)}")
print(f"Deduplicated dataset saved to: {output_file}")
