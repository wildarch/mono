#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from bs4 import BeautifulSoup

def extract_vocabulary_tables(html_content):
    """Extract vocabulary data from HSK HTML tables.
    
    Returns:
        List of tuples: (chinese, pinyin, english, level)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all vocabulary tables
    tables = soup.find_all('table')
    
    vocabulary = []
    for table in tables:
        tbody = table.find('tbody')
        if not tbody:
            continue
            
        for row in tbody.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 4:
                chinese = cells[0].get_text(strip=True)
                pinyin = cells[1].get_text(strip=True)
                english = cells[2].get_text(strip=True)
                level = cells[3].get_text(strip=True)
                
                # Only add if all fields have content
                if chinese and pinyin and english:
                    vocabulary.append((chinese, pinyin, english, level))
    
    return vocabulary

def write_vocabulary_csv(vocabulary, file):
    """Write vocabulary data to CSV format.
    
    Args:
        vocabulary: List of tuples (chinese, pinyin, english, level)
        file: File-like object to write to
    """
    writer = csv.writer(file)
    writer.writerow(['Chinese', 'Pinyin', 'English', 'Level'])
    writer.writerows(vocabulary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract HSK vocabulary from HTML and save as CSV')
    parser.add_argument('input', type=Path, help='Input HTML file path')
    parser.add_argument('-o', '--output', type=Path, help='Output CSV file path (default: input file with .csv extension)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File {args.input} does not exist")
        exit(1)
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        vocabulary = extract_vocabulary_tables(html_content)
        
        if args.output:
            # Write to CSV file
            with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
                write_vocabulary_csv(vocabulary, csvfile)
            print(f"Extracted {len(vocabulary)} vocabulary entries to {args.output}", file=sys.stderr)
        else:
            # Write to stdout
            write_vocabulary_csv(vocabulary, sys.stdout)
            
    except Exception as e:
        print(f"Error processing file: {e}")
        exit(1)