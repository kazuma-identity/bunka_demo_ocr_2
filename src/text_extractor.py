#!/usr/bin/env python3
"""
PDF Text Extractor using pdfplumber
Extracts text content and tables from PDF files
"""

import os
from pathlib import Path
import pdfplumber
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def extract_text_pdfplumber(pdf_path):
    """Extract text using pdfplumber (better for complex layouts)"""
    text_content = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()

                text_content.append({
                    'page': page_num,
                    'text': text.strip() if text else ''
                })

    except Exception as e:
        print(f"Error with pdfplumber: {e}")

    return text_content

def save_text_to_file(text_content, output_dir):
    """Save extracted text to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual pages
    for page_data in text_content:
        page_num = page_data['page']
        text = page_data['text']

        # Save page text
        page_file = output_path / f"page_{page_num:02d}.txt"
        with open(page_file, 'w', encoding='utf-8') as f:
            f.write(f"Page {page_num}\n")
            f.write("=" * 50 + "\n")
            f.write(text)
            f.write("\n\n")


    # Save combined text
    combined_file = output_path / "all_pages.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(f"PDF Text Extraction\n")
        f.write("=" * 70 + "\n\n")

        for page_data in text_content:
            f.write(f"Page {page_data['page']}\n")
            f.write("-" * 30 + "\n")
            f.write(page_data['text'])
            f.write("\n\n")

def main():
    """Main extraction process"""
    pdf_path = "data/image.pdf"
    output_dir = "text_output_1"

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    print(f"Extracting text from: {pdf_path}")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Extract using pdfplumber
    print("Extracting text...")
    pdfplumber_content = extract_text_pdfplumber(pdf_path)
    if pdfplumber_content:
        save_text_to_file(pdfplumber_content, output_dir)
        print(f"Extracted {len(pdfplumber_content)} pages")


    print(f"Text extraction completed! Check {output_dir}/")

if __name__ == "__main__":
    main()
