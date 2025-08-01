import os
import json
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import html
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
    "From": "segaab120@gmail.com",  # Required by SEC API rules
}

def fetch_xbrl_xml(cik: str) -> str:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    for filing in data.get("filings", {}).get("recent", {}).get("accessionNumber", []):
        if "10-Q" in filing or "10-K" in filing:
            acc_number = filing.replace("-", "")
            base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_number}"
            return base_url
    return None

def clean_xbrl_value(value: str) -> str:
    """
    Remove XML/HTML tags and entities from XBRL content.
    """
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r'\s+', ' ', value).strip()

def extract_cleaned_data_from_xbrl(url: str) -> dict:
    index_url = url + "/index.json"
    index_res = requests.get(index_url, headers=HEADERS)
    index_res.raise_for_status()
    index_data = index_res.json()

    # Find the XBRL document (.htm or .xml)
    xbrl_file = next(
        (f for f in index_data['directory']['item'] if "htm" in f['name'] or "xml" in f['name']), None
    )
    if not xbrl_file:
        raise Exception("XBRL document not found")

    file_url = f"{url}/{xbrl_file['name']}"
    print(f"Downloading XBRL from: {file_url}")
    xbrl_res = requests.get(file_url, headers=HEADERS)
    xbrl_res.raise_for_status()

    soup = BeautifulSoup(xbrl_res.content, "lxml-xml")
    elements = soup.find_all(['ix:nonNumeric', 'ix:nonFraction'])

    result = []
    for el in elements:
        name = el.get("name")
        context = el.get("contextRef")
        raw_value = el.text
        cleaned_value = clean_xbrl_value(raw_value)
        if name and cleaned_value:
            result.append({
                "field": name,
                "value": cleaned_value,
                "context": context
            })

    return {
        "source_url": file_url,
        "fields_extracted": result
    }

def write_to_json(data: dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[✓] JSON written to {filename}")

def write_to_pdf(data: dict, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SEC Filing Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Source: {data['source_url']}\n\n")

    for item in data["fields_extracted"]:
        line = f"{item['field']}: {item['value']} (context: {item['context']})\n"
        pdf.multi_cell(0, 10, line)

    pdf.output(filename)
    print(f"[✓] PDF written to {filename}")

# -------------------- USAGE --------------------

if __name__ == "__main__":
    cik = "19617"  # JPMorgan Chase (CIK: 0000019617)

    try:
        filing_url = fetch_xbrl_xml(cik)
        if not filing_url:
            print("No XBRL filing found.")
            exit(1)

        parsed_data = extract_cleaned_data_from_xbrl(filing_url)
        write_to_json(parsed_data, "jpmorgan_filing.json")
        write_to_pdf(parsed_data, "jpmorgan_filing.pdf")

    except Exception as e:
        print(f"[✗] Error: {e}")
