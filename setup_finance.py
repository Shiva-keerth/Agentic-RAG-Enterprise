import os
from fpdf import FPDF
from ingestion import DocumentIngestor

def generate_finance_pdfs():
    os.makedirs("data/finance", exist_ok=True)
    
    print("Generating Finance PDF 1 (Apple 10-K)...")
    pdf1 = FPDF()
    pdf1.add_page()
    pdf1.set_font("Arial", size=12)
    pdf1.cell(200, 10, txt="APPLE INC. - FORM 10-K EXCERPT (FISCAL 2025)", ln=1, align="C")
    pdf1.multi_cell(0, 10, txt="""
FINANCIAL PERFORMANCE SUMMARY:
Total net sales for the fiscal year 2025 were $390.5 billion, representing a 4% increase year-over-year.
Services revenue reached a new all-time high of $85.2 billion, driven by strong performance in App Store subscriptions and Apple Cloud services.
Gross margin for the year was 44.5%, an improvement from 43.3% in the previous year, primarily due to a favorable product mix.

RISK FACTORS:
The Company's business is highly competitive. If the company fails to successfully introduce new products or services, revenue may decline.
Supply chain disruptions in Southeast Asia remain a critical risk to hardware manufacturing timelines.
""")
    pdf1.output("data/finance/apple_10k_excerpt.pdf")

    print("Generating Finance PDF 2 (Tesla Earnings)...")
    pdf2 = FPDF()
    pdf2.add_page()
    pdf2.set_font("Arial", size=12)
    pdf2.cell(200, 10, txt="TESLA INC. - Q3 2026 EARNINGS REPORT", ln=1, align="C")
    pdf2.multi_cell(0, 10, txt="""
QUARTERLY HIGHLIGHTS:
Total automotive revenues achieved $22.4 billion in Q3 2026.
Energy generation and storage revenue increased by 50% year-over-year to $2.1 billion, largely due to Megapack deployments.
Operating margin was 8.2%, impacted by aggressive pricing strategies to increase market share in Europe and China.
Total vehicle deliveries for the quarter were 450,000 units.

FUTURE OUTLOOK:
Cybertruck production is scaling, with a target of 125,000 units annually by the end of 2027.
The company plans to invest $3 billion in expanding the Texas Gigafactory for next-generation vehicle platforms.
""")
    pdf2.output("data/finance/tesla_q3_earnings.pdf")
    print("Dummy finance data successfully generated in data/finance/")

def ingest_finance_data():
    print("Starting Finance Domain Data Ingestion...")
    ingestor = DocumentIngestor()
    finance_data_dir = "data/finance"
    
    pdf_files = ["apple_10k_excerpt.pdf", "tesla_q3_earnings.pdf"]
    for pdf in pdf_files:
        pdf_path = os.path.join(finance_data_dir, pdf)
        text = ingestor.extract_text_from_pdf(pdf_path)
        ingestor.process_and_store(text, source_name=pdf, domain="finance")
        
    print("Phase 4 Ingestion Complete! Financial data is embedded in ChromaDB.")

if __name__ == "__main__":
    generate_finance_pdfs()
    ingest_finance_data()
