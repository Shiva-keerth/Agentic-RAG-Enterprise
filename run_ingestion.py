import os
from ingestion import DocumentIngestor

def main():
    print("Starting Healthcare Domain Data Ingestion...")
    ingestor = DocumentIngestor()
    
    healthcare_data_dir = "data/healthcare"
    
    # 1. Ingest Digital PDFs (PyMuPDF)
    pdf_files = ["john_doe_bloodwork.pdf", "jane_smith_mri.pdf"]
    for pdf in pdf_files:
        pdf_path = os.path.join(healthcare_data_dir, pdf)
        text = ingestor.extract_text_from_pdf(pdf_path)
        ingestor.process_and_store(text, source_name=pdf, domain="healthcare")
        
    # 2. Ingest Physical Image Scans (OpenCV & Tesseract)
    img_files = ["bob_williams_scan.png"]
    for img in img_files:
        img_path = os.path.join(healthcare_data_dir, img)
        text = ingestor.extract_text_from_image(img_path)
        print(f"\n--- TESSERACT OCR OUTPUT ---\n{text}\n----------------------------\n")
        ingestor.process_and_store(text, source_name=img, domain="healthcare")

    print("Phase 3 Ingestion Complete! Medical data is now permanently embedded in ChromaDB.")

if __name__ == "__main__":
    main()
