"""
Document Ingestion Engine — Phase 6
Supports both file-path-based ingestion and real-time file upload from Streamlit UI.
Stores page-level metadata for source citation support.
"""

import os
import fitz  # PyMuPDF
import cv2
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# NOTE FOR WINDOWS USERS:
# If you get a TesseractNotFoundError, you need to install Tesseract-OCR
# and uncomment the line below, updating the path to your installation.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DocumentIngestor:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        print("[Ingestor] Loading BERT Embedding Model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Split text into 1000-character chunks with 200 chars of overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        print("[Ingestor] Initializing ChromaDB Vector Store...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using PyMuPDF. Returns list of (page_num, text) tuples."""
        print(f"[Ingestor] Extracting text from PDF: {pdf_path}")
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    pages.append((page_num + 1, text))
            return pages
        except Exception as e:
            print(f"[Ingestor] Error reading PDF: {e}")
            return []

    def extract_text_from_pdf_bytes(self, file_bytes, filename):
        """Extract text from uploaded PDF bytes (for Streamlit file_uploader)."""
        print(f"[Ingestor] Extracting text from uploaded file: {filename}")
        pages = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    pages.append((page_num + 1, text))
            return pages
        except Exception as e:
            print(f"[Ingestor] Error reading uploaded PDF: {e}")
            return []

    def extract_text_from_image(self, image_path):
        """Extract text from an image using OpenCV and PyTesseract."""
        print(f"[Ingestor] Extracting text from Image: {image_path}")
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text
        except Exception as e:
            print(f"[Ingestor] Error reading Image: {e}")
            return ""

    def process_and_store(self, text, source_name, domain, page_num=None):
        """Chunk the text, embed it, and store in ChromaDB with metadata."""
        if not text.strip():
            return 0

        chunks = self.text_splitter.split_text(text)

        # Build metadata with page-level info for citations
        metadatas = []
        for i, _ in enumerate(chunks):
            meta = {
                "source": source_name,
                "domain": domain,
                "chunk_index": i
            }
            if page_num is not None:
                meta["page"] = page_num
            metadatas.append(meta)

        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        return len(chunks)

    def auto_classify_domain(self, text):
        """Auto-detect domain based on keyword frequency in the text."""
        text_lower = text.lower()
        
        healthcare_keywords = ["patient", "medical", "diagnosis", "health", "blood", "clinic", "symptoms", "treatment", "mri", "prescribed"]
        finance_keywords = ["revenue", "earnings", "finance", "fiscal", "margin", "stock", "income", "quarterly", "corporate", "sales"]
        
        health_score = sum(text_lower.count(kw) for kw in healthcare_keywords)
        finance_score = sum(text_lower.count(kw) for kw in finance_keywords)
        
        if finance_score > health_score:
            return "finance"
        return "healthcare" # Default fallback

    def ingest_uploaded_pdf(self, file_bytes, filename, domain):
        """
        Full pipeline for ingesting an uploaded PDF from the Streamlit UI.
        Returns dict with ingestion stats.
        """
        pages = self.extract_text_from_pdf_bytes(file_bytes, filename)
        if not pages:
            return {"success": False, "error": "Could not extract text from PDF.", "chunks": 0, "pages": 0}

        # Combine text from first few pages to classify
        if domain == "auto":
            sample_text = " ".join([text for _, text in pages[:3]])
            domain = self.auto_classify_domain(sample_text)
            print(f"[Ingestor] Auto-classified '{filename}' as: {domain}")

        total_chunks = 0
        for page_num, page_text in pages:
            chunks_added = self.process_and_store(page_text, filename, domain, page_num=page_num)
            total_chunks += chunks_added

        print(f"[Ingestor] Successfully ingested '{filename}': {len(pages)} pages, {total_chunks} chunks")
        return {
            "success": True,
            "filename": filename,
            "pages": len(pages),
            "chunks": total_chunks,
            "domain": domain
        }

    def ingest_pdf_from_path(self, pdf_path, domain):
        """Ingest a PDF from a file path (for setup scripts)."""
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            return {"success": False, "error": f"Could not extract text from {pdf_path}.", "chunks": 0, "pages": 0}

        filename = os.path.basename(pdf_path)
        total_chunks = 0
        for page_num, page_text in pages:
            chunks_added = self.process_and_store(page_text, filename, domain, page_num=page_num)
            total_chunks += chunks_added

        print(f"[Ingestor] Successfully ingested '{filename}': {len(pages)} pages, {total_chunks} chunks")
        return {
            "success": True,
            "filename": filename,
            "pages": len(pages),
            "chunks": total_chunks,
            "domain": domain
        }


if __name__ == "__main__":
    print("Ingestion engine is ready to be imported and used.")
