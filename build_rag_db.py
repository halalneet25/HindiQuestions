#!/usr/bin/env python3
"""
build_rag_db.py — Step 1 of 2
OCR-ingests all Hindi NCERT Biology PDFs and builds a ChromaDB vector store.

Run this ONCE before running translate_to_hindi_gemini.py.

Why OCR instead of direct PDF text extraction:
    The Hindi NCERT PDFs use legacy Kruti Dev font encoding. Standard PDF
    libraries (pdfplumber, PyMuPDF, pdfminer) extract garbled Latin
    characters instead of Devanagari Unicode. OCR via Tesseract reads the
    rendered page image directly, bypassing the broken font mapping entirely.

Usage:
    1. Place all Hindi NCERT PDFs in ./hindi_pdfs/
    2. GOOGLE_API_KEY=your_key python3 build_rag_db.py

Output:
    ./ncert_chroma_db/   ← ChromaDB vector store (used by translation script)
    ./ocr_text_cache/    ← Raw OCR text per PDF (for debugging / inspection)
"""

import os
import sys
import json
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.schema import Document

# ─── Configuration ────────────────────────────────────────────────────────────

GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY")
PDF_DIR         = "./hindi_pdfs"       # Directory containing all NCERT Hindi PDFs
CHROMA_DB_PATH  = "./ncert_chroma_db"  # Output: ChromaDB vector store
OCR_CACHE_DIR   = "./ocr_text_cache"   # Output: raw OCR text per file (debug)
OCR_DPI         = 300                  # Higher = better OCR accuracy, slower
CHUNK_SIZE      = 800                  # Characters per chunk
CHUNK_OVERLAP   = 120                  # Overlap between chunks

# Tesseract language config: Hindi + English together
# 'hin+eng' ensures both scripts are recognised in the same page
TESS_LANG = "hin+eng"

# Maps each PDF filename to its chapter metadata for filtered retrieval
# Class 11: khbo101.pdf – khbo119.pdf
# Class 12: lhbo101.pdf – lhbo113.pdf
PDF_METADATA = {
    # ── Class 11 ──────────────────────────────────────────────────────────
    "khbo101.pdf": {"class": "Class 11", "chapter_num": 1,  "chapter": "The Living World"},
    "khbo102.pdf": {"class": "Class 11", "chapter_num": 2,  "chapter": "Biological Classification"},
    "khbo103.pdf": {"class": "Class 11", "chapter_num": 3,  "chapter": "Plant Kingdom"},
    "khbo104.pdf": {"class": "Class 11", "chapter_num": 4,  "chapter": "Animal Kingdom"},
    "khbo105.pdf": {"class": "Class 11", "chapter_num": 5,  "chapter": "Morphology of Flowering Plants"},
    "khbo106.pdf": {"class": "Class 11", "chapter_num": 6,  "chapter": "Anatomy of Flowering Plants"},
    "khbo107.pdf": {"class": "Class 11", "chapter_num": 7,  "chapter": "Structural Organisation in Animals"},
    "khbo108.pdf": {"class": "Class 11", "chapter_num": 8,  "chapter": "Cell The Unit of Life"},
    "khbo109.pdf": {"class": "Class 11", "chapter_num": 9,  "chapter": "Biomolecules"},
    "khbo110.pdf": {"class": "Class 11", "chapter_num": 10, "chapter": "Cell Cycle and Cell Division"},
    "khbo111.pdf": {"class": "Class 11", "chapter_num": 11, "chapter": "Photosynthesis in Higher Plants"},
    "khbo112.pdf": {"class": "Class 11", "chapter_num": 12, "chapter": "Respiration in Plants"},
    "khbo113.pdf": {"class": "Class 11", "chapter_num": 13, "chapter": "Plant Growth and Development"},
    "khbo114.pdf": {"class": "Class 11", "chapter_num": 14, "chapter": "Breathing and Exchange of Gases"},
    "khbo115.pdf": {"class": "Class 11", "chapter_num": 15, "chapter": "Body Fluids and Circulation"},
    "khbo116.pdf": {"class": "Class 11", "chapter_num": 16, "chapter": "Excretory Products and Their Elimination"},
    "khbo117.pdf": {"class": "Class 11", "chapter_num": 17, "chapter": "Locomotion and Movement"},
    "khbo118.pdf": {"class": "Class 11", "chapter_num": 18, "chapter": "Neural Control and Coordination"},
    "khbo119.pdf": {"class": "Class 11", "chapter_num": 19, "chapter": "Chemical Coordination and Integration"},
    # ── Class 12 ──────────────────────────────────────────────────────────
    "lhbo101.pdf": {"class": "Class 12", "chapter_num": 1,  "chapter": "Sexual Reproduction in Flowering Plants"},
    "lhbo102.pdf": {"class": "Class 12", "chapter_num": 2,  "chapter": "Human Reproduction"},
    "lhbo103.pdf": {"class": "Class 12", "chapter_num": 3,  "chapter": "Reproductive Health"},
    "lhbo104.pdf": {"class": "Class 12", "chapter_num": 4,  "chapter": "Principles of Inheritance and Variation"},
    "lhbo105.pdf": {"class": "Class 12", "chapter_num": 5,  "chapter": "Molecular Basis of Inheritance"},
    "lhbo106.pdf": {"class": "Class 12", "chapter_num": 6,  "chapter": "Evolution"},
    "lhbo107.pdf": {"class": "Class 12", "chapter_num": 7,  "chapter": "Human Health and Disease"},
    "lhbo108.pdf": {"class": "Class 12", "chapter_num": 8,  "chapter": "Microbes in Human Welfare"},
    "lhbo109.pdf": {"class": "Class 12", "chapter_num": 9,  "chapter": "Biotechnology Principles and Processes"},
    "lhbo110.pdf": {"class": "Class 12", "chapter_num": 10, "chapter": "Biotechnology and Its Applications"},
    "lhbo111.pdf": {"class": "Class 12", "chapter_num": 11, "chapter": "Organisms and Populations"},
    "lhbo112.pdf": {"class": "Class 12", "chapter_num": 12, "chapter": "Ecosystem"},
    "lhbo113.pdf": {"class": "Class 12", "chapter_num": 13, "chapter": "Biodiversity and Conservation"},
}


# ─── OCR Functions ────────────────────────────────────────────────────────────

def ocr_pdf(pdf_path: str, cache_dir: str) -> str:
    """
    Convert a PDF to images and OCR each page.
    Caches the result as a .txt file to avoid re-running OCR on restarts.

    Returns the full extracted text string.
    """
    pdf_name = Path(pdf_path).stem
    cache_file = Path(cache_dir) / f"{pdf_name}.txt"

    # Use cached OCR if available
    if cache_file.exists():
        print(f"  [CACHE HIT] {Path(pdf_path).name}")
        return cache_file.read_text(encoding="utf-8")

    print(f"  [OCR] {Path(pdf_path).name} — converting pages to images @ {OCR_DPI} DPI...")
    images = convert_from_path(pdf_path, dpi=OCR_DPI)

    full_text = ""
    for i, img in enumerate(tqdm(images, desc=f"    Pages", leave=False)):
        page_text = pytesseract.image_to_string(img, lang=TESS_LANG)
        full_text += page_text + "\n\n"

    # Validate OCR output — must have meaningful Devanagari content
    sample = full_text[:5000]
    deva_count = sum(1 for c in sample if '\u0900' <= c <= '\u097F')
    ratio = deva_count / max(len(sample), 1)
    if ratio < 0.05:
        print(f"  WARNING: {Path(pdf_path).name} has only {ratio:.1%} Devanagari after OCR.")
        print(f"  Check: is tesseract-ocr-hin installed? Run: sudo apt-get install tesseract-ocr-hin")

    # Save to cache
    cache_file.write_text(full_text, encoding="utf-8")
    return full_text


def validate_tesseract():
    """Confirm Tesseract is installed with Hindi language pack."""
    try:
        langs = pytesseract.get_languages(config='')
        if 'hin' not in langs:
            print("ERROR: Hindi language pack not found in Tesseract.")
            print("Install it: sudo apt-get install tesseract-ocr-hin")
            sys.exit(1)
        print(f"Tesseract OK. Available languages include Hindi.")
    except Exception as e:
        print(f"ERROR: Tesseract not found. Install it: sudo apt-get install tesseract-ocr")
        print(f"Details: {e}")
        sys.exit(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not GOOGLE_API_KEY:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        sys.exit(1)

    validate_tesseract()

    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory '{PDF_DIR}' not found.")
        print("Create it and place all Hindi NCERT PDFs inside.")
        sys.exit(1)

    Path(OCR_CACHE_DIR).mkdir(exist_ok=True)

    # Discover PDFs
    found_pdfs = sorted([f for f in pdf_dir.glob("*.pdf")])
    if not found_pdfs:
        print(f"ERROR: No PDFs found in '{PDF_DIR}'")
        sys.exit(1)

    print(f"\nFound {len(found_pdfs)} PDFs in '{PDF_DIR}'")

    # Check for any unrecognised PDFs
    known_names = set(PDF_METADATA.keys())
    for pdf in found_pdfs:
        if pdf.name not in known_names:
            print(f"  WARNING: Unrecognised PDF '{pdf.name}' — will be ingested with generic metadata.")

    print(f"\n{'='*60}")
    print("STEP 1: OCR all PDFs")
    print(f"{'='*60}")

    # OCR all PDFs and build LangChain Document objects
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", " "]  # Include Devanagari sentence terminator
    )

    all_documents = []

    for pdf_path in found_pdfs:
        meta = PDF_METADATA.get(pdf_path.name, {
            "class": "Unknown",
            "chapter_num": 0,
            "chapter": pdf_path.stem
        })
        print(f"\n[{meta['class']}] Chapter {meta['chapter_num']}: {meta['chapter']}")

        raw_text = ocr_pdf(str(pdf_path), OCR_CACHE_DIR)

        if not raw_text.strip():
            print(f"  SKIP: Empty OCR output for {pdf_path.name}")
            continue

        # Split into chunks
        chunks = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{
                "source": pdf_path.name,
                "class": meta["class"],
                "chapter_num": meta["chapter_num"],
                "chapter": meta["chapter"],
            }]
        )
        all_documents.extend(chunks)
        print(f"  → {len(chunks)} chunks created from {len(raw_text):,} characters")

    print(f"\n{'='*60}")
    print(f"STEP 2: Build ChromaDB vector store")
    print(f"Total chunks to embed: {len(all_documents)}")
    print(f"{'='*60}")

    if not all_documents:
        print("ERROR: No documents to embed. Check OCR output.")
        sys.exit(1)

    # Build embeddings using Google's multilingual text-embedding-004
    # This model handles mixed Hindi/English text well
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Create ChromaDB — this will take a few minutes for 32 PDFs
    print("\nEmbedding chunks (this takes ~5-10 minutes for all 32 PDFs)...")
    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"\n✓ ChromaDB built at: {CHROMA_DB_PATH}")
    print(f"  Total vectors stored: {vector_db._collection.count()}")

    # Write a manifest so translate_to_hindi_gemini.py can verify the DB
    manifest = {
        "total_chunks": len(all_documents),
        "pdf_count": len(found_pdfs),
        "pdfs_ingested": [p.name for p in found_pdfs],
        "chroma_db_path": CHROMA_DB_PATH,
        "embedding_model": "models/text-embedding-004",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open("rag_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifest written to rag_manifest.json")
    print("\nAll done. You can now run translate_to_hindi_gemini.py")


if __name__ == "__main__":
    main()
