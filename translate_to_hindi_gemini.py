#!/usr/bin/env python3
"""
translate_to_hindi_gemini.py — Step 2 of 2
Translates all 12,771 questions in subjects_fixed.json to Hindi using
Gemini 1.5 Flash, with RAG context retrieved from the pre-built ChromaDB.

Prerequisites:
    - Run build_rag_db.py first (builds ./ncert_chroma_db/)
    - subjects_fixed.json in the same directory

Usage:
    GOOGLE_API_KEY=your_key python3 translate_to_hindi_gemini.py

Features:
    - Checkpoint/resume: safe to interrupt and restart at any time
    - Chapter-scoped RAG: retrieves only from the relevant chapter's vectors
    - Per-batch save: writes output after every 20 questions
    - Garbage character detection: flags bad translations for review
    - Auto-split: if a batch response is truncated, splits and retries halves
    - Failed question log: nothing is silently lost
    - Exponential backoff on rate limit (429) errors
"""

import json
import os
import re
import subprocess
import time
import sys
import copy
from pathlib import Path
from collections import defaultdict

from google import genai
from google.genai import types

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

GOOGLE_API_KEY   = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME       = "gemini-1.5-flash"
EMBEDDING_MODEL  = "models/gemini-embedding-001"

BATCH_SIZE       = 20          # Questions per Gemini call
MAX_RETRIES      = 3           # Retries per batch before marking as failed
SLEEP_BETWEEN    = 0.6         # Seconds between API calls (avoids RPM limit)

INPUT_FILE       = "subjects_fixed.json"
OUTPUT_FILE      = "subjects_hindi.json"
CHECKPOINT_FILE  = "translation_checkpoint.json"
FAILED_FILE      = "failed_questions.json"   # Questions that couldn't be translated
REVIEW_FILE      = "review_question_ids.json"  # Translations with garbage chars
CHROMA_DB_PATH   = "./ncert_chroma_db"        # Built by build_rag_db.py

RAG_K            = 4    # Number of chunks to retrieve per batch
RAG_SCORE_THRESH = 0.4  # Minimum cosine similarity to include a chunk

# Gemini 1.5 Flash pricing (as of 2025, prompts > 128k tokens)
# Input:  $0.075 / 1M tokens  (≤128k context)
# Output: $0.30  / 1M tokens
# Using conservative estimate; actual may be slightly lower for shorter contexts
COST_INPUT_PER_M  = 0.075
COST_OUTPUT_PER_M = 0.30

# ─── Garbage Detection ────────────────────────────────────────────────────────

_ALLOWED_PATTERN = re.compile(
    r'^['
    r'\u0900-\u097F'       # Devanagari
    r'\uA8E0-\uA8FF'       # Devanagari Extended
    r'\u0020-\u007E'       # Basic Latin (ASCII)
    r'\u2000-\u206F'       # General punctuation
    r'\u00B0\u00B2\u00B3\u00B5\u00D7\u00F7\u00B1'  # °²³µ×÷±
    r'\u2190-\u21FF'       # Arrows
    r'\u2260-\u22FF'       # Math operators
    r'\u2022\u2013\u2014\u2026'  # •–—…
    r'\s'
    r']*$',
    re.DOTALL
)


def has_garbage_characters(text: str) -> bool:
    if not text:
        return False
    return _ALLOWED_PATTERN.match(text) is None


# ─── Chapter-to-PDF Chapter Name Mapping ─────────────────────────────────────
# Maps every chapter name variant found in subjects_fixed.json to the
# canonical chapter name stored in ChromaDB metadata (PDF_METADATA in
# build_rag_db.py). Used for filtered RAG retrieval.

CHAPTER_CANONICAL = {
    # AR short names (Class 11)
    "Chapter 1":  "The Living World",
    "Chapter 2":  "Biological Classification",
    "Chapter 3":  "Plant Kingdom",
    "Chapter 4":  "Animal Kingdom",
    "Chapter 5":  "Morphology of Flowering Plants",
    "Chapter 6":  "Anatomy of Flowering Plants",
    "Chapter 7":  "Structural Organisation in Animals",
    "Chapter 8":  "Cell The Unit of Life",
    "Chapter 9":  "Biomolecules",
    "Chapter 10": "Cell Cycle and Cell Division",
    "Topic 1":    "Photosynthesis in Higher Plants",  # AR mislabelled

    # Class 11 full names
    "Chapter 1- The Living World":                       "The Living World",
    "Chapter 2- Biological classification":              "Biological Classification",
    "Chapter 3- Plant kingdom":                          "Plant Kingdom",
    "Chapter 4- Animal Kingdom":                         "Animal Kingdom",
    "Chapter 5- Morphology of Flowering Plants":         "Morphology of Flowering Plants",
    "Chapter 6- ANATOMY OF FLOWERING PLANTS":            "Anatomy of Flowering Plants",
    "Chapter 7- STRUCTURAL ORGANISATION IN ANIMALS":     "Structural Organisation in Animals",
    "Chapter 8- Cell: The Unit of Life":                 "Cell The Unit of Life",
    "Chapter 8- Cell_The Unit of Life":                  "Cell The Unit of Life",
    "Chapter 9- Biomolecules":                           "Biomolecules",
    "Chapter 10- CELL CYCLE AND CELL DIVISION":          "Cell Cycle and Cell Division",
    "Chapter 11- PHOTOSYNTHESIS IN HIGHER PLANTS":       "Photosynthesis in Higher Plants",
    "Chapter 12- Respiration in plants":                 "Respiration in Plants",
    "Chapter 12- RESPIRATION IN PLANTS":                 "Respiration in Plants",
    "Chapter 13- Plant growth and development":          "Plant Growth and Development",
    "Chapter 13- PLANT GROWTH AND DEVELOPMENT":          "Plant Growth and Development",
    "Chapter 14- Breathing and exchange of gases":       "Breathing and Exchange of Gases",
    "Chapter 14- BREATHING N GASEOUS EXCHANGE":          "Breathing and Exchange of Gases",
    "Chapter 15- Body fluids and circulation":           "Body Fluids and Circulation",
    "Chapter 15- BODY FLUIDS N CIRCULATION":             "Body Fluids and Circulation",
    "Chapter 16- Excretory products and their elimination": "Excretory Products and Their Elimination",
    "Chapter 16- EXCRETORY PRODUCTS AND THEIR ELIMINATION": "Excretory Products and Their Elimination",
    "Chapter 17- Locomotion and movement":               "Locomotion and Movement",
    "Chapter 17- LOCOMOTION AND MOVEMENT":               "Locomotion and Movement",
    "Chapter 18- Neural control and coordination":       "Neural Control and Coordination",
    "Chapter 18- NEURAL CONTROL AND COORDINATION":       "Neural Control and Coordination",
    "Chapter 19- Chemical coordination and integration": "Chemical Coordination and Integration",
    "Chapter 19- CHEMICAL COORDINATION AND INTEGRATION": "Chemical Coordination and Integration",

    # Class 12 full names
    "Chapter 1- SEXUAL REPRODUCTION IN FLOWERING PLANTS": "Sexual Reproduction in Flowering Plants",
    "Chapter 2- Human reproduction":                     "Human Reproduction",
    "Chapter 2- HUMAN REPRODUCTION":                     "Human Reproduction",
    "Chapter 3- Reproduction health":                    "Reproductive Health",
    "Chapter 3- REPRODUCTIVE HEALTH":                    "Reproductive Health",
    "Chapter 4- principles of inheritance and variation": "Principles of Inheritance and Variation",
    "Chapter 4- PRINCIPLES OF INHERITANCE AND VARIATION": "Principles of Inheritance and Variation",
    "Chapter 5- Molecular basis of inheritance":         "Molecular Basis of Inheritance",
    "Chapter 5- Molecular Basis of Inheritance":         "Molecular Basis of Inheritance",
    "Chapter 6- Evolution":                              "Evolution",
    "Chapter 7- Human health and diseases":              "Human Health and Disease",
    "Chapter 7- Human health and disease":               "Human Health and Disease",
    "Chapter 8- Microbes in human welfare":              "Microbes in Human Welfare",
    "Chapter 9- Biotechnology: Principles and processes": "Biotechnology Principles and Processes",
    "Chapter 9- Biotechnology_Principles and processes": "Biotechnology Principles and Processes",
    "Chapter 10- Biotechnology and its application":     "Biotechnology and Its Applications",
    "Chapter 10- BIOTECHNOLOGY AND ITS APPLICATION":     "Biotechnology and Its Applications",
    "Chapter 11- Organisms and populations":             "Organisms and Populations",
    "Chapter 11- ORGANISMS AND POPULATIONS":             "Organisms and Populations",
    "Chapter 12- Ecosystem":                             "Ecosystem",
    "Chapter 12- ECOSYSTEM":                             "Ecosystem",
    "Chapter 13- Biodiversity and its conservation":     "Biodiversity and Conservation",
    "Chapter 13- BIODIVERSITY AND CONSERVATION":         "Biodiversity and Conservation",
}

CLASS_CHAPTER_OVERRIDE = {
    # AR "Chapter 1" in Class 12 must map to Sexual Reproduction, not Living World
    ("Class 12", "Chapter 1"): "Sexual Reproduction in Flowering Plants",
}


def get_canonical_chapter(class_name: str, chapter_name: str) -> str | None:
    """Return the canonical chapter name for RAG retrieval filtering."""
    key = (class_name, chapter_name)
    if key in CLASS_CHAPTER_OVERRIDE:
        return CLASS_CHAPTER_OVERRIDE[key]
    return CHAPTER_CANONICAL.get(chapter_name)


# ─── RAG Retrieval ────────────────────────────────────────────────────────────

def retrieve_context(vector_db, canonical_chapter: str | None, query: str) -> str:
    """
    Retrieve relevant Hindi NCERT text chunks for a batch of questions.

    - If canonical_chapter is known: filter to that chapter only (tighter context)
    - If unknown: search the full corpus (fallback for unit-integrated questions)

    Returns a single string of concatenated chunk texts, ready for the prompt.
    """
    if canonical_chapter:
        # Chapter-scoped retrieval: only return chunks from this chapter
        results = vector_db.similarity_search_with_score(
            query,
            k=RAG_K,
            filter={"chapter": canonical_chapter}
        )
    else:
        # Full corpus search
        results = vector_db.similarity_search_with_score(query, k=RAG_K)

    # Filter by minimum similarity score and concatenate
    context_chunks = [
        doc.page_content
        for doc, score in results
        if score >= RAG_SCORE_THRESH
    ]

    return "\n\n---\n\n".join(context_chunks) if context_chunks else ""


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_prompt(rag_context: str, questions_batch: list) -> str:
    """
    Build the full prompt for a batch of questions.
    Returns a single string (Gemini uses a single user turn for batch calls).
    """

    if rag_context.strip():
        vocab_section = f"""HINDI NCERT VOCABULARY REFERENCE:
Use the following official Hindi NCERT Biology text as your ONLY vocabulary source.
Every Hindi Biology term you use MUST appear in this reference text.

{rag_context[:12000]}

---"""
        vocab_rule = "1. Use ONLY Hindi Biology terms that appear in the NCERT vocabulary reference above. Do NOT invent Hindi terms."
    else:
        vocab_section = "(No NCERT reference available for this chapter — use standard official NEET Hindi Biology terminology.)"
        vocab_rule = "1. Use standard official Hindi NCERT Biology terminology as used in NEET Hindi question papers."

    output_schema = """{
  "index": <integer — the index I provide>,
  "question_hindi": "<full Hindi translation of the question>",
  "options_hindi": ["<option A in Hindi>", "<option B in Hindi>", "<option C in Hindi>", "<option D in Hindi>"],
  "explanation_hindi": "<full Hindi translation of the explanation>"
}"""

    questions_block = ""
    for i, q in enumerate(questions_batch):
        questions_block += f"""
--- Question {i} ---
Question: {q.get('question', '')}
Options: {json.dumps(q.get('options', []), ensure_ascii=False)}
Explanation: {q.get('explanation', '')}
"""

    prompt = f"""You are translating NEET Biology questions from English to Hindi for Indian medical entrance exam students.

{vocab_section}

STRICT TRANSLATION RULES:
{vocab_rule}
2. Keep ALL of the following in English (do NOT translate):
   - Scientific/Latin species names (e.g., Mangifera indica, E. coli)
   - Chemical formulas and abbreviations: DNA, RNA, mRNA, tRNA, rRNA, ATP, ADP, NADPH, FADH₂, CO₂, O₂, H₂O
   - Technical abbreviations: PCR, rDNA, GMO, BOD, COD, ORS, ECG, EEG, AIDS, HIV
   - Measurement units: nm, μm, mm, ml, mg, kg, kDa
   - Numeric values, percentages, Roman numerals
   - Proper names of scientists, plasmids (pBR322, Ti), restriction enzymes
3. Assertion-Reason standard translations:
   "Assertion (A):" → "अभिकथन (A):"
   "Reason (R):" → "कारण (R):"
   Option A text pattern → "A और R दोनों सही हैं और R, A की सही व्याख्या है"
   Option B text pattern → "A और R दोनों सही हैं लेकिन R, A की सही व्याख्या नहीं है"
   Option C text pattern → "A सही है लेकिन R गलत है"
   Option D text pattern → "A गलत है लेकिन R सही है"
4. Statement-based questions: "Which of the following statements is/are correct?" → "निम्नलिखित में से कौन-सा/से कथन सही है/हैं?"
5. Match-the-column: "Column I" → "स्तंभ I", "Column II" → "स्तंभ II"
6. Preserve ALL formatting — newlines, bullet points, numbered lists
7. DO NOT translate or modify: question IDs, answer keys, image URLs, question_number fields

OUTPUT FORMAT — respond with ONLY a valid JSON array. No markdown, no backticks, no explanation text, no preamble.
Each element must follow exactly this schema:
{output_schema}

QUESTIONS TO TRANSLATE ({len(questions_batch)} total):
{questions_block}

Respond now with ONLY the JSON array:"""

    return prompt


# ─── API Call ─────────────────────────────────────────────────────────────────

def translate_batch(vector_db, chapter: str | None,
                    questions_batch: list, depth: int = 0) -> tuple[list, int, int]:
    """
    Translate a batch of questions using Gemini 1.5 Flash.

    - Retrieves chapter-scoped RAG context
    - Parses and validates the JSON response
    - Auto-splits the batch if the response is truncated
    - Returns (translations_list, input_tokens, output_tokens)
    """
    if not questions_batch:
        return [], 0, 0

    # Build a representative query from the batch (first question + chapter name)
    rag_query = f"{chapter or ''} {questions_batch[0].get('question', '')}"
    rag_context = retrieve_context(vector_db, chapter, rag_query)

    prompt = build_prompt(rag_context, questions_batch)

    for attempt in range(MAX_RETRIES):
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                )
            )

            # Extract usage metadata
            # usage = response.usage_metadata
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

            # Extract response text
            response_text = response.text.strip()

            # Strip markdown fences if present (shouldn't happen with JSON mode, but defensive)
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:])
                if response_text.rstrip().endswith("```"):
                    response_text = response_text.rstrip()[:-3].strip()

            # Parse JSON
            translations = json.loads(response_text)

            if not isinstance(translations, list):
                raise ValueError(f"Expected JSON array, got {type(translations)}")

            # Validate count matches — if fewer results than questions, batch was too large
            if len(translations) < len(questions_batch) and depth < 3:
                print(f"\n  Incomplete response ({len(translations)}/{len(questions_batch)}) — splitting batch...")
                mid = len(questions_batch) // 2
                res_a, in_a, out_a = translate_batch(vector_db, chapter,
                                                      questions_batch[:mid], depth + 1)
                res_b, in_b, out_b = translate_batch(vector_db, chapter,
                                                      questions_batch[mid:], depth + 1)
                # Re-index res_b
                offset = len(questions_batch[:mid])
                for item in res_b:
                    if 'index' in item:
                        item['index'] += offset
                return res_a + res_b, input_tokens + in_a + in_b, output_tokens + out_a + out_b

            return translations, input_tokens, output_tokens

        except json.JSONDecodeError as e:
            print(f"\n  JSON parse error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if hasattr(response, 'text'):
                print(f"  Response preview: {response.text[:300]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = 2 ** (attempt + 3)  # 8s, 16s, 32s
                print(f"\n  Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    return [], 0, 0


# ─── JSON Traversal ───────────────────────────────────────────────────────────

def collect_all_questions(data: dict) -> list:
    """
    Walk the 6-level JSON tree and return all questions with their full paths.

    Tree structure:
        subjects → classes → units → chapters → topics → segments → questions

    Returns list of (path_tuple, class_name, chapter_name, question_data).
    """
    questions = []
    for subject_name, subject_data in data.get('subjects', {}).items():
        if not isinstance(subject_data, dict) or 'classes' not in subject_data:
            continue
        for class_name, class_data in subject_data['classes'].items():
            if not isinstance(class_data, dict) or 'units' not in class_data:
                continue
            for unit_name, unit_data in class_data['units'].items():
                if not isinstance(unit_data, dict) or 'chapters' not in unit_data:
                    continue
                for chapter_name, chapter_data in unit_data['chapters'].items():
                    if not isinstance(chapter_data, dict) or 'topics' not in chapter_data:
                        continue
                    for topic_name, topic_data in chapter_data['topics'].items():
                        if not isinstance(topic_data, dict) or 'segments' not in topic_data:
                            continue
                        for segment_name, segment_data in topic_data['segments'].items():
                            if not isinstance(segment_data, dict) or 'questions' not in segment_data:
                                continue
                            for q_key, q_data in segment_data['questions'].items():
                                if not isinstance(q_data, dict):
                                    continue
                                path = (subject_name, class_name, unit_name,
                                        chapter_name, topic_name, segment_name, q_key)
                                questions.append((path, class_name, chapter_name, q_data))
    return questions


def navigate_to_question(output_data: dict, path: tuple) -> dict | None:
    """Navigate the output_data tree using a path tuple. Returns the question dict or None."""
    subject, cls, unit, chapter, topic, segment, q_key = path
    try:
        return (output_data['subjects']
                [subject]['classes'][cls]['units'][unit]
                ['chapters'][chapter]['topics'][topic]
                ['segments'][segment]['questions'][q_key])
    except KeyError:
        return None


def mark_skipped(output_data: dict, path: tuple, reason: str = "skipped"):
    """Write [SKIPPED] placeholders for a question that couldn't be translated."""
    q_ref = navigate_to_question(output_data, path)
    if q_ref is not None:
        q_ref['question_hindi']    = f"[SKIPPED: {reason}]"
        q_ref['options_hindi']     = ["[SKIPPED]"] * 4
        q_ref['explanation_hindi'] = f"[SKIPPED: {reason}]"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not GOOGLE_API_KEY:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        sys.exit(1)

    if not Path(CHROMA_DB_PATH).exists():
        print(f"ERROR: ChromaDB not found at '{CHROMA_DB_PATH}'")
        print("Run build_rag_db.py first to build the vector store.")
        sys.exit(1)

    if not Path(INPUT_FILE).exists():
        print(f"ERROR: '{INPUT_FILE}' not found in current directory.")
        sys.exit(1)

    # Initialise Gemini
    # client = genai.Client(api_key=GOOGLE_API_KEY)
    # model = client.GenerativeModel(MODEL_NAME)
    # print(f"Gemini model: {MODEL_NAME}")

    # Load ChromaDB
    print(f"Loading ChromaDB from '{CHROMA_DB_PATH}'...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    vector_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    print(f"ChromaDB loaded. Vectors: {vector_db._collection.count()}")

    # Load source JSON
    print(f"Loading '{INPUT_FILE}'...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect all questions
    print("Traversing JSON tree...")
    all_questions = collect_all_questions(data)
    total = len(all_questions)
    print(f"Total questions found: {total}")

    # Load checkpoint
    completed = set()
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            completed = set(checkpoint.get('completed_paths', []))
        print(f"Checkpoint loaded: {len(completed)} already translated, resuming...")

    # Load or initialise output JSON
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        print(f"Existing output file loaded.")
    else:
        output_data = copy.deepcopy(data)

    # Group remaining questions by (class_name, chapter_name) for batching
    chapter_groups: dict[tuple, list] = defaultdict(list)
    for path, class_name, chapter_name, q_data in all_questions:
        if "/".join(path) not in completed:
            chapter_groups[(class_name, chapter_name)].append((path, q_data))

    remaining = sum(len(v) for v in chapter_groups.values())
    print(f"\nQuestions remaining: {remaining}")

    # Tracking
    review_questions = []
    failed_questions = []
    total_input_tokens  = 0
    total_output_tokens = 0
    translated_count    = len(completed)

    # Git config (for GitHub Actions / VPS auto-commit)
    subprocess.run(["git", "config", "user.name", "medic-translate-bot"], check=False, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bot@medicneet.com"], check=False, capture_output=True)

    print(f"\n{'='*60}")
    print(f"Starting translation — {remaining} questions remaining")
    print(f"Model: {MODEL_NAME}  |  Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    for (class_name, chapter_name), chapter_questions in chapter_groups.items():
        if not chapter_questions:
            continue

        canonical_chapter = get_canonical_chapter(class_name, chapter_name)
        rag_scope = canonical_chapter if canonical_chapter else "FULL CORPUS"

        print(f"\n--- {class_name} | {chapter_name} | RAG: {rag_scope} ({len(chapter_questions)} questions) ---")

        total_batches = (len(chapter_questions) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_start in range(0, len(chapter_questions), BATCH_SIZE):
            batch       = chapter_questions[batch_start:batch_start + BATCH_SIZE]
            batch_q     = [q_data for _, q_data in batch]
            batch_paths = [path for path, _ in batch]
            batch_num   = batch_start // BATCH_SIZE + 1

            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} questions)...", end=" ", flush=True)

            try:
                translations, input_tok, output_tok = translate_batch(
                    vector_db, canonical_chapter, batch_q
                )
                total_input_tokens  += input_tok
                total_output_tokens += output_tok

                for i, path in enumerate(batch_paths):
                    if i >= len(translations):
                        print(f"\n  WARNING: No translation for index {i} — marking skipped")
                        mark_skipped(output_data, path, "missing from API response")
                        failed_questions.append({"path": "/".join(path), "reason": "missing from response"})
                        completed.add("/".join(path))
                        translated_count += 1
                        continue

                    t = translations[i]
                    q_ref = navigate_to_question(output_data, path)

                    if q_ref is None:
                        print(f"\n  WARNING: Path not found in output_data for {path}")
                        failed_questions.append({"path": "/".join(path), "reason": "path not found"})
                        completed.add("/".join(path))
                        translated_count += 1
                        continue

                    question_hindi    = t.get('question_hindi', '')
                    options_hindi     = t.get('options_hindi', [])
                    explanation_hindi = t.get('explanation_hindi', '')

                    # Validate options_hindi is a list with 4 elements
                    if not isinstance(options_hindi, list) or len(options_hindi) < 4:
                        print(f"\n  WARNING: Bad options_hindi format for question {i} — padding")
                        while len(options_hindi) < 4:
                            options_hindi.append("[MISSING]")

                    # Garbage character detection
                    all_hindi = question_hindi + " " + " ".join(str(o) for o in options_hindi) + " " + explanation_hindi
                    if has_garbage_characters(all_hindi):
                        question_hindi = "[FOR REVIEW] " + question_hindi
                        review_questions.append({
                            "id": q_ref.get('id', "/".join(path)),
                            "chapter": chapter_name,
                            "reason": "garbage characters in translation"
                        })

                    q_ref['question_hindi']    = question_hindi
                    q_ref['options_hindi']     = options_hindi
                    q_ref['explanation_hindi'] = explanation_hindi

                    completed.add("/".join(path))
                    translated_count += 1

                # Cost estimate (Gemini 1.5 Flash)
                in_cost  = (total_input_tokens  / 1_000_000) * COST_INPUT_PER_M
                out_cost = (total_output_tokens / 1_000_000) * COST_OUTPUT_PER_M
                total_cost = in_cost + out_cost

                print(f"OK  [{translated_count}/{total}]  Cost: ${total_cost:.3f}")

                # Save checkpoint + output after every batch
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({'completed_paths': list(completed)}, f)
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                # Git commit every 5 batches to reduce overhead
                if batch_num % 5 == 0 or batch_start + BATCH_SIZE >= len(chapter_questions):
                    subprocess.run(
                        ["git", "add", OUTPUT_FILE, CHECKPOINT_FILE],
                        check=False, capture_output=True
                    )
                    subprocess.run(
                        ["git", "commit", "-m",
                         f"[translate] {translated_count}/{total} questions done"],
                        check=False, capture_output=True
                    )
                    subprocess.run(["git", "push"], check=False, capture_output=True)

                time.sleep(SLEEP_BETWEEN)

            except Exception as e:
                print(f"\n  BATCH ERROR: {e}")
                print(f"  Marking {len(batch)} questions as failed and continuing...")
                for path, _ in batch:
                    path_str = "/".join(path)
                    if path_str not in completed:
                        mark_skipped(output_data, path, reason=f"batch error: {e}")
                        failed_questions.append({"path": path_str, "reason": str(e)})
                        completed.add(path_str)
                        translated_count += 1
                # Save state even on error
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({'completed_paths': list(completed)}, f)
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                subprocess.run(
                    ["git", "add", OUTPUT_FILE, CHECKPOINT_FILE],
                    check=False, capture_output=True
                )
                subprocess.run(
                    ["git", "commit", "-m",
                     f"[translate/error-recovery] {translated_count}/{total}"],
                    check=False, capture_output=True
                )
                subprocess.run(["git", "push"], check=False, capture_output=True)
                continue

    # ── Final saves ───────────────────────────────────────────────────────────

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint (done = no need to resume)
    if Path(CHECKPOINT_FILE).exists():
        os.remove(CHECKPOINT_FILE)

    # Save failed questions log
    with open(FAILED_FILE, 'w', encoding='utf-8') as f:
        json.dump({"total_failed": len(failed_questions), "questions": failed_questions}, f, indent=2)

    # Save review list
    with open(REVIEW_FILE, 'w', encoding='utf-8') as f:
        json.dump({"total_for_review": len(review_questions), "questions": review_questions}, f, indent=2)

    in_cost  = (total_input_tokens  / 1_000_000) * COST_INPUT_PER_M
    out_cost = (total_output_tokens / 1_000_000) * COST_OUTPUT_PER_M

    print(f"\n{'='*60}")
    print(f"TRANSLATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total translated  : {translated_count:,}")
    print(f"  Failed (logged)   : {len(failed_questions)}")
    print(f"  Flagged for review: {len(review_questions)}")
    print(f"  Input tokens      : {total_input_tokens:,}")
    print(f"  Output tokens     : {total_output_tokens:,}")
    print(f"  Estimated cost    : ${in_cost + out_cost:.2f}")
    print(f"  Output file       : {OUTPUT_FILE}")
    print(f"  Failed log        : {FAILED_FILE}")
    print(f"  Review log        : {REVIEW_FILE}")


if __name__ == "__main__":
    main()
