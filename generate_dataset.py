"""
Guardian AI — Dataset Generator
=================================

Uses the Gemini API to generate a labeled CSV dataset for fine-tuning
the MuRIL safety classifier.

Output format:
    text,label
    "someone is following me please help",1
    "let's grab some coffee today",0

Usage:
    1. Insert your Gemini API key below
    2. Run: python generate_dataset.py
    3. Data is saved incrementally to safety_dataset.csv

Each batch = 50 rows (25 SAFE + 25 EMERGENCY).
Default: 10 batches = ~500 rows total.
"""

import os
import csv
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# ==================================================================
#  CONFIGURATION — Edit these values before running
# ==================================================================

# API key loaded from .env file
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# Create the Gemini client (new SDK style)
client = genai.Client(api_key=API_KEY)

# ------------------------------------------------------------------
# LANGUAGE SELECTOR
# Change this value to generate data in a different language:
#   1 = English
#   2 = Hindi (Devanagari script)
#   3 = Hinglish (Hindi words written in English letters)
#   4 = Bengali
#   5 = Tamil
# ------------------------------------------------------------------
LANGUAGE = 2

# Output CSV — filename changes automatically based on language chosen
_LANGUAGE_NAMES = {1: "english", 2: "hindi", 3: "hinglish", 4: "bengali", 5: "tamil"}
_lang_name = _LANGUAGE_NAMES.get(LANGUAGE, f"lang{LANGUAGE}")
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"safety_dataset_{_lang_name}.csv"
)

# Number of batches to run (each batch = 50 rows)
TOTAL_BATCHES = 10  # 10 x 50 = 500 rows total

# Rows per batch (keep at 50 to stay within Gemini's response limits)
BATCH_SIZE = 50     # 25 SAFE + 25 EMERGENCY

# Seconds to wait between API calls (60s keeps you within free tier limits)
DELAY_SECONDS = 35


# ==================================================================
#  PROMPTS — one per language
# ==================================================================

# Shared role/purpose header injected into every prompt
_ROLE_HEADER = """
[ROLE & PURPOSE]
You are helping build a women's safety AI called Guardian AI.
It runs passively on a woman's phone and listens to ambient audio.
If it detects a dangerous situation, it silently alerts emergency contacts.
The model needs to classify speech into two categories:
  - SAFE (label=0): normal daily conversation — no danger
  - EMERGENCY (label=1): a woman in distress, being threatened, stalked,
    harassed, assaulted, or in immediate danger

Realistic context: audio is captured from real environments — cafes, streets,
homes, auto-rickshaws, public transport. Speech can be fragmented, whispered,
or show fear even without explicit distress words. A woman may try to secretly
communicate her danger without the attacker noticing.
"""

# Shared output format instruction appended to every prompt
_OUTPUT_FORMAT = """
Output ONLY this JSON, with no extra text, markdown, or explanation:
{
  "data": [
    {"text": "...", "label": 0},
    {"text": "...", "label": 1}
  ]
}
"""

# ------------------------------------------------------------------
# Language-specific generation instructions
# ------------------------------------------------------------------

_PROMPTS = {

    # ── 1. ENGLISH ──────────────────────────────────────────────
    1: _ROLE_HEADER + """
[GENERATION INSTRUCTIONS]
Generate exactly 25 SAFE and 25 EMERGENCY sentences IN ENGLISH.

Variety requirements:
  - Length mix: Short (3-6 words), medium (7-15 words), long (16-30 words)
  - EMERGENCY types: direct cries for help, coded/veiled distress,
    stalking, threats, captivity, fear without explicit words,
    secretly signalling danger to someone else
  - SAFE types: casual chat, food/movies/weather, scary topics in safe
    context (e.g. 'that horror film was terrifying' -> SAFE)

Important:
  - Do NOT repeat sentences
  - Realistic spoken English, not formal text
  - Do NOT include quotation marks inside text values
""" + _OUTPUT_FORMAT,

    # ── 2. HINDI (Devanagari script) ─────────────────────────────
    2: _ROLE_HEADER + """
[GENERATION INSTRUCTIONS]
Generate exactly 25 SAFE and 25 EMERGENCY sentences IN HINDI (Devanagari script).
Example: "मुझे बचाओ" (EMERGENCY), "आज खाना बहुत अच्छा था" (SAFE)

Variety requirements:
  - Length mix: Short (3-6 words), medium (7-15 words), long (16-30 words)
  - EMERGENCY types:
      * Direct cries: "बचाओ", "मुझे जाने दो", "मत छुओ मुझे"
      * Coded distress: nervously asking to "step outside for a second"
      * Stalking: "वो मेरे पीछे आ रहा है"
      * Threats: "उसने धमकी दी कि अगर बोली तो..."
      * Captivity: "दरवाज़ा बंद है, मैं निकल नहीं सकती"
      * Fear: "मुझे यहाँ डर लग रहा है"
  - SAFE types:
      * Casual chat: "आज मौसम बहुत अच्छा है"
      * Food/plans: "रात को खाने में क्या बनाएं?"
      * Scary topic in safe context: "वो horror film बहुत डरावनी थी" -> SAFE

Important:
  - All sentences must be in Devanagari (Hindi) script
  - Realistic spoken Hindi, not textbook formal language
  - Do NOT include quotation marks inside text values
""" + _OUTPUT_FORMAT,

    # ── 3. HINGLISH (Hindi words in English letters) ─────────────
    3: _ROLE_HEADER + """
[GENERATION INSTRUCTIONS]
Generate exactly 25 SAFE and 25 EMERGENCY sentences IN HINGLISH.
Hinglish = casual mix of Hindi + English written in English (Roman) script.
Example: "yaar kuch to karo please" (EMERGENCY), "aaj dinner mein kya banaye?" (SAFE)

Variety requirements:
  - Length mix: Short (3-6 words), medium (7-15 words), long (16-30 words)
  - EMERGENCY types:
      * Direct: "bachao mujhe please", "chhod do mujhe", "mat karo yeh"
      * Coded: "bas ek second bahar jaati hoon" (said nervously)
      * Stalking: "woh mera peecha kar raha hai"
      * Fear: "mujhe dar lag raha hai yahan"
      * Captivity: "band kar diya hai andar, nikl nahi sakti"
      * Secretly signalling: "call karna meri didi ko, code hai 'morning chai'"
  - SAFE types:
      * "aaj movie dekhne chalte hain"
      * "kal ka plan kya hai?"
      * "woh horror film toh bahut scary thi yaar, but maza aaya" -> SAFE

Important:
  - ALL text must be in Roman (English) script — no Devanagari
  - Natural street-style Hinglish, like WhatsApp messages spoken aloud
  - Contractions and incomplete sentences are fine
  - Do NOT include quotation marks inside text values
""" + _OUTPUT_FORMAT,

    # ── 4. BENGALI ───────────────────────────────────────────────
    4: _ROLE_HEADER + """
[GENERATION INSTRUCTIONS]
Generate exactly 25 SAFE and 25 EMERGENCY sentences IN BENGALI (Bengali script).
Example: "আমাকে বাঁচাও" (EMERGENCY), "আজকে রান্না খুব ভালো হয়েছে" (SAFE)

Variety requirements:
  - Length mix: Short (3-6 words), medium (7-15 words), long (16-30 words)
  - EMERGENCY: direct distress, stalking, threats, captivity, coded danger signals
  - SAFE: casual chat, food, weather, plans, scary topics in safe context

Important:
  - All text must be in Bengali script
  - Natural spoken Bengali, not formal/written style
  - Do NOT include quotation marks inside text values
""" + _OUTPUT_FORMAT,

    # ── 5. TAMIL ─────────────────────────────────────────────────
    5: _ROLE_HEADER + """
[GENERATION INSTRUCTIONS]
Generate exactly 25 SAFE and 25 EMERGENCY sentences IN TAMIL (Tamil script).
Example: "என்னை காப்பாற்றுங்கள்" (EMERGENCY), "இன்று காலை சாப்பாடு நல்லா இருந்தது" (SAFE)

Variety requirements:
  - Length mix: Short (3-6 words), medium (7-15 words), long (16-30 words)
  - EMERGENCY: direct distress, stalking, threats, captivity, coded danger signals
  - SAFE: casual chat, food, weather, plans, scary topics in safe context

Important:
  - All text must be in Tamil script
  - Natural spoken Tamil, not formal/written style
  - Do NOT include quotation marks inside text values
""" + _OUTPUT_FORMAT,
}

# Select the active prompt based on LANGUAGE variable
if LANGUAGE not in _PROMPTS:
    raise ValueError(f"LANGUAGE={LANGUAGE} is not supported. Choose from: {list(_PROMPTS.keys())}")

ACTIVE_PROMPT = _PROMPTS[LANGUAGE]



# ══════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_and_filter(raw_data: list, seen_sentences: set) -> list:
    """
    Validate each row from Gemini's response.
    Rejects rows that:
      - Are missing 'text' or 'label' keys
      - Have an empty or non-string text
      - Have a label that is not exactly 0 or 1
      - Are duplicate sentences (already seen in a previous batch)
    Returns a list of clean, unique rows.
    """
    valid_rows = []
    for item in raw_data:
        # Check required keys exist
        if "text" not in item or "label" not in item:
            print(f"  ⚠  Rejected (missing keys): {item}")
            continue

        text = item["text"]
        label = item["label"]

        # Text must be a non-empty string
        if not isinstance(text, str) or not text.strip():
            print(f"  ⚠  Rejected (empty/invalid text): {item}")
            continue

        # Label must be exactly 0 or 1
        if label not in (0, 1):
            print(f"  ⚠  Rejected (invalid label={label}): {text[:40]}")
            continue

        # Strip leading/trailing whitespace
        text = text.strip()

        # Deduplication check
        text_lower = text.lower()
        if text_lower in seen_sentences:
            print(f"  ⚠  Rejected (duplicate): {text[:40]}")
            continue

        seen_sentences.add(text_lower)
        valid_rows.append({"text": text, "label": label})

    return valid_rows


# ══════════════════════════════════════════════════════════════════
#  CSV HELPERS
# ══════════════════════════════════════════════════════════════════

def init_csv(filepath: str):
    """Create the CSV file with headers if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
        print(f"  [NEW] Created dataset file: {filepath}")
    else:
        print(f"  [CSV] Appending to existing file: {filepath}")


def append_to_csv(filepath: str, rows: list):
    """Append validated rows to the CSV file."""
    with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writerows(rows)


def count_existing_rows(filepath: str) -> tuple[int, set]:
    """
    Count rows and collect all existing sentences (for deduplication).
    Returns (row_count, seen_sentences_set).
    """
    if not os.path.exists(filepath):
        return 0, set()
    seen = set()
    count = 0
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(row["text"].lower())
            count += 1
    return count, seen


# ==================================================================
#  GEMINI CALL
# ==================================================================

def call_gemini(prompt: str) -> list:
    """
    Call the Gemini API and parse the JSON response.
    Returns a list of raw data items, or an empty list on failure.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=4096,
            ),
        )
        raw_text = response.text.strip()

        # Gemini sometimes wraps output in ```json ... ``` fences - strip them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        return parsed.get("data", [])

    except json.JSONDecodeError as e:
        print(f"  XX JSON parse error: {e}")
        print(f"     Raw response (first 300 chars): {response.text[:300]}")
        return []
    except Exception as e:
        print(f"  XX API call failed: {e}")
        return []


# ==================================================================
#  MAIN
# ==================================================================

def main():
    print()
    print(f"  +" + "=" * 54 + "+")
    print(f"  |   Guardian AI -- Fine-Tuning Dataset Generator       |")
    print(f"  +" + "=" * 54 + "+")
    print()
    print(f"  Language : {_lang_name.upper()} (LANGUAGE={LANGUAGE})")
    print(f"  Output   : {OUTPUT_FILE}")
    print(f"  [OK] Gemini client ready (gemini-2.5-flash, temperature=1.0)")

    # ── Load existing data ────────────────────────────────────────
    init_csv(OUTPUT_FILE)
    existing_count, seen_sentences = count_existing_rows(OUTPUT_FILE)
    print(f"  Existing rows : {existing_count}")
    print(f"  Target        : {TOTAL_BATCHES} batches x {BATCH_SIZE} rows = ~{TOTAL_BATCHES * BATCH_SIZE} new rows")
    print(f"  Delay         : {DELAY_SECONDS}s between batches")
    print()

    # ── Run batches ───────────────────────────────────────────────
    total_saved = 0

    for batch_num in range(1, TOTAL_BATCHES + 1):
        print(f"  -- Batch {batch_num}/{TOTAL_BATCHES} {'-' * 40}")

        # Call Gemini
        raw_data = call_gemini(ACTIVE_PROMPT)

        if not raw_data:
            print(f"  [SKIP] Batch {batch_num} returned no data.")
        else:
            # Validate and deduplicate
            valid_rows = validate_and_filter(raw_data, seen_sentences)
            safe_count  = sum(1 for r in valid_rows if r["label"] == 0)
            emrg_count  = sum(1 for r in valid_rows if r["label"] == 1)

            # Save incrementally
            if valid_rows:
                append_to_csv(OUTPUT_FILE, valid_rows)
                total_saved += len(valid_rows)
                print(f"  [OK]   Saved {len(valid_rows)} rows "
                      f"(SAFE={safe_count}, EMERGENCY={emrg_count}) "
                      f"| Total so far: {existing_count + total_saved}")
            else:
                print(f"  [WARN] No valid rows after validation.")

        # Wait between batches (skip wait after the last batch)
        if batch_num < TOTAL_BATCHES:
            print(f"  [WAIT] {DELAY_SECONDS}s before next batch...")
            time.sleep(DELAY_SECONDS)

    # ── Final summary ─────────────────────────────────────────────
    final_count, _ = count_existing_rows(OUTPUT_FILE)
    print()
    print(f"  {'=' * 55}")
    print(f"  DONE! Dataset generation complete.")
    print(f"  File    : {OUTPUT_FILE}")
    print(f"  Total   : {final_count} rows")
    print(f"  Added   : {total_saved} rows this run")
    print(f"  {'=' * 55}")
    print()
    print("  Next step: fine-tune MuRIL using this dataset.")
    print("  Run the training script on Google Colab (T4 GPU recommended).")


if __name__ == "__main__":
    main()
