"""
Guardian AI — Offline Safety Classifier (MuRIL)
=================================================

Uses Google's MuRIL (Multilingual Representations for Indian Languages)
for binary text classification: SAFE vs EMERGENCY.

WHAT IS MuRIL?
--------------
MuRIL is Google's BERT-based model specifically built for Indian languages.
- Pre-trained on 17 Indian languages + English
- Handles Hinglish (code-mixing Hindi + English) natively
- Understands: Hindi, Gujarati, Marathi, Tamil, Telugu, Bengali,
  Kannada, Malayalam, Punjabi, Urdu, and more
- Uses BERT architecture: 12 layers, 768 hidden dim, 512 token context
- Size: ~900 MB

WHY MuRIL OVER LONGFORMER?
---------------------------
- Longformer: English-only, 4096 tokens, ~570 MB
- MuRIL:      17 Indian languages + English, 512 tokens, ~900 MB
- Since Guardian AI is for Indian women's safety, MuRIL is the right choice
- The memory system handles context beyond 512 tokens by managing
  short-term buffer + long-term memory within the token limit

ARCHITECTURE:
  Text → Tokenizer → 512 token IDs → MuRIL Encoder → 768-dim embedding
                                                        ↓
                                       Classification Head (768 → 2)
                                                        ↓
                                        [SAFE probability, EMERGENCY probability]
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# MuRIL: Multilingual Representations for Indian Languages (by Google)
# "google/muril-base-cased" means:
#   - google     → created by Google Research India
#   - muril      → Multilingual Representations for Indian Languages
#   - base       → base size (12 layers, 768 hidden dim)
#   - cased      → treats "Help" and "help" differently (case-sensitive)
#                  Important for Indian scripts where casing matters
MODEL_NAME = "google/muril-base-cased"

# Where to save the model locally (offline use after first download)
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "fine-tuned-v1",
)

# Classification labels
# 0 = SAFE (normal conversation)
# 1 = EMERGENCY (woman in danger, needs help)
LABELS = {0: "SAFE", 1: "EMERGENCY"}
NUM_LABELS = len(LABELS)

# Max tokens per input (MuRIL supports up to 512)
# We use 480 to leave room for special tokens [CLS] and [SEP]
MAX_LENGTH = 480


# ══════════════════════════════════════════════════════════════════
#  CLASSIFIER CLASS
# ══════════════════════════════════════════════════════════════════
class SafetyClassifier:
    """
    Binary text classifier: SAFE vs EMERGENCY.

    Uses MuRIL for multilingual understanding — works with
    English, Hindi, Hinglish, Gujarati, Marathi, and more.

    Usage:
        classifier = SafetyClassifier()
        classifier.load()
        probability = classifier.classify("mujhe bachao koi help karo")
        # → 0.87 (87% probability of emergency)
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cpu")

    def load(self):
        """Load the tokenizer and model from local disk."""
        print(f"  Loading MuRIL from {self.model_dir} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )
        self.model.to(self.device)
        self.model.eval()
        print("  ✅ Model loaded and ready.\n")

    def classify(self, text: str) -> float:
        """
        Classify text and return the EMERGENCY probability.

        Parameters
        ----------
        text : str
            Text to classify (English, Hindi, Hinglish, Gujarati, etc.)

        Returns
        -------
        float
            Probability of EMERGENCY (0.0 = safe, 1.0 = emergency)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        emergency_prob = probabilities[0][1].item()

        return emergency_prob

    def classify_with_details(self, text: str) -> dict:
        """Classify with full details for debug display."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        token_count = (inputs["attention_mask"] == 1).sum().item()

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        safe_prob = probs[0][0].item()
        emergency_prob = probs[0][1].item()
        predicted = 1 if emergency_prob > safe_prob else 0

        return {
            "emergency_prob": emergency_prob,
            "safe_prob": safe_prob,
            "predicted_label": LABELS[predicted],
            "token_count": token_count,
        }

    @staticmethod
    def download(model_name: str = MODEL_NAME, save_dir: str = MODEL_DIR):
        """
        Download MuRIL and save locally. ONE-TIME internet needed.
        """
        print("=" * 60)
        print("  Downloading MuRIL Model")
        print("=" * 60)
        print(f"  Source:  {model_name}")
        print(f"  Target:  {save_dir}")
        print(f"  This may take a few minutes (~900 MB) …\n")

        print("  [1/2] Downloading tokenizer …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        print("  ✅ Tokenizer saved.")

        print("  [2/2] Downloading model …")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        model.save_pretrained(save_dir)
        print("  ✅ Model saved.")

        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)
        print(f"\n  Model Summary:")
        print(f"  {'─' * 40}")
        print(f"  Parameters:      {total_params:,}")
        print(f"  Size:            ~{model_size_mb:.0f} MB")
        print(f"  Context window:  512 tokens")
        print(f"  Languages:       17 Indian + English")
        print(f"  Classification:  SAFE / EMERGENCY")
        print(f"  Offline ready:   ✅")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════
#  MAIN — Download and test
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║    Guardian AI — MuRIL Safety Classifier Setup      ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    if not os.path.exists(MODEL_DIR):
        SafetyClassifier.download()
    else:
        print(f"  ℹ️  Model already exists at {MODEL_DIR}")
        print(f"      Delete this folder to re-download.\n")

    print("\n  Loading model for test …")
    classifier = SafetyClassifier()
    classifier.load()

    print("  ⚠️  Predictions are RANDOM — model hasn't been fine-tuned yet!\n")

    test_sentences = [
        # English
        "someone is following me I am scared",
        "the weather is really nice today",
        # Hinglish
        "yeh aadmi mujhe follow kar raha hai help karo",
        "aaj ka dinner bahut accha tha",
        # Hindi
        "मुझे बचाओ कोई यह आदमी पीछा कर रहा है",
        "आज मौसम बहुत अच्छा है",
    ]

    for sentence in test_sentences:
        result = classifier.classify_with_details(sentence)
        label_color = "\033[91m" if result["predicted_label"] == "EMERGENCY" else "\033[92m"
        print(f"  Input:  \"{sentence}\"")
        print(f"  Result: {label_color}{result['predicted_label']}\033[0m "
              f"(emergency={result['emergency_prob']:.1%}, "
              f"tokens={result['token_count']})")
        print()

    print("  ─" * 30)
    print("  ℹ️  Next step: Fine-tune on safety dataset.")
