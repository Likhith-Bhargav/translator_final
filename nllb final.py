"""
Direct Facebook NLLB-200 Distilled 600M implementation using local files only.

This module loads a locally available `facebook/nllb-200-distilled-600M` model
from a user-provided folder path. No external downloads or network calls.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import re

logger = logging.getLogger(__name__)

# --- Placeholder Protection Utilities ---
VAR_PATTERN = re.compile(r"\{v\d+\}")

def protect_placeholders(text):
    """Protects {vNNN} placeholders so the translation model won't corrupt them."""
    mapping = {}
    def _replacer(match):
        key = f"__PLACEHOLDER_v{match.group(0)[2:-1]}__"
        mapping[key] = match.group(0)
        return key
    out = VAR_PATTERN.sub(_replacer, text)
    return out, mapping

def restore_placeholders(text, mapping):
    """Restores protected placeholder tags to their original {vNNN}."""
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


class NLLBDirect:
    """Local NLLB translator wrapper (no network)."""

    def __init__(self, model_path: Optional[str] = None):
        # The user will place the model folder locally and provide its path.
        if model_path is None:
            # Safe default; user will override when running.
            self.model_path = Path(
                "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/facebook_nllb-200-distilled-600M"
            )
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"NLLB model not found at {self.model_path}. Please provide a valid local model path."
            )

        self._tokenizer = None
        self._model = None
        self._device = "cpu"

        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            logger.info(f"NLLB model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load NLLB model from {self.model_path}: {e}")
            raise

    def _map_lang(self, code: str) -> str:
        """Map common short codes to NLLB language tags.

        If the input already looks like an NLLB tag (e.g. eng_Latn), return as-is.
        """
        if "_" in code:
            return code
        m = {
            "en": "eng_Latn",
            "zh": "zho_Hans",
            "zh-cn": "zho_Hans",
            "zh-hans": "zho_Hans",
            "zh-tw": "zho_Hant",
            "zh-hant": "zho_Hant",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "es": "spa_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "hindi": "hin_Deva",
            "id": "ind_Latn",
            "vi": "vie_Latn",
            "ms": "zsm_Latn",
        }
        return m.get(code.lower(), code)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("NLLB model not properly loaded")

        src = self._map_lang(source_lang)
        tgt = self._map_lang(target_lang)

        try:
            self._tokenizer.src_lang = src
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            with torch.no_grad():
                # Calculate reasonable max_length based on input to prevent over-generation
                input_length = inputs['input_ids'].shape[1]
                max_output_len = min(512, input_length * 2)  # Reduced from 3x to 2x to prevent hallucinations
                
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt),
                    max_length=max_output_len,
                    num_beams=1,
                    length_penalty=0.8,  # Reduced to favor shorter outputs
                    repetition_penalty=1.3,  # Penalize repetition to prevent loops
                    no_repeat_ngram_size=3,  # Prevent repeating 3-word sequences
                    early_stopping=True,
                )
            return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"NLLB translation failed: {e}")
            raise

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("NLLB model not properly loaded")

        src = self._map_lang(source_lang)
        tgt = self._map_lang(target_lang)

        try:
            self._tokenizer.src_lang = src
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            )
            with torch.no_grad():
                # Calculate reasonable max_length based on input to prevent over-generation
                max_input_len = max(len(self._tokenizer.encode(t, add_special_tokens=False)) for t in texts)
                max_output_len = min(512, max_input_len * 2)  # Reduced from 3x to 2x to prevent hallucinations
                
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt),
                    max_length=max_output_len,
                    num_beams=1,
                    length_penalty=0.8,  # Reduced to favor shorter outputs
                    repetition_penalty=1.3,  # Penalize repetition to prevent loops
                    no_repeat_ngram_size=3,  # Prevent repeating 3-word sequences
                    early_stopping=True,
                )
            return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"NLLB batch translation failed: {e}")
            raise


_nllb_direct: NLLBDirect | None = None


def get_nllb_direct(model_path: Optional[str] = None) -> NLLBDirect:
    global _nllb_direct
    if _nllb_direct is None:
        _nllb_direct = NLLBDirect(model_path)
    return _nllb_direct


def translate_text(text: str, source_lang: str, target_lang: str, model_path: Optional[str] = None) -> str:
    nllb = get_nllb_direct(model_path)
    return nllb.translate_text(text, source_lang, target_lang)


def translate_batch(texts: list[str], source_lang: str, target_lang: str, model_path: Optional[str] = None) -> list[str]:
    nllb = get_nllb_direct(model_path)
    return nllb.translate_batch(texts, source_lang, target_lang)



