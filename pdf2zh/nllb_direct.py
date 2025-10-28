"""
Direct Facebook NLLB-200 Distilled 600M implementation using local files only.

This module loads a locally available `facebook/nllb-200-distilled-600M` model
from a user-provided folder path. No external downloads or network calls.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import torch

logger = logging.getLogger(__name__)


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
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt),
                    max_length=1024,
                    num_beams=4,
                    length_penalty=1.0,
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
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt),
                    max_length=1024,
                    num_beams=4,
                    length_penalty=1.0,
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


