"""
Direct Facebook M2M100 418M implementation without external downloads.

This module uses pre-downloaded M2M100 models directly using transformers library,
avoiding any external API calls or downloads.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import os
import torch

logger = logging.getLogger(__name__)

class M2M100Direct:
    """Direct implementation of Facebook M2M100 418M translation functionality using local models only"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize M2M100 translator

        Args:
            model_path: Path to the M2M100 model directory. If None, uses default path.
        """
        if model_path is None:
            # Default path - user will update this
            model_path = "/path/to/m2m100_418M_model"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"M2M100 model not found at {self.model_path}. "
                "Please ensure the model is downloaded and update the model path."
            )

        self._tokenizer = None
        self._model = None
        self._device = "cpu"  # Use CPU by default, can be changed to "cuda" if available

        self._load_model()

    def _load_model(self):
        """Load the M2M100 model and tokenizer from local files"""
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            # Load tokenizer
            self._tokenizer = M2M100Tokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )

            # Load model
            self._model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_path,
                local_files_only=True
            )

            logger.info(f"M2M100 model loaded successfully from {self.model_path}")

        except ImportError as e:
            logger.error(
                "transformers library not available. Please install transformers: "
                "pip install transformers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load M2M100 model from {self.model_path}: {e}")
            raise

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for M2M100"""
        # M2M100 418M supports many languages
        # These are the language codes that M2M100 expects
        return {
            "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "ast": "Asturian",
            "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
            "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan",
            "ceb": "Cebuano", "cs": "Czech", "cy": "Welsh", "da": "Danish",
            "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
            "et": "Estonian", "fa": "Persian", "ff": "Fulah", "fi": "Finnish",
            "fr": "French", "fy": "Western Frisian", "ga": "Irish", "gd": "Scottish Gaelic",
            "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew",
            "hi": "Hindi", "hr": "Croatian", "ht": "Haitian", "hu": "Hungarian",
            "hy": "Armenian", "id": "Indonesian", "ig": "Igbo", "ilo": "Iloko",
            "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jv": "Javanese",
            "ka": "Georgian", "kk": "Kazakh", "km": "Central Khmer", "kn": "Kannada",
            "ko": "Korean", "ku": "Kurdish", "ky": "Kirghiz", "la": "Latin",
            "lb": "Luxembourgish", "lg": "Ganda", "ln": "Lingala", "lo": "Lao",
            "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mk": "Macedonian",
            "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay",
            "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
            "no": "Norwegian", "ns": "Northern Sotho", "oc": "Occitan", "or": "Oriya",
            "pa": "Panjabi", "pl": "Polish", "ps": "Pushto", "pt": "Portuguese",
            "ro": "Romanian", "ru": "Russian", "sd": "Sindhi", "si": "Sinhala",
            "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
            "sr": "Serbian", "ss": "Swati", "su": "Sundanese", "sv": "Swedish",
            "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
            "tl": "Tagalog", "tn": "Tswana", "tr": "Turkish", "ug": "Uighur",
            "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
            "wo": "Wolof", "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba",
            "zh": "Chinese", "zu": "Zulu"
        }

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language is supported by M2M100"""
        supported = self.get_supported_languages()
        return lang_code.lower() in supported

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using M2M100 model

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("M2M100 model not properly loaded")

        # Check if languages are supported
        supported_langs = self.get_supported_languages()
        if source_lang.lower() not in supported_langs:
            raise ValueError(f"Source language '{source_lang}' not supported by M2M100")
        if target_lang.lower() not in supported_langs:
            raise ValueError(f"Target language '{target_lang}' not supported by M2M100")

        try:
            # Set source language
            self._tokenizer.src_lang = source_lang.lower()

            # Tokenize input
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Generate translation
            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.get_lang_id(target_lang.lower()),
                    max_length=512,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )

            # Decode translation
            translated_text = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """
        Translate multiple texts in batch

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translated texts
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("M2M100 model not properly loaded")

        # Check if languages are supported
        supported_langs = self.get_supported_languages()
        if source_lang.lower() not in supported_langs:
            raise ValueError(f"Source language '{source_lang}' not supported by M2M100")
        if target_lang.lower() not in supported_langs:
            raise ValueError(f"Target language '{target_lang}' not supported by M2M100")

        try:
            # Set source language
            self._tokenizer.src_lang = source_lang.lower()

            # Tokenize inputs
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Generate translations
            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.get_lang_id(target_lang.lower()),
                    max_length=512,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )

            # Decode translations
            translated_texts = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            return translated_texts

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise


# Global instance
_m2m100_direct = None

def get_m2m100_direct(model_path: Optional[str] = None) -> M2M100Direct:
    """Get the global M2M100Direct instance"""
    global _m2m100_direct
    if _m2m100_direct is None:
        _m2m100_direct = M2M100Direct(model_path)
    return _m2m100_direct


def translate_text(text: str, source_lang: str, target_lang: str, model_path: Optional[str] = None) -> str:
    """Translate text using local M2M100 model only"""
    m2m100 = get_m2m100_direct(model_path)
    return m2m100.translate_text(text, source_lang, target_lang)


def translate_batch(texts: list[str], source_lang: str, target_lang: str, model_path: Optional[str] = None) -> list[str]:
    """Translate multiple texts using local M2M100 model only"""
    m2m100 = get_m2m100_direct(model_path)
    return m2m100.translate_batch(texts, source_lang, target_lang)


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages for M2M100"""
    m2m100 = get_m2m100_direct()
    return m2m100.get_supported_languages()
