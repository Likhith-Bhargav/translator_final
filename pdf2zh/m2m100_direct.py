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
    """Direct implementation of Facebook mBART translation functionality using local models only"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize mBART translator
        Args:
            model_path: Path to the mBART model directory. If None, uses default path.
        """
        if model_path is None:
            # Default path - user will update this
            self.model_path = Path("/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/mbart-large-50-many-to-many-mmt")
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"mBART model not found at {self.model_path}. "
                "Please ensure the model is downloaded and update the model path."
            )

        self._tokenizer = None
        self._model = None
        self._device = "cpu"  # Use CPU by default, can be changed to "cuda" if available

        self._load_model()

    def _load_model(self):
        """Load the mBART model and tokenizer from local files"""
        try:
            from transformers import MBartForConditionalGeneration, MBart50Tokenizer

            # Load tokenizer
            self._tokenizer = MBart50Tokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )

            # Load model
            self._model = MBartForConditionalGeneration.from_pretrained(
                self.model_path,
                local_files_only=True
            )

            logger.info(f"mBART model loaded successfully from {self.model_path}")

        except ImportError as e:
            logger.error(
                "transformers library not available. Please install transformers: "
                "pip install transformers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load mBART model from {self.model_path}: {e}")
            raise

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for mBART"""
        # mBART-50 supports 50 languages
        # These are the language codes that mBART expects
        return {
            "ar": "Arabic", "cs": "Czech", "de": "German", "en": "English",
            "es": "Spanish", "et": "Estonian", "fi": "Finnish", "fr": "French",
            "gu": "Gujarati", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
            "kk": "Kazakh", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
            "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "ro": "Romanian",
            "ru": "Russian", "si": "Sinhala", "tr": "Turkish", "vi": "Vietnamese",
            "zh": "Chinese", "af": "Afrikaans", "az": "Azerbaijani", "bn": "Bengali",
            "fa": "Persian", "he": "Hebrew", "hr": "Croatian", "id": "Indonesian",
            "ka": "Georgian", "km": "Khmer", "mk": "Macedonian", "ml": "Malayalam",
            "mn": "Mongolian", "mr": "Marathi", "pl": "Polish", "ps": "Pashto",
            "pt": "Portuguese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
            "te": "Telugu", "th": "Thai", "tl": "Tagalog", "uk": "Ukrainian",
            "ur": "Urdu", "xh": "Xhosa", "gl": "Galician", "sl": "Slovenian"
        }

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language is supported by mBART"""
        supported = self.get_supported_languages()
        return lang_code.lower() in supported

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using mBART model

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("mBART model not properly loaded")

        # Check if languages are supported
        supported_langs = self.get_supported_languages()
        if source_lang.lower() not in supported_langs:
            raise ValueError(f"Source language '{source_lang}' not supported by mBART")
        if target_lang.lower() not in supported_langs:
            raise ValueError(f"Target language '{target_lang}' not supported by mBART")

        try:
            # mBART language code mapping (simplified to mBART format)
            mbart_lang_map = {
                "en": "en_XX", "zh": "zh_CN", "fr": "fr_XX", "de": "de_DE",
                "es": "es_XX", "it": "it_IT", "pt": "pt_XX", "ru": "ru_RU",
                "ja": "ja_XX", "ko": "ko_KR", "ar": "ar_AR", "hi": "hi_IN",
                "nl": "nl_XX", "sv": "sv_SE", "da": "da_DK", "no": "nb_NO",
                "fi": "fi_FI", "et": "et_EE", "lv": "lv_LV", "lt": "lt_LT",
                "pl": "pl_PL", "cs": "cs_CZ", "sk": "sk_SK", "sl": "sl_SI",
                "hr": "hr_HR", "bs": "bs_BA", "sr": "sr_RS", "mk": "mk_MK",
                "bg": "bg_BG", "uk": "uk_UA", "ro": "ro_RO", "hu": "hu_HU",
                "tr": "tr_TR", "el": "el_GR", "he": "he_IL", "fa": "fa_IR",
                "vi": "vi_VN", "th": "th_TH", "id": "id_ID", "ms": "ms_MY",
                "tl": "tl_XX", "sw": "sw_KE", "af": "af_ZA", "xh": "xh_ZA"
            }

            source_lang_code = mbart_lang_map.get(source_lang.lower(), source_lang.lower())
            target_lang_code = mbart_lang_map.get(target_lang.lower(), target_lang.lower())

            # Set source language for tokenizer
            self._tokenizer.src_lang = source_lang_code

            # Tokenize input
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Generate translation
            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.lang_code_to_id[target_lang_code],
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
        Translate multiple texts in batch using mBART model

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translated texts
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("mBART model not properly loaded")

        # Check if languages are supported
        supported_langs = self.get_supported_languages()
        if source_lang.lower() not in supported_langs:
            raise ValueError(f"Source language '{source_lang}' not supported by mBART")
        if target_lang.lower() not in supported_langs:
            raise ValueError(f"Target language '{target_lang}' not supported by mBART")

        try:
            # mBART language code mapping (simplified to mBART format)
            mbart_lang_map = {
                "en": "en_XX", "zh": "zh_CN", "fr": "fr_XX", "de": "de_DE",
                "es": "es_XX", "it": "it_IT", "pt": "pt_XX", "ru": "ru_RU",
                "ja": "ja_XX", "ko": "ko_KR", "ar": "ar_AR", "hi": "hi_IN",
                "nl": "nl_XX", "sv": "sv_SE", "da": "da_DK", "no": "nb_NO",
                "fi": "fi_FI", "et": "et_EE", "lv": "lv_LV", "lt": "lt_LT",
                "pl": "pl_PL", "cs": "cs_CZ", "sk": "sk_SK", "sl": "sl_SI",
                "hr": "hr_HR", "bs": "bs_BA", "sr": "sr_RS", "mk": "mk_MK",
                "bg": "bg_BG", "uk": "uk_UA", "ro": "ro_RO", "hu": "hu_HU",
                "tr": "tr_TR", "el": "el_GR", "he": "he_IL", "fa": "fa_IR",
                "vi": "vi_VN", "th": "th_TH", "id": "id_ID", "ms": "ms_MY",
                "tl": "tl_XX", "sw": "sw_KE", "af": "af_ZA", "xh": "xh_ZA"
            }

            source_lang_code = mbart_lang_map.get(source_lang.lower(), source_lang.lower())
            target_lang_code = mbart_lang_map.get(target_lang.lower(), target_lang.lower())

            # Set source language for tokenizer
            self._tokenizer.src_lang = source_lang_code

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
                    forced_bos_token_id=self._tokenizer.lang_code_to_id[target_lang_code],
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
