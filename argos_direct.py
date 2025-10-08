"""
Direct Argos Translate implementation without argostranslate dependency.

This module uses pre-downloaded Argos Translate models directly using CTranslate2
and SentencePiece, avoiding the need for the argostranslate package and runtime downloads.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import ctranslate2
import sentencepiece as spm

logger = logging.getLogger(__name__)


class ArgosPackage:
    """Represents an Argos Translate package (equivalent to argostranslate.package.IPackage)"""

    def __init__(self, metadata: Dict[str, Any]):
        self.package_version = metadata.get("package_version", "")
        self.argos_version = metadata.get("argos_version", "")
        self.from_code = metadata.get("from_code")
        self.from_name = metadata.get("from_name", "")
        self.to_code = metadata.get("to_code")
        self.to_name = metadata.get("to_name", "")
        self.links = metadata.get("links", [])
        self.type = metadata.get("type", "translate")
        self.package_path: Optional[Path] = None
        self.tokenizer = None
        self.translator = None
        self.target_prefix = ""

    def __repr__(self):
        return f"{self.from_name} -> {self.to_name}"


class ArgosDirect:
    """Direct implementation of Argos Translate functionality using local models only"""

    # Remove remote package index and download directories
    LOCAL_MODELS_DIR = Path(__file__).parent.parent / "models"

    def __init__(self):
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def get_available_packages(self) -> List[ArgosPackage]:
        """Get list of available local packages only"""
        return self.get_installed_packages()

    def download_package(self, package: ArgosPackage) -> Path:
        """Download a package file - REMOVED - only local models supported"""
        raise NotImplementedError("Runtime downloads are not supported. Please download models manually using download_models.py")

    def install_package(self, package_path: Path) -> ArgosPackage:
        """Install a package from a .argosmodel file - REMOVED - only local models supported"""
        raise NotImplementedError("Runtime installation is not supported. Please download models manually using download_models.py")

    def get_package_for_languages(self, from_code: str, to_code: str) -> Optional[ArgosPackage]:
        """Get an installed package for the given language pair"""
        # Check local models directory first
        if self.LOCAL_MODELS_DIR.exists():
            for package_dir in self.LOCAL_MODELS_DIR.iterdir():
                if package_dir.is_dir():
                    metadata_path = package_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path) as f:
                                metadata = json.load(f)
                            if (metadata.get("from_code") == from_code and
                                metadata.get("to_code") == to_code and
                                metadata.get("type") == "translate"):
                                package = ArgosPackage(metadata)
                                package.package_path = package_dir

                                # Load tokenizer and translator
                                sp_model_path = package_dir / "sentencepiece.model"
                                if sp_model_path.exists():
                                    package.tokenizer = spm.SentencePieceProcessor(str(sp_model_path))

                                model_path = package_dir / "model"
                                if model_path.exists():
                                    package.translator = ctranslate2.Translator(str(model_path))

                                return package
                        except Exception as e:
                            logger.warning(f"Failed to load local package from {package_dir}: {e}")

        # Then check user directory
        packages = self.get_installed_packages()
        for package in packages:
            if (package.from_code == from_code and
                package.to_code == to_code and
                package.type == "translate"):
                return package
        return None

    def get_installed_packages(self) -> List[ArgosPackage]:
        """Get list of installed packages from local models directory only"""
        packages = []
        if not self.LOCAL_MODELS_DIR.exists():
            return packages

        for package_dir in self.LOCAL_MODELS_DIR.iterdir():
            if package_dir.is_dir():
                try:
                    metadata_path = package_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)

                        package = ArgosPackage(metadata)
                        package.package_path = package_dir

                        # Load tokenizer and translator if files exist
                        sp_model_path = package_dir / "sentencepiece.model"
                        if sp_model_path.exists():
                            package.tokenizer = spm.SentencePieceProcessor(str(sp_model_path))

                        model_path = package_dir / "model"
                        if model_path.exists():
                            package.translator = ctranslate2.Translator(str(model_path))

                        packages.append(package)
                except Exception as e:
                    logger.warning(f"Failed to load package from {package_dir}: {e}")

        return packages

    def download_and_install_package(self, from_code: str, to_code: str) -> ArgosPackage:
        """Download and install a package - REMOVED - only local models supported"""
        raise NotImplementedError("Runtime downloads are not supported. Please download models manually using download_models.py")


# Global instance
_argos_direct = None

def get_argos_direct() -> ArgosDirect:
    """Get the global ArgosDirect instance"""
    global _argos_direct
    if _argos_direct is None:
        _argos_direct = ArgosDirect()
    return _argos_direct


def translate_text(text: str, from_code: str, to_code: str) -> str:
    """Translate text using local Argos Translate models only"""
    argos = get_argos_direct()

    # Get the required package - no runtime downloads allowed
    package = argos.get_package_for_languages(from_code, to_code)
    if package is None:
        raise ValueError(
            f"No translation model found for {from_code} -> {to_code}. "
            "Please download the required model manually using download_models.py. "
            f"Available models: {[f'{p.from_code}->{p.to_code}' for p in argos.get_installed_packages()]}"
        )

    if package.translator is None or package.tokenizer is None:
        raise RuntimeError(f"Package {from_code}->{to_code} is not properly loaded")

    # Tokenize the input text
    tokens = package.tokenizer.encode(text, out_type=str)

    # Translate using CTranslate2
    # For simplicity, we'll translate in batches of sentences
    sentences = _split_into_sentences(text)
    translated_sentences = []

    for sentence in sentences:
        sentence_tokens = package.tokenizer.encode(sentence, out_type=str)
        if not sentence_tokens:
            translated_sentences.append(sentence)
            continue

        # Translate the tokens
        target_prefix = [[package.target_prefix]] if package.target_prefix else None
        results = package.translator.translate_batch(
            [sentence_tokens],
            target_prefix=target_prefix,
            beam_size=4,
            num_hypotheses=1
        )

        # Decode the translated tokens
        translated_tokens = results[0].hypotheses[0]  # Get best hypothesis
        translated_text = package.tokenizer.decode(translated_tokens)

        # Remove target prefix if present
        if translated_text.startswith(package.target_prefix):
            translated_text = translated_text[len(package.target_prefix):].lstrip()

        translated_sentences.append(translated_text)

    return ' '.join(translated_sentences)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for better translation quality"""
    # Simple sentence splitting - split on sentence endings followed by whitespace and capital letters
    import re

    # Split on common sentence endings
    sentences = re.split(r'([.!?]+)\s+(?=[A-Z])', text)

    # Recombine the delimiters with the following sentence
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            # Combine sentence with its delimiter
            sentence = sentences[i] + sentences[i + 1]
            if sentence.strip():
                result.append(sentence.strip())
        else:
            # Last part without delimiter
            if sentences[i].strip():
                result.append(sentences[i].strip())

    # Handle the last part if it's a complete sentence
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())

    # Filter out empty sentences
    return [s for s in result if s and len(s.strip()) > 0]


def get_installed_languages() -> List[Dict[str, str]]:
    """Get list of installed language codes from local models only"""
    argos = get_argos_direct()
    packages = argos.get_installed_packages()

    languages = set()
    for package in packages:
        if package.from_code:
            languages.add(package.from_code)
        if package.to_code:
            languages.add(package.to_code)

    return [{"code": code, "name": code} for code in sorted(languages)]
