"""
PDF Document Translation Backend using NLLB-200
Complete standalone implementation with meaningful variable names
All imports are from libraries already used in the original application
"""

# ============================================================================
# IMPORTS - All from original application, no new libraries
# ============================================================================

# Standard library (Python built-in)
import asyncio
import concurrent.futures
import io
import json
import logging
import os
import re
import tempfile
import unicodedata
from asyncio import CancelledError
from copy import copy
from enum import Enum
from pathlib import Path
from string import Template
from threading import RLock
from typing import Any, BinaryIO, Dict, List, Optional, Tuple
import multiprocessing

# Third-party libraries (already in original application)
import numpy as np
import regex
import torch
from peewee import AutoField, CharField, Model, SqliteDatabase, TextField, SQL
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font
from tenacity import retry, wait_fixed
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pdfminer.pdfcolor import PREDEFINED_COLORSPACE, PDFColorSpace
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import (
    PDFPageInterpreter,
    Color,
    LITERAL_FORM,
    LITERAL_IMAGE,
    PDFContentParser,
    PDFInterpreterError,
    PDFStackT,
)
from pdfminer.pdffont import PDFFont
from pdfminer.pdftypes import (
    PDFObjRef,
    dict_value,
    list_value,
    resolve1,
    stream_value,
)
from pdfminer.psexceptions import PSEOF
from pdfminer.psparser import (
    PSKeyword,
    keyword_name,
    literal_name,
)
from pdfminer.utils import (
    MATRIX_IDENTITY,
    Matrix,
    Rect,
    mult_matrix,
    apply_matrix_pt,
)
from pdfminer import settings

# ONNX for layout detection (already in original)
try:
    import ast
    import cv2
    import onnx
    import onnxruntime
except ImportError as e:
    if "DLL load failed" in str(e):
        raise OSError(
            "Microsoft Visual C++ Redistributable is not installed. "
            "Download it at https://aka.ms/vs/17/release/vc_redist.x64.exe"
        ) from e
    raise

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigManager:
    """Singleton configuration manager with thread-safe operations."""
    _instance = None
    _lock = RLock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._config_path = Path.home() / ".config" / "PDFMathTranslate" / "config.json"
        self._config_data = {}
        self._ensure_config_exists()

    def _ensure_config_exists(self):
        if not self._config_path.exists():
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_data = {}
            self._save_config()
        else:
            self._load_config()

    def _load_config(self):
        with self._lock:
            with self._config_path.open("r", encoding="utf-8") as config_file:
                self._config_data = json.load(config_file)

    def _save_config(self):
        with self._lock:
            cleaned_data = self._remove_circular_references(self._config_data)
            with self._config_path.open("w", encoding="utf-8") as config_file:
                json.dump(cleaned_data, config_file, indent=4, ensure_ascii=False)

    def _remove_circular_references(self, obj, seen=None):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        if isinstance(obj, dict):
            return {key: self._remove_circular_references(value, seen) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._remove_circular_references(item, seen) for item in obj]
        return obj

    @classmethod
    def get(cls, key, default=None):
        instance = cls.get_instance()
        if key in instance._config_data:
            return instance._config_data[key]
        if key in os.environ:
            value = os.environ[key]
            instance._config_data[key] = value
            instance._save_config()
            return value
        if default is not None:
            instance._config_data[key] = default
            instance._save_config()
            return default
        return default

    @classmethod
    def set(cls, key, value):
        instance = cls.get_instance()
        with instance._lock:
            instance._config_data[key] = value
            instance._save_config()


# ============================================================================
# Translation Cache
# ============================================================================

translation_cache_db = SqliteDatabase(None)


class _TranslationCache(Model):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    translate_engine_params = TextField()
    original_text = TextField()
    translation = TextField()

    class Meta:
        database = translation_cache_db
        constraints = [
            SQL(
                """
            UNIQUE (
                translate_engine,
                translate_engine_params,
                original_text
                )
            ON CONFLICT REPLACE
            """
            )
        ]


class TranslationCache:
    @staticmethod
    def _sort_dict_recursively(obj):
        if isinstance(obj, dict):
            return {
                key: TranslationCache._sort_dict_recursively(value)
                for key in sorted(obj.keys())
                for value in [obj[key]]
            }
        elif isinstance(obj, list):
            return [TranslationCache._sort_dict_recursively(item) for item in obj]
        return obj

    def __init__(self, translate_engine: str, translate_engine_params: dict = None):
        assert len(translate_engine) < 20
        self.translate_engine = translate_engine
        self.replace_params(translate_engine_params)

    def replace_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params = params
        params = self._sort_dict_recursively(params)
        self.translate_engine_params = json.dumps(params)

    def get(self, original_text: str) -> Optional[str]:
        result = _TranslationCache.get_or_none(
            translate_engine=self.translate_engine,
            translate_engine_params=self.translate_engine_params,
            original_text=original_text,
        )
        return result.translation if result else None

    def set(self, original_text: str, translation: str):
        try:
            _TranslationCache.create(
                translate_engine=self.translate_engine,
                translate_engine_params=self.translate_engine_params,
                original_text=original_text,
                translation=translation,
            )
        except Exception as error:
            logger.debug(f"Error setting cache: {error}")


def init_cache_database():
    """Initialize the translation cache database."""
    cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "pdf2zh")
    os.makedirs(cache_folder, exist_ok=True)
    cache_db_path = os.path.join(cache_folder, "cache.v1.db")
    translation_cache_db.init(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    translation_cache_db.create_tables([_TranslationCache], safe=True)


# Initialize cache on module load
init_cache_database()


# ============================================================================
# NLLB Translator
# ============================================================================

FORMULA_PLACEHOLDER_PATTERN = re.compile(r"\{v\d+\}")


def protect_formula_placeholders(text):
    """Protects {vNNN} placeholders so the translation model won't corrupt them."""
    placeholder_mapping = {}

    def replace_with_protected_placeholder(match):
        protected_key = f"__PLACEHOLDER_v{match.group(0)[2:-1]}__"
        placeholder_mapping[protected_key] = match.group(0)
        return protected_key

    protected_text = FORMULA_PLACEHOLDER_PATTERN.sub(replace_with_protected_placeholder, text)
    return protected_text, placeholder_mapping


def restore_formula_placeholders(text, placeholder_mapping):
    """Restores protected placeholder tags to their original {vNNN}."""
    for protected_key, original_placeholder in placeholder_mapping.items():
        text = text.replace(protected_key, original_placeholder)
    return text


class NLLBDirect:
    """Local NLLB translator wrapper (no network)."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
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
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            logger.info(f"NLLB model loaded from {self.model_path}")
        except Exception as error:
            logger.error(f"Failed to load NLLB model from {self.model_path}: {error}")
            raise

    def _map_language_code(self, code: str) -> str:
        """Map common short codes to NLLB language tags."""
        if "_" in code:
            return code
        language_mapping = {
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
            "th": "tha_Thai",
            "tr": "tur_Latn",
            "pl": "pol_Latn",
            "nl": "nld_Latn",
            "sv": "swe_Latn",
            "da": "dan_Latn",
            "fi": "fin_Latn",
            "no": "nob_Latn",
            "cs": "ces_Latn",
            "uk": "ukr_Cyrl",
            "ro": "ron_Latn",
            "el": "ell_Grek",
        }
        return language_mapping.get(code.lower(), code)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("NLLB model not properly loaded")

        source_language = self._map_language_code(source_lang)
        target_language = self._map_language_code(target_lang)

        try:
            self._tokenizer.src_lang = source_language
            tokenized_inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            with torch.no_grad():
                input_length = tokenized_inputs["input_ids"].shape[1]
                max_output_length = min(512, input_length * 2)

                generated_tokens = self._model.generate(
                    **tokenized_inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_language),
                    max_length=max_output_length,
                    num_beams=1,
                    length_penalty=0.8,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            return self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        except Exception as error:
            logger.error(f"NLLB translation failed: {error}")
            raise


_global_nllb_translator: Optional[NLLBDirect] = None


def get_nllb_translator(model_path: Optional[str] = None) -> NLLBDirect:
    global _global_nllb_translator
    if _global_nllb_translator is None:
        _global_nllb_translator = NLLBDirect(model_path)
    return _global_nllb_translator


class NLLBTranslator:
    """NLLB translator with caching support."""
    name = "nllb"

    def __init__(
        self,
        source_language: str,
        target_language: str,
        model_path: Optional[str] = None,
        ignore_cache: bool = False,
    ):
        self.lang_in = source_language
        self.lang_out = target_language
        self.model_path = model_path
        self.ignore_cache = ignore_cache
        self._passthrough = self.lang_in == self.lang_out

        if not self._passthrough:
            self.nllb_translator = get_nllb_translator(self.model_path)

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": source_language,
                "lang_out": target_language,
                "model": model_path or "default",
            },
        )

    def translate(self, text: str) -> str:
        if self._passthrough:
            return text

        if not self.ignore_cache:
            cached_translation = self.cache.get(text)
            if cached_translation is not None:
                return cached_translation

        try:
            translated_text = self.nllb_translator.translate_text(
                text, self.lang_in, self.lang_out
            )
            self.cache.set(text, translated_text)
            return translated_text
        except Exception:
            raise


# ============================================================================
# Document Layout Detection
# ============================================================================

class DetectionBox:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, detection_data):
        self.xyxy = detection_data[:4]
        self.confidence = detection_data[-2]
        self.class_id = detection_data[-1]


class LayoutDetectionResult:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, boxes, class_names):
        self.boxes = [DetectionBox(data=detection) for detection in boxes]
        self.boxes.sort(key=lambda box: box.confidence, reverse=True)
        self.names = class_names


class DocumentLayoutModel:
    """ONNX-based document layout detection model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        onnx_model = onnx.load(model_path)
        model_metadata = {property.key: property.value for property in onnx_model.metadata_props}
        self._stride = ast.literal_eval(model_metadata["stride"])
        self._class_names = ast.literal_eval(model_metadata["names"])
        self.model = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    @staticmethod
    def from_pretrained():
        model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/doclayout_yolo_docstructbench_imgsz1024.onnx"
        return DocumentLayoutModel(model_path)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, target_shape):
        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)

        original_height, original_width = image.shape[:2]
        target_height, target_width = target_shape
        scale_ratio = min(target_height / original_height, target_width / original_width)
        resized_height = int(round(original_height * scale_ratio))
        resized_width = int(round(original_width * scale_ratio))

        resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_width = (target_width - resized_width) % self.stride
        pad_height = (target_height - resized_height) % self.stride
        top_padding = pad_height // 2
        bottom_padding = pad_height - top_padding
        left_padding = pad_width // 2
        right_padding = pad_width - left_padding

        padded_image = cv2.copyMakeBorder(
            resized_image, top_padding, bottom_padding, left_padding, right_padding,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded_image

    def scale_boxes(self, processed_image_shape, boxes, original_image_shape):
        scale_gain = min(
            processed_image_shape[0] / original_image_shape[0],
            processed_image_shape[1] / original_image_shape[1]
        )
        pad_x = round((processed_image_shape[1] - original_image_shape[1] * scale_gain) / 2 - 0.1)
        pad_y = round((processed_image_shape[0] - original_image_shape[0] * scale_gain) / 2 - 0.1)
        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / scale_gain
        return boxes

    def predict(self, image, image_size=1024, **kwargs):
        original_height, original_width = image.shape[:2]
        processed_image = self.resize_and_pad_image(image, target_shape=image_size)
        processed_image = np.transpose(processed_image, (2, 0, 1))
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_height, processed_width = processed_image.shape[2:]

        predictions = self.model.run(None, {"images": processed_image})[0]
        predictions = predictions[predictions[..., 4] > 0.25]
        predictions[..., :4] = self.scale_boxes(
            (processed_height, processed_width),
            predictions[..., :4],
            (original_height, original_width)
        )
        return [LayoutDetectionResult(boxes=predictions, class_names=self._class_names)]


# ============================================================================
# Font Management
# ============================================================================

def get_font_path_for_language(target_language: str) -> str:
    """Get the appropriate font path for the target language."""
    normalized_language = target_language.lower()
    assets_directory = Path("/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/BabelDOC-Assets-main")
    fonts_directory = assets_directory / "fonts"

    # Language to font mapping
    if normalized_language in ["zh", "zh-cn", "zh-hans", "zh-tw", "zh-hant"]:
        font_filename = "SourceHanSerifCN-Regular.ttf"
    elif normalized_language == "ja":
        font_filename = "SourceHanSerifJP-Regular.ttf"
    elif normalized_language == "ko":
        font_filename = "SourceHanSerifKR-Regular.ttf"
    else:
        font_filename = "GoNotoKurrent-Regular.ttf"

    font_path = fonts_directory / font_filename
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    logger.info(f"Using font: {font_path}")
    return str(font_path)


# ============================================================================
# PDF Processing
# ============================================================================

class ExtendedPDFPageInterpreter(PDFPageInterpreter):
    """Extended PDF page interpreter with object patching support."""

    def __init__(
        self, resource_manager: PDFResourceManager, device: PDFDevice, object_patch_dict: dict
    ) -> None:
        self.rsrcmgr = resource_manager
        self.device = device
        self.obj_patch = object_patch_dict

    def duplicate_interpreter(self) -> "ExtendedPDFPageInterpreter":
        """Creates a duplicate copy of this PDF page interpreter."""
        return self.__class__(self.rsrcmgr, self.device, self.obj_patch)

    def init_resources(self, resources: Dict[object, object]) -> None:
        self.resources = resources
        self.fontmap: Dict[object, PDFFont] = {}
        self.fontid: Dict[PDFFont, object] = {}
        self.xobjmap = {}
        self.csmap: Dict[str, PDFColorSpace] = PREDEFINED_COLORSPACE.copy()
        if not resources:
            return

        def get_colorspace(color_spec: object) -> Optional[PDFColorSpace]:
            if isinstance(color_spec, list):
                colorspace_name = literal_name(color_spec[0])
            else:
                colorspace_name = literal_name(color_spec)
            if colorspace_name == "ICCBased" and isinstance(color_spec, list) and len(color_spec) >= 2:
                return PDFColorSpace(colorspace_name, stream_value(color_spec[1])["N"])
            elif colorspace_name == "DeviceN" and isinstance(color_spec, list) and len(color_spec) >= 2:
                return PDFColorSpace(colorspace_name, len(list_value(color_spec[1])))
            else:
                return PREDEFINED_COLORSPACE.get(colorspace_name)

        for resource_key, resource_value in dict_value(resources).items():
            if resource_key == "Font":
                for font_id, font_spec in dict_value(resource_value).items():
                    object_id = None
                    if isinstance(font_spec, PDFObjRef):
                        object_id = font_spec.objid
                    font_spec = dict_value(font_spec)
                    self.fontmap[font_id] = self.rsrcmgr.get_font(object_id, font_spec)
                    self.fontmap[font_id].descent = 0
                    self.fontid[self.fontmap[font_id]] = font_id
            elif resource_key == "ColorSpace":
                for colorspace_id, colorspace_spec in dict_value(resource_value).items():
                    colorspace = get_colorspace(resolve1(colorspace_spec))
                    if colorspace is not None:
                        self.csmap[colorspace_id] = colorspace
            elif resource_key == "ProcSet":
                self.rsrcmgr.get_procset(list_value(resource_value))
            elif resource_key == "XObject":
                for xobject_id, xobject_stream in dict_value(resource_value).items():
                    self.xobjmap[xobject_id] = xobject_stream

    def do_S(self) -> None:
        def is_black_color(color: Color) -> bool:
            if isinstance(color, Tuple):
                return sum(color) == 0
            else:
                return color == 0

        if (
            len(self.curpath) == 2
            and self.curpath[0][0] == "m"
            and self.curpath[1][0] == "l"
            and apply_matrix_pt(self.ctm, self.curpath[0][-2:])[1]
            == apply_matrix_pt(self.ctm, self.curpath[1][-2:])[1]
            and is_black_color(self.graphicstate.scolor)
        ):
            self.device.paint_path(self.graphicstate, True, False, False, self.curpath)
            self.curpath = []
            return "n"
        else:
            self.curpath = []

    def do_f(self) -> None:
        self.curpath = []

    def do_F(self) -> None:
        pass

    def do_f_a(self) -> None:
        self.curpath = []

    def do_B(self) -> None:
        self.curpath = []

    def do_B_a(self) -> None:
        self.curpath = []

    def do_SCN(self) -> None:
        if self.scs:
            num_components = self.scs.ncomponents
        else:
            if settings.STRICT:
                raise PDFInterpreterError("No colorspace specified!")
            num_components = 1
        color_args = self.pop(num_components)
        self.graphicstate.scolor = color_args
        return color_args

    def do_scn(self) -> None:
        if self.ncs:
            num_components = self.ncs.ncomponents
        else:
            if settings.STRICT:
                raise PDFInterpreterError("No colorspace specified!")
            num_components = 1
        color_args = self.pop(num_components)
        self.graphicstate.ncolor = color_args
        return color_args

    def do_SC(self) -> None:
        return self.do_SCN()

    def do_sc(self) -> None:
        return self.do_scn()

    def do_Do(self, xobject_id_arg: PDFStackT) -> None:
        xobject_id = literal_name(xobject_id_arg)
        try:
            xobject = stream_value(self.xobjmap[xobject_id])
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError("Undefined xobject id: %r" % xobject_id)
            return
        xobject_subtype = xobject.get("Subtype")
        if xobject_subtype is LITERAL_FORM and "BBox" in xobject:
            interpreter = self.duplicate_interpreter()
            bounding_box = list_value(xobject["BBox"])
            transformation_matrix = list_value(xobject.get("Matrix", MATRIX_IDENTITY))
            xobject_resources = xobject.get("Resources")
            if xobject_resources:
                resources = dict_value(xobject_resources)
            else:
                resources = self.resources.copy()
            self.device.begin_figure(xobject_id, bounding_box, transformation_matrix)
            current_transformation_matrix = mult_matrix(transformation_matrix, self.ctm)
            base_operations = interpreter.render_contents(resources, [xobject], ctm=current_transformation_matrix)
            self.ncs = interpreter.ncs
            self.scs = interpreter.scs
            try:
                self.device.fontid = interpreter.fontid
                self.device.fontmap = interpreter.fontmap
                new_operations = self.device.end_figure(xobject_id)
                inverse_matrix = np.linalg.inv(np.array(current_transformation_matrix[:4]).reshape(2, 2))
                numpy_version = np.__version__
                if numpy_version.split(".")[0] >= "2":
                    inverse_position = -np.asmatrix(current_transformation_matrix[4:]) * inverse_matrix
                else:
                    inverse_position = -np.mat(current_transformation_matrix[4:]) * inverse_matrix
                matrix_a, matrix_b, matrix_c, matrix_d = inverse_matrix.reshape(4).tolist()
                matrix_e, matrix_f = inverse_position.tolist()[0]
                self.obj_patch[self.xobjmap[xobject_id].objid] = (
                    f"q {base_operations}Q {matrix_a} {matrix_b} {matrix_c} {matrix_d} {matrix_e} {matrix_f} cm {new_operations}"
                )
            except Exception:
                pass
        elif xobject_subtype is LITERAL_IMAGE and "Width" in xobject and "Height" in xobject:
            self.device.begin_figure(xobject_id, (0, 0, 1, 1), MATRIX_IDENTITY)
            self.device.render_image(xobject_id, xobject)
            self.device.end_figure(xobject_id)
        else:
            pass

    def process_page(self, page: PDFPage) -> None:
        (crop_x0, crop_y0, crop_x1, crop_y1) = page.cropbox
        if page.rotate == 90:
            current_transformation_matrix = (0, -1, 1, 0, -crop_y0, crop_x1)
        elif page.rotate == 180:
            current_transformation_matrix = (-1, 0, 0, -1, crop_x1, crop_y1)
        elif page.rotate == 270:
            current_transformation_matrix = (0, 1, -1, 0, crop_y1, -crop_x0)
        else:
            current_transformation_matrix = (1, 0, 0, 1, -crop_x0, -crop_y0)
        self.device.begin_page(page, current_transformation_matrix)
        base_operations = self.render_contents(page.resources, page.contents, ctm=current_transformation_matrix)
        self.device.fontid = self.fontid
        self.device.fontmap = self.fontmap
        new_operations = self.device.end_page(page)
        self.obj_patch[page.page_xref] = (
            f"q {base_operations}Q 1 0 0 1 {crop_x0} {crop_y0} cm {new_operations}"
        )
        for content_object in page.contents:
            self.obj_patch[content_object.objid] = ""

    def render_contents(
        self,
        resources: Dict[object, object],
        content_streams: list,
        ctm: Matrix = MATRIX_IDENTITY,
    ) -> None:
        self.init_resources(resources)
        self.init_state(ctm)
        return self.execute(list_value(content_streams))

    def execute(self, content_streams: list) -> None:
        pdf_operations = ""
        try:
            content_parser = PDFContentParser(content_streams)
        except PSEOF:
            return
        while True:
            try:
                (_, parsed_object) = content_parser.nextobject()
            except PSEOF:
                break
            if isinstance(parsed_object, PSKeyword):
                operator_name = keyword_name(parsed_object)
                method_name = "do_%s" % operator_name.replace("*", "_a").replace('"', "_w").replace(
                    "'", "_q"
                )
                if hasattr(self, method_name):
                    method_function = getattr(self, method_name)
                    num_args = method_function.__code__.co_argcount - 1
                    if num_args:
                        method_args = self.pop(num_args)
                        if len(method_args) == num_args:
                            method_function(*method_args)
                            if not (
                                operator_name[0] == "T"
                                or operator_name in ['"', "'", "EI", "MP", "DP", "BMC", "BDC"]
                            ):
                                args_string = " ".join(
                                    [
                                        (
                                            f"{arg:f}"
                                            if isinstance(arg, float)
                                            else str(arg).replace("'", "")
                                        )
                                        for arg in method_args
                                    ]
                                )
                                pdf_operations += f"{args_string} {operator_name} "
                    else:
                        returned_args = method_function()
                        if returned_args is None:
                            returned_args = []
                        if not (operator_name[0] == "T" or operator_name in ["BI", "ID", "EMC"]):
                            args_string = " ".join(
                                [
                                    (
                                        f"{arg:f}"
                                        if isinstance(arg, float)
                                        else str(arg).replace("'", "")
                                    )
                                    for arg in returned_args
                                ]
                            )
                            pdf_operations += f"{args_string} {operator_name} "
                elif settings.STRICT:
                    error_message = "Unknown operator: %r" % operator_name
                    raise PDFInterpreterError(error_message)
            else:
                self.push(parsed_object)
        return pdf_operations


# ============================================================================
# PDF Converter with Translation
# ============================================================================

class ExtendedPDFConverter(PDFConverter):
    def __init__(self, resource_manager: PDFResourceManager) -> None:
        PDFConverter.__init__(self, resource_manager, None, "utf-8", 1, None)

    def begin_page(self, page, current_transformation_matrix) -> None:
        (crop_x0, crop_y0, crop_x1, crop_y1) = page.cropbox
        (crop_x0, crop_y0) = apply_matrix_pt(current_transformation_matrix, (crop_x0, crop_y0))
        (crop_x1, crop_y1) = apply_matrix_pt(current_transformation_matrix, (crop_x1, crop_y1))
        media_box = (0, 0, abs(crop_x0 - crop_x1), abs(crop_y0 - crop_y1))
        self.cur_item = LTPage(page.pageno, media_box)

    def end_page(self, page):
        return self.receive_layout(self.cur_item)

    def begin_figure(self, figure_name, bounding_box, transformation_matrix) -> None:
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(figure_name, bounding_box, mult_matrix(transformation_matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, figure_name: str) -> None:
        completed_figure = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(completed_figure)
        return self.receive_layout(completed_figure)

    def render_char(
        self,
        transformation_matrix,
        font,
        font_size: float,
        scaling: float,
        rise: float,
        character_id: int,
        non_stroking_colorspace,
        graphic_state: PDFGraphicState,
    ) -> float:
        try:
            character_text = font.to_unichr(character_id)
            assert isinstance(character_text, str), str(type(character_text))
        except PDFUnicodeNotDefined:
            character_text = self.handle_undefined_char(font, character_id)
        text_width = font.char_width(character_id)
        text_displacement = font.char_disp(character_id)
        layout_item = LTChar(
            transformation_matrix,
            font,
            font_size,
            scaling,
            rise,
            character_text,
            text_width,
            text_displacement,
            non_stroking_colorspace,
            graphic_state,
        )
        self.cur_item.add(layout_item)
        layout_item.cid = character_id
        layout_item.font = font
        return layout_item.adv


class ParagraphInfo:
    def __init__(self, initial_y, initial_x, left_boundary, right_boundary, top_boundary, bottom_boundary, font_size, has_linebreak):
        self.y: float = initial_y
        self.x: float = initial_x
        self.x0: float = left_boundary
        self.x1: float = right_boundary
        self.y0: float = top_boundary
        self.y1: float = bottom_boundary
        self.size: float = font_size
        self.brk: bool = has_linebreak


class OperationType(Enum):
    TEXT = "text"
    LINE = "line"


class TranslationConverter(ExtendedPDFConverter):
    def __init__(
        self,
        resource_manager,
        formula_font_pattern: str = None,
        formula_char_pattern: str = None,
        thread_count: int = 0,
        layout_map={},
        source_language: str = "",
        target_language: str = "",
        noto_font_name: str = "",
        noto_font: Font = None,
        ignore_cache: bool = False,
    ) -> None:
        super().__init__(resource_manager)
        self.formula_font_pattern = formula_font_pattern
        self.formula_char_pattern = formula_char_pattern
        self.thread_count = thread_count
        self.layout_map = layout_map
        self.noto_font_name = noto_font_name
        self.noto_font = noto_font
        self.translator = NLLBTranslator(source_language, target_language, ignore_cache=ignore_cache)

    def receive_layout(self, layout_page: LTPage):
        is_hindi_language = (self.translator.lang_in or "").lower().startswith("hi")

        # Text and paragraph tracking
        text_segments_stack: list[str] = []
        paragraph_stack: list[ParagraphInfo] = []

        # Formula tracking
        formula_bracket_count: int = 0
        formula_chars_stack: list[LTChar] = []
        formula_lines_stack: list[LTLine] = []
        formula_vertical_offset: float = 0

        # Formula groups
        formula_groups: list[list[LTChar]] = []
        formula_lines_groups: list[list[LTLine]] = []
        formula_vertical_offsets: list[float] = []
        formula_widths: list[float] = []

        # Global tracking
        global_lines_stack: list[LTLine] = []
        previous_char: LTChar = None
        previous_layout_class: int = -1
        max_inline_formula_width: float = layout_page.width / 4
        pdf_operations: str = ""

        def is_formula_character(font_name: str, character: str):
            if isinstance(font_name, bytes):
                try:
                    font_name = font_name.decode("utf-8")
                except UnicodeDecodeError:
                    font_name = ""
            font_name = font_name.split("+")[-1]

            if re.match(r"\(cid:", character):
                return True

            if self.formula_font_pattern:
                if re.match(self.formula_font_pattern, font_name):
                    return True
            else:
                if re.match(
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font_name,
                ):
                    return True

            if self.formula_char_pattern:
                if re.match(self.formula_char_pattern, character):
                    return True
            else:
                if (
                    character
                    and character != " "
                    and (
                        unicodedata.category(character[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]
                        or ord(character[0]) in range(0x370, 0x400)
                    )
                ):
                    return True
            return False

        def split_into_grapheme_clusters(text):
            return regex.findall(r"\X", text)

        line_threshold_pixels = 10
        previous_hindi_char = None

        for child_element in layout_page:
            if isinstance(child_element, LTChar):
                if is_hindi_language:
                    if not text_segments_stack:
                        text_segments_stack.append("")
                        paragraph_stack.append(
                            ParagraphInfo(
                                child_element.y0,
                                child_element.x0,
                                child_element.x0,
                                child_element.x0,
                                child_element.y0,
                                child_element.y1,
                                child_element.size,
                                False,
                            )
                        )
                    elif (
                        previous_hindi_char is not None
                        and abs(child_element.y0 - previous_hindi_char.y0) > line_threshold_pixels
                    ):
                        text_segments_stack.append("")
                        paragraph_stack.append(
                            ParagraphInfo(
                                child_element.y0,
                                child_element.x0,
                                child_element.x0,
                                child_element.x0,
                                child_element.y0,
                                child_element.y1,
                                child_element.size,
                                False,
                            )
                        )
                    for grapheme_cluster in split_into_grapheme_clusters(child_element.get_text()):
                        text_segments_stack[-1] += grapheme_cluster
                    paragraph_stack[-1].x0 = min(paragraph_stack[-1].x0, child_element.x0)
                    paragraph_stack[-1].x1 = max(paragraph_stack[-1].x1, child_element.x1)
                    paragraph_stack[-1].y0 = min(paragraph_stack[-1].y0, child_element.y0)
                    paragraph_stack[-1].y1 = max(paragraph_stack[-1].y1, child_element.y1)
                    previous_hindi_char = child_element
                    previous_char = child_element
                    previous_layout_class = 0
                    continue

                is_current_formula = False
                page_layout = self.layout_map[layout_page.pageid]
                layout_height, layout_width = page_layout.shape
                char_x = np.clip(int(child_element.x0), 0, layout_width - 1)
                char_y = np.clip(int(child_element.y0), 0, layout_height - 1)
                current_layout_class = page_layout[char_y, char_x]

                if child_element.get_text() == "•":
                    current_layout_class = 0

                if (
                    current_layout_class == 0
                    or (
                        current_layout_class == previous_layout_class
                        and len(text_segments_stack[-1].strip()) > 1
                        and child_element.size < paragraph_stack[-1].size * 0.79
                    )
                    or is_formula_character(child_element.fontname, child_element.get_text())
                    or (child_element.matrix[0] == 0 and child_element.matrix[3] == 0)
                ):
                    is_current_formula = True

                if not is_current_formula:
                    if formula_chars_stack and child_element.get_text() == "(":
                        is_current_formula = True
                        formula_bracket_count += 1
                    if formula_bracket_count and child_element.get_text() == ")":
                        is_current_formula = True
                        formula_bracket_count -= 1

                if (
                    not is_current_formula
                    or current_layout_class != previous_layout_class
                    or (text_segments_stack[-1] != "" and abs(child_element.x0 - previous_char.x0) > max_inline_formula_width)
                ):
                    if formula_chars_stack:
                        if (
                            not is_current_formula
                            and current_layout_class == previous_layout_class
                            and child_element.x0 > max([formula_char.x0 for formula_char in formula_chars_stack])
                        ):
                            formula_vertical_offset = formula_chars_stack[0].y0 - child_element.y0
                        if text_segments_stack[-1] == "":
                            previous_layout_class = -1
                        text_segments_stack[-1] += f"{{v{len(formula_groups)}}}"
                        formula_groups.append(formula_chars_stack)
                        formula_lines_groups.append(formula_lines_stack)
                        formula_vertical_offsets.append(formula_vertical_offset)
                        formula_chars_stack = []
                        formula_lines_stack = []
                        formula_vertical_offset = 0

                if not formula_chars_stack:
                    if current_layout_class == previous_layout_class:
                        if child_element.x0 > previous_char.x1 + 1:
                            text_segments_stack[-1] += " "
                        elif child_element.x1 < previous_char.x0:
                            text_segments_stack[-1] += " "
                            paragraph_stack[-1].brk = True
                    else:
                        text_segments_stack.append("")
                        paragraph_stack.append(
                            ParagraphInfo(
                                child_element.y0,
                                child_element.x0,
                                child_element.x0,
                                child_element.x0,
                                child_element.y0,
                                child_element.y1,
                                child_element.size,
                                False,
                            )
                        )

                if not is_current_formula:
                    if (
                        child_element.size > paragraph_stack[-1].size or len(text_segments_stack[-1].strip()) == 1
                    ) and child_element.get_text() != " ":
                        paragraph_stack[-1].y -= child_element.size - paragraph_stack[-1].size
                        paragraph_stack[-1].size = child_element.size
                    if is_hindi_language:
                        for grapheme_cluster in split_into_grapheme_clusters(child_element.get_text()):
                            text_segments_stack[-1] += grapheme_cluster
                    else:
                        text_segments_stack[-1] += child_element.get_text()
                else:
                    if not formula_chars_stack and current_layout_class == previous_layout_class and child_element.x0 > previous_char.x0:
                        formula_vertical_offset = child_element.y0 - previous_char.y0
                    formula_chars_stack.append(child_element)

                paragraph_stack[-1].x0 = min(paragraph_stack[-1].x0, child_element.x0)
                paragraph_stack[-1].x1 = max(paragraph_stack[-1].x1, child_element.x1)
                paragraph_stack[-1].y0 = min(paragraph_stack[-1].y0, child_element.y0)
                paragraph_stack[-1].y1 = max(paragraph_stack[-1].y1, child_element.y1)
                previous_char = child_element
                previous_layout_class = current_layout_class

            elif isinstance(child_element, LTFigure):
                pass
            elif isinstance(child_element, LTLine):
                page_layout = self.layout_map[layout_page.pageid]
                layout_height, layout_width = page_layout.shape
                line_x = np.clip(int(child_element.x0), 0, layout_width - 1)
                line_y = np.clip(int(child_element.y0), 0, layout_height - 1)
                current_layout_class = page_layout[line_y, line_x]
                if formula_chars_stack and current_layout_class == previous_layout_class:
                    formula_lines_stack.append(child_element)
                else:
                    global_lines_stack.append(child_element)
            else:
                pass

        if formula_chars_stack:
            text_segments_stack[-1] += f"{{v{len(formula_groups)}}}"
            formula_groups.append(formula_chars_stack)
            formula_lines_groups.append(formula_lines_stack)
            formula_vertical_offsets.append(formula_vertical_offset)

        for formula_id, formula_char_group in enumerate(formula_groups):
            formula_width = max([char.x1 for char in formula_char_group]) - formula_char_group[0].x0
            formula_widths.append(formula_width)

        @retry(wait=wait_fixed(1))
        def translate_text_segment(text_segment: str):
            if not text_segment.strip() or re.match(r"^\{v\d+\}$", text_segment):
                return text_segment
            try:
                translated_segment = self.translator.translate(text_segment)
                return translated_segment
            except BaseException as translation_error:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(translation_error)
                else:
                    logger.exception(translation_error, exc_info=False)
                raise translation_error

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.thread_count)
        ) as executor:
            translated_segments = list(executor.map(translate_text_segment, text_segments_stack))

        def encode_as_raw_string(current_font: str, character_stack: str):
            if current_font == self.noto_font_name:
                return "".join(
                    ["%04x" % self.noto_font.has_glyph(ord(char)) for char in character_stack]
                )
            elif isinstance(self.fontmap[current_font], PDFCIDFont):
                return "".join(["%04x" % ord(char) for char in character_stack])
            else:
                return "".join(["%02x" % ord(char) for char in character_stack])

        LANGUAGE_LINE_HEIGHT_MAP = {
            "zh-cn": 1.4,
            "zh-tw": 1.4,
            "zh-hans": 1.4,
            "zh-hant": 1.4,
            "zh": 1.4,
            "ja": 1.1,
            "ko": 1.2,
            "en": 1.2,
            "ar": 1.0,
            "ru": 0.8,
            "uk": 0.8,
            "ta": 0.8,
        }
        default_line_height = LANGUAGE_LINE_HEIGHT_MAP.get(
            self.translator.lang_out.lower(), 1.1
        )
        operations_list = []

        def generate_text_operation(font_name, font_size, position_x, position_y, raw_text):
            return f"/{font_name} {font_size:f} Tf 1 0 0 1 {position_x:f} {position_y:f} Tm [<{raw_text}>] TJ "

        def generate_line_operation(position_x, position_y, line_length_x, line_length_y, line_width):
            return f"ET q 1 0 0 1 {position_x:f} {position_y:f} cm [] 0 d 0 J {line_width:f} w 0 0 m {line_length_x:f} {line_length_y:f} l S Q BT "

        for segment_id, translated_text in enumerate(translated_segments):
            paragraph_x: float = paragraph_stack[segment_id].x
            paragraph_y: float = paragraph_stack[segment_id].y
            left_boundary: float = paragraph_stack[segment_id].x0
            right_boundary: float = paragraph_stack[segment_id].x1
            paragraph_height: float = paragraph_stack[segment_id].y1 - paragraph_stack[segment_id].y0
            font_size: float = paragraph_stack[segment_id].size
            has_linebreak: bool = paragraph_stack[segment_id].brk
            character_stack: str = ""
            current_font: str = None
            line_index = 0
            text_x_position = paragraph_x
            next_font = current_font
            text_pointer = 0

            operation_values: list[dict] = []

            while text_pointer < len(translated_text):
                formula_match = re.match(r"\{\s*v([\d\s]+)\}", translated_text[text_pointer:], re.IGNORECASE)
                modifier_width = 0

                if formula_match:
                    text_pointer += len(formula_match.group(0))
                    try:
                        formula_id = int(formula_match.group(1).replace(" ", ""))
                        advance_width = formula_widths[formula_id]
                    except Exception:
                        continue
                    if formula_groups[formula_id][-1].get_text() and unicodedata.category(
                        formula_groups[formula_id][-1].get_text()[0]
                    ) in ["Lm", "Mn", "Sk"]:
                        modifier_width = formula_groups[formula_id][-1].width
                else:
                    current_character = translated_text[text_pointer]
                    next_font = None
                    try:
                        if (
                            next_font is None
                            and self.fontmap["tiro"].to_unichr(ord(current_character)) == current_character
                        ):
                            next_font = "tiro"
                    except Exception:
                        pass
                    if next_font is None:
                        next_font = self.noto_font_name
                    if next_font == self.noto_font_name:
                        advance_width = self.noto_font.char_lengths(current_character, font_size)[0]
                    else:
                        advance_width = self.fontmap[next_font].char_width(ord(current_character)) * font_size
                    text_pointer += 1

                if next_font != current_font or formula_match or paragraph_x + advance_width > right_boundary + 0.1 * font_size:
                    if character_stack:
                        operation_values.append(
                            {
                                "type": OperationType.TEXT,
                                "font": current_font,
                                "size": font_size,
                                "x": text_x_position,
                                "dy": 0,
                                "rtxt": encode_as_raw_string(current_font, character_stack),
                                "lidx": line_index,
                            }
                        )
                        character_stack = ""

                if has_linebreak and paragraph_x + advance_width > right_boundary + 0.1 * font_size:
                    paragraph_x = left_boundary
                    line_index += 1

                if formula_match:
                    vertical_fix = 0
                    if current_font is not None:
                        vertical_fix = formula_vertical_offsets[formula_id]
                    for formula_char in formula_groups[formula_id]:
                        formula_char_code = chr(formula_char.cid)
                        operation_values.append(
                            {
                                "type": OperationType.TEXT,
                                "font": self.fontid[formula_char.font],
                                "size": formula_char.size,
                                "x": paragraph_x + formula_char.x0 - formula_groups[formula_id][0].x0,
                                "dy": vertical_fix + formula_char.y0 - formula_groups[formula_id][0].y0,
                                "rtxt": encode_as_raw_string(self.fontid[formula_char.font], formula_char_code),
                                "lidx": line_index,
                            }
                        )
                    for formula_line in formula_lines_groups[formula_id]:
                        if formula_line.linewidth < 5:
                            operation_values.append(
                                {
                                    "type": OperationType.LINE,
                                    "x": formula_line.pts[0][0] + paragraph_x - formula_groups[formula_id][0].x0,
                                    "dy": formula_line.pts[0][1] + vertical_fix - formula_groups[formula_id][0].y0,
                                    "linewidth": formula_line.linewidth,
                                    "xlen": formula_line.pts[1][0] - formula_line.pts[0][0],
                                    "ylen": formula_line.pts[1][1] - formula_line.pts[0][1],
                                    "lidx": line_index,
                                }
                            )
                else:
                    if not character_stack:
                        text_x_position = paragraph_x
                        if paragraph_x == left_boundary and current_character == " ":
                            advance_width = 0
                        else:
                            character_stack += current_character
                    else:
                        character_stack += current_character

                advance_width -= modifier_width
                current_font = next_font
                paragraph_x += advance_width

            if character_stack:
                operation_values.append(
                    {
                        "type": OperationType.TEXT,
                        "font": current_font,
                        "size": font_size,
                        "x": text_x_position,
                        "dy": 0,
                        "rtxt": encode_as_raw_string(current_font, character_stack),
                        "lidx": line_index,
                    }
                )

            line_height = default_line_height

            while (line_index + 1) * font_size * line_height > paragraph_height and line_height >= 1:
                line_height -= 0.05

            for operation_value in operation_values:
                if operation_value["type"] == OperationType.TEXT:
                    operations_list.append(
                        generate_text_operation(
                            operation_value["font"],
                            operation_value["size"],
                            operation_value["x"],
                            operation_value["dy"] + paragraph_y - operation_value["lidx"] * font_size * line_height,
                            operation_value["rtxt"],
                        )
                    )
                elif operation_value["type"] == OperationType.LINE:
                    operations_list.append(
                        generate_line_operation(
                            operation_value["x"],
                            operation_value["dy"] + paragraph_y - operation_value["lidx"] * font_size * line_height,
                            operation_value["xlen"],
                            operation_value["ylen"],
                            operation_value["linewidth"],
                        )
                    )

        for global_line in global_lines_stack:
            if global_line.linewidth < 5:
                operations_list.append(
                    generate_line_operation(
                        global_line.pts[0][0],
                        global_line.pts[0][1],
                        global_line.pts[1][0] - global_line.pts[0][0],
                        global_line.pts[1][1] - global_line.pts[0][1],
                        global_line.linewidth,
                    )
                )

        pdf_operations = f"BT {''.join(operations_list)}ET "
        return pdf_operations


# ============================================================================
# High-Level Translation Functions
# ============================================================================

def translate_pdf_page_with_patch(
    input_stream: BinaryIO,
    pages: Optional[list[int]] = None,
    formula_font_pattern: str = "",
    formula_char_pattern: str = "",
    thread_count: int = 0,
    translated_document: Document = None,
    source_language: str = "",
    target_language: str = "",
    noto_font_name: str = "",
    noto_font: Font = None,
    progress_callback: object = None,
    cancellation_event: asyncio.Event = None,
    layout_model: DocumentLayoutModel = None,
    ignore_cache: bool = False,
    **additional_kwargs: Any,
) -> None:
    resource_manager = PDFResourceManager()
    layout_map = {}
    translation_device = TranslationConverter(
        resource_manager,
        formula_font_pattern,
        formula_char_pattern,
        thread_count,
        layout_map,
        source_language,
        target_language,
        noto_font_name,
        noto_font,
        ignore_cache,
    )

    assert translation_device is not None
    object_patch_dict = {}
    page_interpreter = ExtendedPDFPageInterpreter(resource_manager, translation_device, object_patch_dict)

    if pages:
        total_pages = len(pages)
    else:
        total_pages = translated_document.page_count

    pdf_parser = PDFParser(input_stream)
    pdf_document = PDFDocument(pdf_parser)

    with tqdm(total=total_pages) as progress_bar:
        for page_number, pdf_page in enumerate(PDFPage.create_pages(pdf_document)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("Translation task cancelled")
            if pages and (page_number not in pages):
                continue
            progress_bar.update()
            if progress_callback:
                progress_callback(progress_bar)

            pdf_page.pageno = page_number
            page_pixmap = translated_document[pdf_page.pageno].get_pixmap()
            page_image = np.frombuffer(page_pixmap.samples, np.uint8).reshape(
                page_pixmap.height, page_pixmap.width, 3
            )[:, :, ::-1]

            detected_layout = layout_model.predict(page_image, image_size=int(page_pixmap.height / 32) * 32)[0]
            layout_box = np.ones((page_pixmap.height, page_pixmap.width))
            box_height, box_width = layout_box.shape

            formula_classes = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]

            for detection_index, detected_box in enumerate(detected_layout.boxes):
                if detected_layout.names[int(detected_box.class_id)] not in formula_classes:
                    box_x0, box_y0, box_x1, box_y1 = detected_box.xyxy.squeeze()
                    box_x0, box_y0, box_x1, box_y1 = (
                        np.clip(int(box_x0 - 1), 0, box_width - 1),
                        np.clip(int(box_height - box_y1 - 1), 0, box_height - 1),
                        np.clip(int(box_x1 + 1), 0, box_width - 1),
                        np.clip(int(box_height - box_y0 + 1), 0, box_height - 1),
                    )
                    layout_box[box_y0:box_y1, box_x0:box_x1] = detection_index + 2

            for detection_index, detected_box in enumerate(detected_layout.boxes):
                if detected_layout.names[int(detected_box.class_id)] in formula_classes:
                    box_x0, box_y0, box_x1, box_y1 = detected_box.xyxy.squeeze()
                    box_x0, box_y0, box_x1, box_y1 = (
                        np.clip(int(box_x0 - 1), 0, box_width - 1),
                        np.clip(int(box_height - box_y1 - 1), 0, box_height - 1),
                        np.clip(int(box_x1 + 1), 0, box_width - 1),
                        np.clip(int(box_height - box_y0 + 1), 0, box_height - 1),
                    )
                    layout_box[box_y0:box_y1, box_x0:box_x1] = 0

            layout_map[pdf_page.pageno] = layout_box
            pdf_page.page_xref = translated_document.get_new_xref()
            translated_document.update_object(pdf_page.page_xref, "<<>>")
            translated_document.update_stream(pdf_page.page_xref, b"")
            translated_document[pdf_page.pageno].set_contents(pdf_page.page_xref)
            page_interpreter.process_page(pdf_page)

    translation_device.close()
    return object_patch_dict


def translate_pdf_stream(
    pdf_bytes: bytes,
    pages: Optional[list[int]] = None,
    source_language: str = "",
    target_language: str = "",
    thread_count: int = 0,
    formula_font_pattern: str = "",
    formula_char_pattern: str = "",
    progress_callback: object = None,
    cancellation_event: asyncio.Event = None,
    layout_model: DocumentLayoutModel = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **additional_kwargs: Any,
):
    font_list = [("tiro", None)]

    font_path = get_font_path_for_language(target_language.lower())
    noto_font_name = "noto"
    noto_font = Font(noto_font_name, font_path)
    font_list.append((noto_font_name, font_path))

    dual_document = Document(stream=pdf_bytes)
    byte_stream = io.BytesIO()
    dual_document.save(byte_stream)
    translated_document = Document(stream=byte_stream)
    page_count = translated_document.page_count

    font_id_map = {}
    for page in translated_document:
        for font_name, font_path in font_list:
            font_id_map[font_name] = page.insert_font(font_name, font_path)

    xref_length = translated_document.xref_length()
    for xref_number in range(1, xref_length):
        for resource_label in ["Resources/", ""]:
            try:
                font_resource = translated_document.xref_get_key(xref_number, f"{resource_label}Font")
                target_key_prefix = f"{resource_label}Font/"
                if font_resource[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_resource[1]).group(1)
                    xref_number = int(resource_xref_id)
                    font_resource = ("dict", translated_document.xref_object(xref_number))
                    target_key_prefix = ""

                if font_resource[0] == "dict":
                    for font_name, font_path in font_list:
                        target_key = f"{target_key_prefix}{font_name}"
                        font_exists = translated_document.xref_get_key(xref_number, target_key)
                        if font_exists[0] == "null":
                            translated_document.xref_set_key(
                                xref_number,
                                target_key,
                                f"{font_id_map[font_name]} 0 R",
                            )
            except Exception:
                pass

    byte_stream = io.BytesIO()

    translated_document.save(byte_stream)
    object_patch_dict: dict = translate_pdf_page_with_patch(byte_stream, **locals())

    for object_id, new_operations in object_patch_dict.items():
        translated_document.update_stream(object_id, new_operations.encode())

    dual_document.insert_file(translated_document)
    for page_id in range(page_count):
        dual_document.move_page(page_count + page_id, page_id * 2 + 1)

    if not skip_subset_fonts:
        translated_document.subset_fonts(fallback=True)
        dual_document.subset_fonts(fallback=True)

    return (
        translated_document.write(deflate=True, garbage=3, use_objstms=1),
        dual_document.write(deflate=True, garbage=3, use_objstms=1),
    )


def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    source_language: str = "",
    target_language: str = "",
    formula_font_pattern: str = "",
    formula_char_pattern: str = "",
    progress_callback: object = None,
    cancellation_event: asyncio.Event = None,
    layout_model: DocumentLayoutModel = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **additional_kwargs: Any,
):
    """
    Main translation function.

    Thread count is automatically determined based on CPU cores (no user input needed).
    """
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = [
        file_path
        for file_path in files
        if not file_path.startswith("http://")
        and not file_path.startswith("https://")
        and not os.path.exists(file_path)
    ]

    if missing_files:
        raise PDFValueError(f"Some files do not exist: {missing_files}")

    # Automatically determine optimal thread count based on CPU cores
    thread_count = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {thread_count} threads for translation (auto-detected from {multiprocessing.cpu_count()} CPU cores)")

    result_files = []

    for file_path in files:
        with open(file_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        local_vars = locals()
        del local_vars["file_path"]
        mono_pdf_bytes, dual_pdf_bytes = translate_pdf_stream(pdf_bytes, **local_vars)

        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        mono_output_path = Path(output) / f"{filename_without_extension}-mono.pdf"
        dual_output_path = Path(output) / f"{filename_without_extension}-dual.pdf"

        with open(mono_output_path, "wb") as mono_file:
            mono_file.write(mono_pdf_bytes)

        with open(dual_output_path, "wb") as dual_file:
            dual_file.write(dual_pdf_bytes)

        result_files.append((str(mono_output_path), str(dual_output_path)))

    return result_files


# Alias for backward compatibility
OnnxModel = DocumentLayoutModel


# ============================================================================
# Main Entry Point for Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Backend module loaded successfully!")
    print("This module provides PDF translation functionality using NLLB-200.")
    print("All imports are from original application - no new dependencies!")
    print("All variables now have meaningful, self-documenting names!")
