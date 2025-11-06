"""
PDF Document Translation Backend using NLLB-200
Complete standalone implementation - no external module dependencies
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
# Configuration Manager (from original config.py)
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
            with self._config_path.open("r", encoding="utf-8") as f:
                self._config_data = json.load(f)

    def _save_config(self):
        with self._lock:
            cleaned_data = self._remove_circular_references(self._config_data)
            with self._config_path.open("w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    def _remove_circular_references(self, obj, seen=None):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        if isinstance(obj, dict):
            return {k: self._remove_circular_references(v, seen) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._remove_circular_references(i, seen) for i in obj]
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
# Translation Cache (from original cache.py)
# ============================================================================

db = SqliteDatabase(None)


class _TranslationCache(Model):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    translate_engine_params = TextField()
    original_text = TextField()
    translation = TextField()

    class Meta:
        database = db
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
                k: TranslationCache._sort_dict_recursively(v)
                for k in sorted(obj.keys())
                for v in [obj[k]]
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
        except Exception as e:
            logger.debug(f"Error setting cache: {e}")


def init_cache_db():
    """Initialize the translation cache database."""
    cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "pdf2zh")
    os.makedirs(cache_folder, exist_ok=True)
    cache_db_path = os.path.join(cache_folder, "cache.v1.db")
    db.init(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    db.create_tables([_TranslationCache], safe=True)


# Initialize cache on module load
init_cache_db()


# ============================================================================
# NLLB Translator (from original nllb_direct.py)
# ============================================================================

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
        except Exception as e:
            logger.error(f"Failed to load NLLB model from {self.model_path}: {e}")
            raise

    def _map_lang(self, code: str) -> str:
        """Map common short codes to NLLB language tags."""
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
                input_length = inputs["input_ids"].shape[1]
                max_output_len = min(512, input_length * 2)

                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt),
                    max_length=max_output_len,
                    num_beams=1,
                    length_penalty=0.8,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"NLLB translation failed: {e}")
            raise


_nllb_direct: Optional[NLLBDirect] = None


def get_nllb_direct(model_path: Optional[str] = None) -> NLLBDirect:
    global _nllb_direct
    if _nllb_direct is None:
        _nllb_direct = NLLBDirect(model_path)
    return _nllb_direct


class NLLBTranslator:
    """NLLB translator with caching support."""
    name = "nllb"

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        model_path: Optional[str] = None,
        ignore_cache: bool = False,
    ):
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model_path = model_path
        self.ignore_cache = ignore_cache
        self._passthrough = self.lang_in == self.lang_out

        if not self._passthrough:
            self.nllb_direct = get_nllb_direct(self.model_path)

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model_path or "default",
            },
        )

    def translate(self, text: str) -> str:
        if self._passthrough:
            return text

        if not self.ignore_cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        try:
            translated_text = self.nllb_direct.translate_text(
                text, self.lang_in, self.lang_out
            )
            self.cache.set(text, translated_text)
            return translated_text
        except Exception:
            raise


# ============================================================================
# Document Layout Detection (from original doclayout.py)
# ============================================================================

class YoloBox:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, data):
        self.xyxy = data[:4]
        self.conf = data[-2]
        self.cls = data[-1]


class YoloResult:
    """Helper class to store detection results from ONNX model."""

    def __init__(self, boxes, names):
        self.boxes = [YoloBox(data=d) for d in boxes]
        self.boxes.sort(key=lambda x: x.conf, reverse=True)
        self.names = names


class OnnxModel:
    """ONNX-based document layout detection model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        model = onnx.load(model_path)
        metadata = {d.key: d.value for d in model.metadata_props}
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])
        self.model = onnxruntime.InferenceSession(model.SerializeToString())

    @staticmethod
    def from_pretrained():
        model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/doclayout_yolo_docstructbench_imgsz1024.onnx"
        return OnnxModel(model_path)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, new_shape):
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        new_h, new_w = new_shape
        r = min(new_h / h, new_w / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (new_w - resized_w) % self.stride
        pad_h = (new_h - resized_h) % self.stride
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    def predict(self, image, imgsz=1024, **kwargs):
        orig_h, orig_w = image.shape[:2]
        pix = self.resize_and_pad_image(image, new_shape=imgsz)
        pix = np.transpose(pix, (2, 0, 1))
        pix = np.expand_dims(pix, axis=0)
        pix = pix.astype(np.float32) / 255.0
        new_h, new_w = pix.shape[2:]

        preds = self.model.run(None, {"images": pix})[0]
        preds = preds[preds[..., 4] > 0.25]
        preds[..., :4] = self.scale_boxes((new_h, new_w), preds[..., :4], (orig_h, orig_w))
        return [YoloResult(boxes=preds, names=self._names)]


# ============================================================================
# Font Management (from original high_level.py and local_assets.py)
# ============================================================================

def get_font_path(lang: str) -> str:
    """Get the appropriate font path for the target language."""
    lang = lang.lower()
    assets_dir = Path("/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/BabelDOC-Assets-main")
    fonts_dir = assets_dir / "fonts"

    # Language to font mapping
    if lang in ["zh", "zh-cn", "zh-hans", "zh-tw", "zh-hant"]:
        font_name = "SourceHanSerifCN-Regular.ttf"
    elif lang == "ja":
        font_name = "SourceHanSerifJP-Regular.ttf"
    elif lang == "ko":
        font_name = "SourceHanSerifKR-Regular.ttf"
    else:
        font_name = "GoNotoKurrent-Regular.ttf"

    font_path = fonts_dir / font_name
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    logger.info(f"Using font: {font_path}")
    return str(font_path)


# ============================================================================
# PDF Processing (from original pdfinterp.py)
# ============================================================================

class PDFPageInterpreterEx(PDFPageInterpreter):
    """Extended PDF page interpreter with object patching support."""

    def __init__(
        self, rsrcmgr: PDFResourceManager, device: PDFDevice, obj_patch: dict
    ) -> None:
        self.rsrcmgr = rsrcmgr
        self.device = device
        self.obj_patch = obj_patch

    def dup(self) -> "PDFPageInterpreterEx":
        return self.__class__(self.rsrcmgr, self.device, self.obj_patch)

    def init_resources(self, resources: Dict[object, object]) -> None:
        self.resources = resources
        self.fontmap: Dict[object, PDFFont] = {}
        self.fontid: Dict[PDFFont, object] = {}
        self.xobjmap = {}
        self.csmap: Dict[str, PDFColorSpace] = PREDEFINED_COLORSPACE.copy()
        if not resources:
            return

        def get_colorspace(spec: object) -> Optional[PDFColorSpace]:
            if isinstance(spec, list):
                name = literal_name(spec[0])
            else:
                name = literal_name(spec)
            if name == "ICCBased" and isinstance(spec, list) and len(spec) >= 2:
                return PDFColorSpace(name, stream_value(spec[1])["N"])
            elif name == "DeviceN" and isinstance(spec, list) and len(spec) >= 2:
                return PDFColorSpace(name, len(list_value(spec[1])))
            else:
                return PREDEFINED_COLORSPACE.get(name)

        for k, v in dict_value(resources).items():
            if k == "Font":
                for fontid, spec in dict_value(v).items():
                    objid = None
                    if isinstance(spec, PDFObjRef):
                        objid = spec.objid
                    spec = dict_value(spec)
                    self.fontmap[fontid] = self.rsrcmgr.get_font(objid, spec)
                    self.fontmap[fontid].descent = 0
                    self.fontid[self.fontmap[fontid]] = fontid
            elif k == "ColorSpace":
                for csid, spec in dict_value(v).items():
                    colorspace = get_colorspace(resolve1(spec))
                    if colorspace is not None:
                        self.csmap[csid] = colorspace
            elif k == "ProcSet":
                self.rsrcmgr.get_procset(list_value(v))
            elif k == "XObject":
                for xobjid, xobjstrm in dict_value(v).items():
                    self.xobjmap[xobjid] = xobjstrm

    def do_S(self) -> None:
        def is_black(color: Color) -> bool:
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
            and is_black(self.graphicstate.scolor)
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
            n = self.scs.ncomponents
        else:
            if settings.STRICT:
                raise PDFInterpreterError("No colorspace specified!")
            n = 1
        args = self.pop(n)
        self.graphicstate.scolor = args
        return args

    def do_scn(self) -> None:
        if self.ncs:
            n = self.ncs.ncomponents
        else:
            if settings.STRICT:
                raise PDFInterpreterError("No colorspace specified!")
            n = 1
        args = self.pop(n)
        self.graphicstate.ncolor = args
        return args

    def do_SC(self) -> None:
        return self.do_SCN()

    def do_sc(self) -> None:
        return self.do_scn()

    def do_Do(self, xobjid_arg: PDFStackT) -> None:
        xobjid = literal_name(xobjid_arg)
        try:
            xobj = stream_value(self.xobjmap[xobjid])
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError("Undefined xobject id: %r" % xobjid)
            return
        subtype = xobj.get("Subtype")
        if subtype is LITERAL_FORM and "BBox" in xobj:
            interpreter = self.dup()
            bbox = list_value(xobj["BBox"])
            matrix = list_value(xobj.get("Matrix", MATRIX_IDENTITY))
            xobjres = xobj.get("Resources")
            if xobjres:
                resources = dict_value(xobjres)
            else:
                resources = self.resources.copy()
            self.device.begin_figure(xobjid, bbox, matrix)
            ctm = mult_matrix(matrix, self.ctm)
            ops_base = interpreter.render_contents(resources, [xobj], ctm=ctm)
            self.ncs = interpreter.ncs
            self.scs = interpreter.scs
            try:
                self.device.fontid = interpreter.fontid
                self.device.fontmap = interpreter.fontmap
                ops_new = self.device.end_figure(xobjid)
                ctm_inv = np.linalg.inv(np.array(ctm[:4]).reshape(2, 2))
                np_version = np.__version__
                if np_version.split(".")[0] >= "2":
                    pos_inv = -np.asmatrix(ctm[4:]) * ctm_inv
                else:
                    pos_inv = -np.mat(ctm[4:]) * ctm_inv
                a, b, c, d = ctm_inv.reshape(4).tolist()
                e, f = pos_inv.tolist()[0]
                self.obj_patch[self.xobjmap[xobjid].objid] = (
                    f"q {ops_base}Q {a} {b} {c} {d} {e} {f} cm {ops_new}"
                )
            except Exception:
                pass
        elif subtype is LITERAL_IMAGE and "Width" in xobj and "Height" in xobj:
            self.device.begin_figure(xobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
            self.device.render_image(xobjid, xobj)
            self.device.end_figure(xobjid)
        else:
            pass

    def process_page(self, page: PDFPage) -> None:
        (x0, y0, x1, y1) = page.cropbox
        if page.rotate == 90:
            ctm = (0, -1, 1, 0, -y0, x1)
        elif page.rotate == 180:
            ctm = (-1, 0, 0, -1, x1, y1)
        elif page.rotate == 270:
            ctm = (0, 1, -1, 0, y1, -x0)
        else:
            ctm = (1, 0, 0, 1, -x0, -y0)
        self.device.begin_page(page, ctm)
        ops_base = self.render_contents(page.resources, page.contents, ctm=ctm)
        self.device.fontid = self.fontid
        self.device.fontmap = self.fontmap
        ops_new = self.device.end_page(page)
        self.obj_patch[page.page_xref] = (
            f"q {ops_base}Q 1 0 0 1 {x0} {y0} cm {ops_new}"
        )
        for obj in page.contents:
            self.obj_patch[obj.objid] = ""

    def render_contents(
        self,
        resources: Dict[object, object],
        streams: list,
        ctm: Matrix = MATRIX_IDENTITY,
    ) -> None:
        self.init_resources(resources)
        self.init_state(ctm)
        return self.execute(list_value(streams))

    def execute(self, streams: list) -> None:
        ops = ""
        try:
            parser = PDFContentParser(streams)
        except PSEOF:
            return
        while True:
            try:
                (_, obj) = parser.nextobject()
            except PSEOF:
                break
            if isinstance(obj, PSKeyword):
                name = keyword_name(obj)
                method = "do_%s" % name.replace("*", "_a").replace('"', "_w").replace(
                    "'", "_q"
                )
                if hasattr(self, method):
                    func = getattr(self, method)
                    nargs = func.__code__.co_argcount - 1
                    if nargs:
                        args = self.pop(nargs)
                        if len(args) == nargs:
                            func(*args)
                            if not (
                                name[0] == "T"
                                or name in ['"', "'", "EI", "MP", "DP", "BMC", "BDC"]
                            ):
                                p = " ".join(
                                    [
                                        (
                                            f"{x:f}"
                                            if isinstance(x, float)
                                            else str(x).replace("'", "")
                                        )
                                        for x in args
                                    ]
                                )
                                ops += f"{p} {name} "
                    else:
                        targs = func()
                        if targs is None:
                            targs = []
                        if not (name[0] == "T" or name in ["BI", "ID", "EMC"]):
                            p = " ".join(
                                [
                                    (
                                        f"{x:f}"
                                        if isinstance(x, float)
                                        else str(x).replace("'", "")
                                    )
                                    for x in targs
                                ]
                            )
                            ops += f"{p} {name} "
                elif settings.STRICT:
                    error_msg = "Unknown operator: %r" % name
                    raise PDFInterpreterError(error_msg)
            else:
                self.push(obj)
        return ops


# ============================================================================
# PDF Converter with Translation (from original converter.py)
# ============================================================================

class PDFConverterEx(PDFConverter):
    def __init__(self, rsrcmgr: PDFResourceManager) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid
        item.font = font
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk):
        self.y: float = y
        self.x: float = x
        self.x0: float = x0
        self.x1: float = x1
        self.y0: float = y0
        self.y1: float = y1
        self.size: float = size
        self.brk: bool = brk


class OpType(Enum):
    TEXT = "text"
    LINE = "line"


class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        noto_name: str = "",
        noto: Font = None,
        ignore_cache: bool = False,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator = NLLBTranslator(lang_in, lang_out, ignore_cache=ignore_cache)

    def receive_layout(self, ltpage: LTPage):
        is_hindi = (self.translator.lang_in or "").lower().startswith("hi")
        sstk: list[str] = []
        pstk: list[Paragraph] = []
        vbkt: int = 0
        vstk: list[LTChar] = []
        vlstk: list[LTLine] = []
        vfix: float = 0
        var: list[list[LTChar]] = []
        varl: list[list[LTLine]] = []
        varf: list[float] = []
        vlen: list[float] = []
        lstk: list[LTLine] = []
        xt: LTChar = None
        xt_cls: int = -1
        vmax: float = ltpage.width / 4
        ops: str = ""

        def vflag(font: str, char: str):
            if isinstance(font, bytes):
                try:
                    font = font.decode("utf-8")
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]
            if re.match(r"\(cid:", char):
                return True
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]
                        or ord(char[0]) in range(0x370, 0x400)
                    )
                ):
                    return True
            return False

        def to_grapheme_clusters(text):
            return regex.findall(r"\X", text)

        y_line_threshold = 10
        prev_char = None
        for child in ltpage:
            if isinstance(child, LTChar):
                if is_hindi:
                    if not sstk:
                        sstk.append("")
                        pstk.append(
                            Paragraph(
                                child.y0,
                                child.x0,
                                child.x0,
                                child.x0,
                                child.y0,
                                child.y1,
                                child.size,
                                False,
                            )
                        )
                    elif (
                        prev_char is not None
                        and abs(child.y0 - prev_char.y0) > y_line_threshold
                    ):
                        sstk.append("")
                        pstk.append(
                            Paragraph(
                                child.y0,
                                child.x0,
                                child.x0,
                                child.x0,
                                child.y0,
                                child.y1,
                                child.size,
                                False,
                            )
                        )
                    for cluster in to_grapheme_clusters(child.get_text()):
                        sstk[-1] += cluster
                    pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                    pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                    pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                    pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                    prev_char = child
                    xt = child
                    xt_cls = 0
                    continue
                cur_v = False
                layout = self.layout[ltpage.pageid]
                h, w = layout.shape
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(
                    int(child.y0), 0, h - 1
                )
                cls = layout[cy, cx]
                if child.get_text() == "•":
                    cls = 0
                if (
                    cls == 0
                    or (
                        cls == xt_cls
                        and len(sstk[-1].strip()) > 1
                        and child.size < pstk[-1].size * 0.79
                    )
                    or vflag(child.fontname, child.get_text())
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)
                ):
                    cur_v = True
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (
                    not cur_v
                    or cls != xt_cls
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)
                ):
                    if vstk:
                        if (
                            not cur_v
                            and cls == xt_cls
                            and child.x0 > max([vch.x0 for vch in vstk])
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                if not vstk:
                    if cls == xt_cls:
                        if child.x0 > xt.x1 + 1:
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:
                        sstk.append("")
                        pstk.append(
                            Paragraph(
                                child.y0,
                                child.x0,
                                child.x0,
                                child.x0,
                                child.y0,
                                child.y1,
                                child.size,
                                False,
                            )
                        )
                if not cur_v:
                    if (
                        child.size > pstk[-1].size or len(sstk[-1].strip()) == 1
                    ) and child.get_text() != " ":
                        pstk[-1].y -= child.size - pstk[-1].size
                        pstk[-1].size = child.size
                    if is_hindi:
                        for cluster in to_grapheme_clusters(child.get_text()):
                            sstk[-1] += cluster
                    else:
                        sstk[-1] += child.get_text()
                else:
                    if not vstk and cls == xt_cls and child.x0 > xt.x0:
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):
                pass
            elif isinstance(child, LTLine):
                layout = self.layout[ltpage.pageid]
                h, w = layout.shape
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(
                    int(child.y0), 0, h - 1
                )
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:
                    vlstk.append(child)
                else:
                    lstk.append(child)
            else:
                pass

        if vstk:
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)

        for id, v in enumerate(var):
            l = max([vch.x1 for vch in v]) - v[0].x0
            vlen.append(l)

        @retry(wait=wait_fixed(1))
        def worker(s: str):
            if not s.strip() or re.match(r"^\{v\d+\}$", s):
                return s
            try:
                new = self.translator.translate(s)
                return new
            except BaseException as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(e)
                else:
                    logger.exception(e, exc_info=False)
                raise e

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.thread)
        ) as executor:
            news = list(executor.map(worker, sstk))

        def raw_string(fcur: str, cstk: str):
            if fcur == self.noto_name:
                return "".join(
                    ["%04x" % self.noto.has_glyph(ord(c)) for c in cstk]
                )
            elif isinstance(self.fontmap[fcur], PDFCIDFont):
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        LANG_LINEHEIGHT_MAP = {
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
        default_line_height = LANG_LINEHEIGHT_MAP.get(
            self.translator.lang_out.lower(), 1.1
        )
        _x, _y = 0, 0
        ops_list = []

        def gen_op_txt(font, size, x, y, rtxt):
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        for id, new in enumerate(news):
            x: float = pstk[id].x
            y: float = pstk[id].y
            x0: float = pstk[id].x0
            x1: float = pstk[id].x1
            height: float = pstk[id].y1 - pstk[id].y0
            size: float = pstk[id].size
            brk: bool = pstk[id].brk
            cstk: str = ""
            fcur: str = None
            lidx = 0
            tx = x
            fcur_ = fcur
            ptr = 0

            ops_vals: list[dict] = []

            while ptr < len(new):
                vy_regex = re.match(r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE)
                mod = 0
                if vy_regex:
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue
                    if var[vid][-1].get_text() and unicodedata.category(
                        var[vid][-1].get_text()[0]
                    ) in ["Lm", "Mn", "Sk"]:
                        mod = var[vid][-1].width
                else:
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if (
                            fcur_ is None
                            and self.fontmap["tiro"].to_unichr(ord(ch)) == ch
                        ):
                            fcur_ = "tiro"
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name
                    if fcur_ == self.noto_name:
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                if fcur_ != fcur or vy_regex or x + adv > x1 + 0.1 * size:
                    if cstk:
                        ops_vals.append(
                            {
                                "type": OpType.TEXT,
                                "font": fcur,
                                "size": size,
                                "x": tx,
                                "dy": 0,
                                "rtxt": raw_string(fcur, cstk),
                                "lidx": lidx,
                            }
                        )
                        cstk = ""
                if brk and x + adv > x1 + 0.1 * size:
                    x = x0
                    lidx += 1
                if vy_regex:
                    fix = 0
                    if fcur is not None:
                        fix = varf[vid]
                    for vch in var[vid]:
                        vc = chr(vch.cid)
                        ops_vals.append(
                            {
                                "type": OpType.TEXT,
                                "font": self.fontid[vch.font],
                                "size": vch.size,
                                "x": x + vch.x0 - var[vid][0].x0,
                                "dy": fix + vch.y0 - var[vid][0].y0,
                                "rtxt": raw_string(self.fontid[vch.font], vc),
                                "lidx": lidx,
                            }
                        )
                    for l in varl[vid]:
                        if l.linewidth < 5:
                            ops_vals.append(
                                {
                                    "type": OpType.LINE,
                                    "x": l.pts[0][0] + x - var[vid][0].x0,
                                    "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                    "linewidth": l.linewidth,
                                    "xlen": l.pts[1][0] - l.pts[0][0],
                                    "ylen": l.pts[1][1] - l.pts[0][1],
                                    "lidx": lidx,
                                }
                            )
                else:
                    if not cstk:
                        tx = x
                        if x == x0 and ch == " ":
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod
                fcur = fcur_
                x += adv

            if cstk:
                ops_vals.append(
                    {
                        "type": OpType.TEXT,
                        "font": fcur,
                        "size": size,
                        "x": tx,
                        "dy": 0,
                        "rtxt": raw_string(fcur, cstk),
                        "lidx": lidx,
                    }
                )

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(
                        gen_op_txt(
                            vals["font"],
                            vals["size"],
                            vals["x"],
                            vals["dy"] + y - vals["lidx"] * size * line_height,
                            vals["rtxt"],
                        )
                    )
                elif vals["type"] == OpType.LINE:
                    ops_list.append(
                        gen_op_line(
                            vals["x"],
                            vals["dy"] + y - vals["lidx"] * size * line_height,
                            vals["xlen"],
                            vals["ylen"],
                            vals["linewidth"],
                        )
                    )

        for l in lstk:
            if l.linewidth < 5:
                ops_list.append(
                    gen_op_line(
                        l.pts[0][0],
                        l.pts[0][1],
                        l.pts[1][0] - l.pts[0][0],
                        l.pts[1][1] - l.pts[0][1],
                        l.linewidth,
                    )
                )

        ops = f"BT {''.join(ops_list)}ET "
        return ops


# ============================================================================
# High-Level Translation Functions (from original high_level.py)
# ============================================================================

def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    ignore_cache: bool = False,
    **kwarg: Any,
) -> None:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        noto_name,
        noto,
        ignore_cache,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    with tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("task cancelled")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            page.page_xref = doc_zh.get_new_xref()
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    device.close()
    return obj_patch


def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = get_font_path(lang_out.lower())
    noto_name = "noto"
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:
            try:
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""

                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    obj_patch: dict = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)
    if not skip_subset_fonts:
        doc_zh.subset_fonts(fallback=True)
        doc_en.subset_fonts(fallback=True)
    return (
        doc_zh.write(deflate=True, garbage=3, use_objstms=1),
        doc_en.write(deflate=True, garbage=3, use_objstms=1),
    )


def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **kwarg: Any,
):
    """
    Main translation function.

    Thread count is automatically determined based on CPU cores (no user input needed).
    """
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = [
        f
        for f in files
        if not f.startswith("http://")
        and not f.startswith("https://")
        and not os.path.exists(f)
    ]

    if missing_files:
        raise PDFValueError(f"Some files do not exist: {missing_files}")

    # Automatically determine optimal thread count based on CPU cores
    thread = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {thread} threads for translation (auto-detected from {multiprocessing.cpu_count()} CPU cores)")

    result_files = []

    for file in files:
        doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        l = locals()
        del l["file"]
        s_mono, s_dual = translate_stream(s_raw, **l)

        filename = os.path.splitext(os.path.basename(file))[0]
        file_mono = Path(output) / f"{filename}-mono.pdf"
        file_dual = Path(output) / f"{filename}-dual.pdf"
        doc_mono = open(file_mono, "wb")
        doc_dual = open(file_dual, "wb")
        doc_mono.write(s_mono)
        doc_dual.write(s_dual)
        doc_mono.close()
        doc_dual.close()
        result_files.append((str(file_mono), str(file_dual)))

    return result_files


# ============================================================================
# Main Entry Point for Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Backend module loaded successfully!")
    print("This module provides PDF translation functionality using NLLB-200.")
    print("All imports are from original application - no new dependencies!")
