from dataiku.customwebapp import *
from flask import request, jsonify, send_from_directory
from pathlib import Path
import logging

# Import your main backend logic
from pdf_backend import translate, OnnxModel  

app = get_webapp_container_flask_app()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_instance = None

def ensure_model_loaded():
    """Load ONNX model once (only when needed)."""
    global _model_instance
    if _model_instance is None:
        logger.info("Loading ONNX model...")
        _model_instance = OnnxModel.from_pretrained()
        logger.info("Model loaded successfully.")
    return _model_instance


@app.route("/translate", methods=["POST"])
def run_translate():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        lang_in = request.form.get("lang_in", "en")
        lang_out = request.form.get("lang_out", "zh")

        uploads_dir = Path("uploads")
        outputs_dir = Path("outputs")
        uploads_dir.mkdir(exist_ok=True)
        outputs_dir.mkdir(exist_ok=True)

        file_path = uploads_dir / file.filename
        file.save(file_path)

        model = ensure_model_loaded()

        logger.info(f"Translating {file_path} from {lang_in} → {lang_out}")
        results = translate(
            files=[str(file_path)],
            output=str(outputs_dir),
            source_language=lang_in,
            target_language=lang_out,
            layout_model=model,
        )

        if not results:
            return jsonify({"error": "No results generated"}), 500

        mono, dual = results[-1]
        return jsonify({
            "message": "Translation successful",
            "mono": Path(mono).name,
            "dual": Path(dual).name
        })
    except Exception as e:
        logger.exception("Translation error")
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    outputs_dir = Path("outputs")
    file_path = outputs_dir / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(outputs_dir, filename, as_attachment=True)
