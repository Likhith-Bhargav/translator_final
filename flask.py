# ============================================================================
# Flask Web Application for Dataiku Integration
# ============================================================================

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Global model instance (loaded once on startup)
_global_model: Optional[OnnxModel] = None


def get_model():
    """Get or initialize the ONNX model (singleton pattern)."""
    global _global_model
    if _global_model is None:
        logger.info("Loading ONNX model...")
        try:
            _global_model = OnnxModel.from_pretrained()
            logger.info("ONNX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    return _global_model


@app.route("/translate", methods=["POST"])
def translate_endpoint():
    """
    Handle PDF translation request from frontend.

    Expects:
        - file: PDF file (multipart/form-data)
        - lang_in: Source language code (e.g., 'en', 'hi', 'fr')
        - lang_out: Target language code (e.g., 'zh', 'en', 'hi')

    Returns:
        JSON: {"mono": "filename-mono.pdf", "dual": "filename-dual.pdf"}
        Or error: {"error": "error message"}
    """
    try:
        # Read uploaded file
        file = request.files.get("file")
        lang_in = request.form.get("lang_in")
        lang_out = request.form.get("lang_out")

        # Validate inputs
        if not file:
            logger.error("No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        if not lang_in or not lang_out:
            logger.error("Missing language parameters")
            return jsonify({"error": "Missing language parameters"}), 400

        logger.info(f"Translation request: {file.filename}, {lang_in} -> {lang_out}")

        # Try to use Dataiku managed folder, fall back to temp directory
        try:
            import dataiku
            temp_dir = dataiku.Folder("upload_folder").get_path()
            output_dir = dataiku.Folder("output_folder").get_path()
            logger.info(f"Using Dataiku folders: {temp_dir}, {output_dir}")
        except Exception as e:
            logger.warning(f"Dataiku folders not available, using temp directory: {e}")
            temp_dir = tempfile.gettempdir()
            output_dir = tempfile.gettempdir()

        # Ensure directories exist
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save uploaded file temporarily
        input_filename = os.path.join(temp_dir, file.filename)
        file.save(input_filename)
        logger.info(f"File saved: {input_filename}")

        # Load ONNX model
        model = get_model()

        # Perform translation using the translate() function
        logger.info("Starting translation...")
        result_files = translate(
            files=[input_filename],
            output=output_dir,
            pages=None,  # Translate all pages
            lang_in=lang_in,
            lang_out=lang_out,
            vfont="",
            vchar="",
            model=model,
            skip_subset_fonts=False,
            ignore_cache=False,
        )

        # Extract output file paths
        mono_path, dual_path = result_files[0]

        # Get just the filenames
        mono_filename = os.path.basename(mono_path)
        dual_filename = os.path.basename(dual_path)

        logger.info(f"Translation completed successfully")
        logger.info(f"Mono output: {mono_filename}")
        logger.info(f"Dual output: {dual_filename}")

        # Clean up input file
        try:
            os.remove(input_filename)
        except Exception as e:
            logger.warning(f"Could not delete input file: {e}")

        # Return JSON response to frontend
        return jsonify({
            "mono": mono_filename,
            "dual": dual_filename
        })

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>", methods=["GET"])
def download_endpoint(filename):
    """
    Handle file download request.

    Args:
        filename: Name of the file to download

    Returns:
        PDF file for download
    """
    try:
        # Try to use Dataiku managed folder, fall back to temp directory
        try:
            import dataiku
            output_dir = dataiku.Folder("output_folder").get_path()
        except Exception:
            output_dir = tempfile.gettempdir()

        file_path = os.path.join(output_dir, filename)

        if not os.path.exists(file_path):
            logger.error(f"File not found: {filename}")
            return jsonify({"error": "File not found"}), 404

        logger.info(f"Serving file: {filename}")
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Dataiku will automatically use the 'app' object above
# No need to run app.run() - Dataiku handles the server
# ============================================================================

