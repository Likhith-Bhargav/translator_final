"""
Gradio UI for PDF Document Translation
This module provides a web interface for translating PDF documents using NLLB-200.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from backend import translate, OnnxModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_model_instance: Optional[OnnxModel] = None


def ensure_model_loaded() -> OnnxModel:
    """Load the ONNX model if not already loaded."""
    global _model_instance
    if _model_instance is None:
        logger.info("Loading ONNX layout detection model...")
        _model_instance = OnnxModel.from_pretrained()
        logger.info("Model loaded successfully!")
    return _model_instance


def run_translate(
    files: List,
    lang_in: str,
    lang_out: str,
    pages: str,
    output_dir: str,
    skip_subset_fonts: bool,
    ignore_cache: bool,
) -> Tuple[str, str, str]:
    """
    Run PDF translation with the provided parameters.
    Thread count is automatically determined from CPU cores.
    """
    try:
        # Ensure model is loaded
        model = ensure_model_loaded()

        # Normalize gradio file inputs to filesystem paths
        file_paths: List[str] = [f.name for f in files] if files else []

        if not file_paths:
            return "Error: No files provided", None, None

        # Parse page ranges
        pages_list: Optional[list[int]] = None
        if pages.strip():
            try:
                parts = [p.strip() for p in pages.split(",") if p.strip()]
                out: list[int] = []
                for part in parts:
                    if "-" in part:
                        a, b = part.split("-", 1)
                        out.extend(list(range(int(a), int(b) + 1)))
                    else:
                        out.append(int(part))
                pages_list = out
            except Exception as e:
                return f"Error parsing page range: {e}", None, None

        # Create output directory
        output = Path(output_dir or ".").resolve()
        output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting translation: {lang_in} -> {lang_out}")
        logger.info(f"Files: {file_paths}")
        logger.info(f"Output: {output}")

        # Run translation (thread count is auto-detected inside translate function)
        results = translate(
            files=file_paths,
            output=str(output),
            pages=pages_list,
            lang_in=lang_in,
            lang_out=lang_out,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            model=model,
        )

        # Return results
        if results:
            last = results[-1]
            mono, dual = last
            logger.info(f"Translation completed! Files saved to: {output}")
            return (str(output), mono, dual)
        else:
            return "Error: No results generated", None, None

    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return f"Error: {str(e)}", None, None


def build_ui() -> gr.Blocks:
    """Build the Gradio UI interface."""

    # Extended language list (20+ languages)
    languages = [
        ("English", "en"),
        ("Chinese (Simplified)", "zh"),
        ("Chinese (Traditional)", "zh-tw"),
        ("Spanish", "es"),
        ("French", "fr"),
        ("German", "de"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("Arabic", "ar"),
        ("Russian", "ru"),
        ("Portuguese", "pt"),
        ("Italian", "it"),
        ("Hindi", "hi"),
        ("Indonesian", "id"),
        ("Vietnamese", "vi"),
        ("Malay", "ms"),
        ("Thai", "th"),
        ("Turkish", "tr"),
        ("Polish", "pl"),
        ("Dutch", "nl"),
        ("Swedish", "sv"),
        ("Danish", "da"),
        ("Finnish", "fi"),
        ("Norwegian", "no"),
        ("Czech", "cs"),
        ("Ukrainian", "uk"),
        ("Romanian", "ro"),
        ("Greek", "el"),
    ]

    lang_choices = [code for _, code in languages]
    lang_labels = {code: label for label, code in languages}

    with gr.Blocks(
        title="PDF Document Translator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # PDF Document Translator (NLLB-200)

        Translate PDF documents while preserving formulas, figures, tables, and layout.
        Uses Facebook's NLLB-200 Distilled 600M model for high-quality offline translation.

        ### Features:
        - **Preserves Layout**: Maintains original document structure
        - **Formula Protection**: Mathematical formulas remain unchanged
        - **Bilingual Output**: Generates both translated-only and side-by-side versions
        - **Offline Translation**: No internet connection required
        - **200+ Languages**: Supports extensive language pairs
        - **Auto-Optimized**: Thread count automatically detected from your CPU
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Input Configuration")

                files = gr.File(
                    label="PDF Files",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )

                with gr.Row():
                    lang_in = gr.Dropdown(
                        choices=lang_choices,
                        value="en",
                        label="Source Language",
                        info="Language of the input document",
                    )
                    lang_out = gr.Dropdown(
                        choices=lang_choices,
                        value="zh",
                        label="Target Language",
                        info="Language to translate to",
                    )

                with gr.Row():
                    pages = gr.Textbox(
                        value="",
                        label="Page Range (Optional)",
                        placeholder="e.g., 1-5,8,10-15",
                        info="Leave empty to translate all pages",
                    )
                    output_dir = gr.Textbox(
                        value="output",
                        label="Output Directory",
                        placeholder="output",
                    )

                gr.Markdown("### Advanced Options")

                with gr.Row():
                    skip_subset_fonts = gr.Checkbox(
                        value=False,
                        label="Skip Font Subsetting",
                        info="Larger file size but faster processing",
                    )
                    ignore_cache = gr.Checkbox(
                        value=False,
                        label="Ignore Translation Cache",
                        info="Force retranslation of all text",
                    )

                run_btn = gr.Button(
                    "Translate Document",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Translation Guide")
                gr.Markdown("""
                **How to use:**
                1. Upload one or more PDF files
                2. Select source and target languages
                3. (Optional) Specify page ranges
                4. Click "Translate Document"
                5. Download the translated PDFs

                **Output Files:**
                - **Mono PDF**: Translated text only
                - **Dual PDF**: Original and translation side-by-side

                **Tips:**
                - Use page ranges for large documents
                - Enable cache for repeated translations
                - Thread count is auto-optimized
                - Font subsetting reduces file size

                **Supported Content:**
                - ✅ Text paragraphs
                - ✅ Mathematical formulas
                - ✅ Tables and figures
                - ✅ Complex layouts
                - ✅ Multi-column documents

                **Performance:**
                - Translation speed is automatically
                  optimized based on your CPU
                - Multi-threading is handled internally
                - No configuration needed!
                """)

        gr.Markdown("### Output")

        with gr.Row():
            out_dir = gr.Textbox(
                label="Saved to Directory",
                interactive=False,
            )

        with gr.Row():
            mono_pdf = gr.File(
                label="Translated PDF (Mono)",
                interactive=False,
            )
            dual_pdf = gr.File(
                label="Bilingual PDF (Dual)",
                interactive=False,
            )

        gr.Markdown("""
        ---
        ### About

        This application uses:
        - **NLLB-200 Distilled 600M**: Meta's No Language Left Behind model
        - **DocLayout YOLO**: Document structure detection
        - **PyMuPDF**: PDF processing and manipulation
        - **Gradio**: Web interface framework

        All processing is done locally on your machine. No data is sent to external servers.
        Translation speed is automatically optimized based on your CPU cores.
        """)

        # Connect the button to the function (no thread_count parameter)
        run_btn.click(
            fn=run_translate,
            inputs=[
                files,
                lang_in,
                lang_out,
                pages,
                output_dir,
                skip_subset_fonts,
                ignore_cache,
            ],
            outputs=[out_dir, mono_pdf, dual_pdf],
        )

    return demo


def main() -> None:
    """Launch the Gradio UI."""
    logger.info("Starting PDF Translator UI...")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
