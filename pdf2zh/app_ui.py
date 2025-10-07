import gradio as gr
from pathlib import Path
from typing import List, Optional, Tuple

from pdf2zh.high_level import translate
from pdf2zh.doclayout import OnnxModel, ModelInstance


def _ensure_model_loaded() -> None:
    if ModelInstance.value is None:
        ModelInstance.value = OnnxModel.load_available()


def run_translate(
    files: List,
    lang_in: str,
    lang_out: str,
    pages: str,
    output_dir: str,
    skip_subset_fonts: bool,
    ignore_cache: bool,
) -> Tuple[str, str, str]:
    _ensure_model_loaded()

    # Normalize gradio file inputs to filesystem paths
    file_paths: List[str] = [f.name for f in files] if files else []
    pages_list: Optional[list[int]] = None
    if pages.strip():
        # Accept formats like: 1-3,5,8
        parts = [p.strip() for p in pages.split(",") if p.strip()]
        out: list[int] = []
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(list(range(int(a), int(b) + 1)))
            else:
                out.append(int(part))
        pages_list = out

    output = Path(output_dir or ".").resolve()
    output.mkdir(parents=True, exist_ok=True)

    results = translate(
        files=file_paths,
        output=str(output),
        pages=pages_list,
        lang_in=lang_in,
        lang_out=lang_out,
        service="argos",
        skip_subset_fonts=skip_subset_fonts,
        ignore_cache=ignore_cache,
        model=ModelInstance.value,
    )

    # Return up to the last job's mono/dual
    last = results[-1] if results else ("", "")
    mono, dual = last
    return (str(output), mono, dual)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PDFMathTranslate - Lightweight") as demo:
        gr.Markdown("""
        ### PDFMathTranslate (Lightweight)
        Offline PDF translation that preserves formulas and layout. Uses Argos Translate.
        """)

        with gr.Row():
            files = gr.File(label="PDF files", file_types=[".pdf"], file_count="multiple", type="filepath")

        with gr.Row():
            lang_in = gr.Textbox(value="en", label="Source language (e.g. en)")
            lang_out = gr.Dropdown(
                choices=["zh", "zh-TW", "en", "ja", "ko", "fr", "de", "es", "it", "ru", "hi"],
                value="zh",
                label="Target language",
            )

        with gr.Row():
            pages = gr.Textbox(value="", label="Pages (e.g. 1-3,5)")
            output_dir = gr.Textbox(value="output", label="Output directory")

        with gr.Row():
            skip_subset_fonts = gr.Checkbox(value=False, label="Skip font subsetting")
            ignore_cache = gr.Checkbox(value=False, label="Ignore translation cache")

        run_btn = gr.Button("Translate")

        with gr.Row():
            out_dir = gr.Textbox(label="Saved to", interactive=False)
        with gr.Row():
            mono_pdf = gr.File(label="Mono PDF")
            dual_pdf = gr.File(label="Dual PDF")

        run_btn.click(
            fn=run_translate,
            inputs=[files, lang_in, lang_out, pages, output_dir, skip_subset_fonts, ignore_cache],
            outputs=[out_dir, mono_pdf, dual_pdf],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()


