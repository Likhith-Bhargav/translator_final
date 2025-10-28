
class NLLBTranslator(BaseTranslator):
    name = "nllb"

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        # Passthrough mode for same-language translation (e.g., en->en)
        self._passthrough = self.lang_in == self.lang_out
        if self._passthrough:
            return

        try:
            from pdf2zh.nllb_direct import get_nllb_direct
            # model contains optional local path supplied via service string
            self.nllb_direct = get_nllb_direct(self.model)
        except ImportError:
            logger.warning(
                "nllb_direct module not available. Ensure transformers and torch are installed."
            )
            raise

    def translate(self, text: str, ignore_cache: bool = False):
        if getattr(self, "_passthrough", False):
            return text
        # Translate using direct implementation
        from pdf2zh.nllb_direct import translate_text

        try:
            translated_text = translate_text(text, self.lang_in, self.lang_out, self.model)
            return translated_text
        except Exception:
            # No fallbacks allowed per requirements; re-raise
            raise
