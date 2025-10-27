class M2M100Translator(BaseTranslator):
    name = "m2m100"

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
            from pdf2zh.m2m100_direct import get_m2m100_direct
            self.m2m100_direct = get_m2m100_direct()
        except ImportError:
            logger.warning(
                "m2m100_direct module not available, if you want to use mBART translator, please ensure transformers and torch are installed. If you don't use mBART translator, you can safely ignore this warning."
            )
            raise

    def translate(self, text: str, ignore_cache: bool = False):
        if getattr(self, "_passthrough", False):
            return text
        # Translate using direct implementation
        from pdf2zh.m2m100_direct import translate_text

        try:
            translated_text = translate_text(text, self.lang_in, self.lang_out)
            return translated_text
        except ValueError as e:
            if "not supported by M2M100" in str(e):
                logger.error(
                    f"Language pair {self.lang_in}->{self.lang_out} not supported by mBART. "
                    f"Supported languages: {list(self.m2m100_direct.get_supported_languages().keys()) if hasattr(self, 'm2m100_direct') else 'N/A'}"
                )
                # Return original text as fallback
                return text
            elif "model not found" in str(e):
                logger.error(
                    "mBART model not found locally. "
                    "Please ensure the model is downloaded to the correct path and update the model path in m2m100_direct.py"
                )
                # Return original text as fallback
                return text
            else:
                raise
