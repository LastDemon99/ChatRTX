from core.whisper.trt_whisper import WhisperTRTLLM, decode_audio_file
from core.config.models import WhisperConfig

class Whisper:
    def __init__(self):
        self.model: WhisperTRTLLM = None
        self.language: str = "English"

    def decode_audio_file(self, new_file_path, asr_assets_path, language=None):
        if language is None:
            language = self.language
        decode_audio_file(new_file_path, self.model, language=language, mel_filters_dir=asr_assets_path)

    def is_loaded(self):
        return self.model is not None

    def load(self, whisper_config: WhisperConfig, language="English"):
        self.model = WhisperTRTLLM(whisper_config.asr_engine_path, assets_dir=whisper_config.asr_assets_path)
        self.language: language

    def unload(self):
        self.model.unload_model()
        del self.model
        self.model = None