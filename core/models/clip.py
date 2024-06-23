from core.whisper.trt_whisper import WhisperTRTLLM
from core.clip.CLIP import CLIPEmbeddingStorageEngine
from transformers import CLIPProcessor, CLIPModel

from core.config.models import ModelConfig

class Clip:
    def __init__(self):
        self.model: WhisperTRTLLM = None
        self.processor: CLIPProcessor = None
        self.engine: CLIPEmbeddingStorageEngine = None
        self.model_path = None

    def is_loaded(self):
        return self.model is not None

    def load(self, model_config: ModelConfig, data_dir):
        self.model = CLIPModel.from_pretrained(model_config["model_path"]).to('cuda')
        self.processor = CLIPProcessor.from_pretrained(model_config["model_path"])
        self.generate_engine(data_dir)
        self.model_path = model_config["model_path"]

    def generate_engine(self, data_dir, force_rewrite=False):
        self.engine = CLIPEmbeddingStorageEngine(data_dir, self.model_path, self.model, self.processor)
        self.engine.create_nodes(force_rewrite)
        self.engine.initialize_index(force_rewrite)

    def unload(self):
        del self.model
        del self.processor
        del self.engine
        self.model = None
        self.processor = None
        self.engine = None