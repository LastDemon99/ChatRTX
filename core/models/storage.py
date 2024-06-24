import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.core.base_query_engine import BaseQueryEngine

from core.models.faiss_vector_storage import FaissEmbeddingStorage
from core.config.app import AppConfig
from core.config.models import StorageConfig

class Storage:
    def __init__(self):
        self.embed_model: HuggingFaceEmbeddings = None
        self.engine: BaseChatEngine | BaseQueryEngine = None
        self.faiss_storage: FaissEmbeddingStorage = None
        self.config: StorageConfig = None

    def load(self, app_config: AppConfig):
        self.config = StorageConfig(app_config)
        self.embed_model = HuggingFaceEmbeddings(model_name=self.config.embedded_model_name) # create embeddings model object

    def unload(self):
        del self.model
        self.model = None

    def generate_inferance_engine(self, app_config: AppConfig, force_rewrite=False):
        """
        Initialize and return a FAISS-based inference engine.

        Args:
            data: The directory where the data for the inference engine is located.
            force_rewrite (bool): If True, force rewriting the index.

        Returns:
            The initialized inference engine.

        Raises:
            RuntimeError: If unable to generate the inference engine.
        """
        self.faiss_storage = FaissEmbeddingStorage(self.config.embedded_dimension)
        self.faiss_storage.initialize_index(app_config.data_dir, force_rewrite)
        self.engine = self.faiss_storage.get_engine(app_config.is_chat_engine, app_config.streaming,
                                        self.config.similarity_top_k)