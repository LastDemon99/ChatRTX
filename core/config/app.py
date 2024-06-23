import os
from core.utils.config import DefaultModels, read_config, APP_CONFIG_FILE, PREFERENCE_CONFIG_FILE, MODEL_CONFIG_FILE
from core.config.models import ModelConfig, WhisperConfig, StorageConfig

class AppConfig:
    def __init__(self):
        self.config = read_config(APP_CONFIG_FILE)
        self.download_path = os.path.join(os.getcwd(), "model")
        self.dataset_is_allowed = True
        self.is_chat_engine = self.config["is_chat_engine"]
        self.streaming = self.config["streaming"]

        if os.path.exists(PREFERENCE_CONFIG_FILE):
            config = read_config(PREFERENCE_CONFIG_FILE)
            self.data_dir = config.get('dataset', {}).get('path')
            self.data_is_relative = config.get('dataset', {}).get('isRelative')
            self.selected_model = config.get('models', {}).get('selected')
        else:
            config = read_config(MODEL_CONFIG_FILE)
            self.data_dir = config["dataset"]["path"]
            self.selected_model = config["models"].get("selected")
            
    def is_selected_clip(self):
        return self.selected_model == DefaultModels.CLIP
    
    def is_selected_chatGLM(self):
        return self.selected_model == DefaultModels.ChatGLM    

    def is_selected_mistral(self):
        return self.selected_model == DefaultModels.MISTRAL

    def get_model_config(self):
        return ModelConfig(self.selected_model)
    
    def get_storage_config(self):
        return StorageConfig(self.config) if self.dataset_is_allowed else None
    
    def get_whisper_config(self, model_config: ModelConfig):
        return WhisperConfig(model_config)

    def __getitem__(self, key):
        return self.config[key]