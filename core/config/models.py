import os
from core.utils.config import DefaultModels, read_config, MODEL_CONFIG_FILE

class ModelConfig:
    def __init__(self, selected_model):
        self.config: dict = read_config(MODEL_CONFIG_FILE)
        self.set_model_config(selected_model)

    def set_model_config(self, model_name):
        selected_model = next((model for model in self.config["models"]["supported"] if model["name"] == model_name), self.config["models"]["supported"][0])
        metadata = selected_model["metadata"]

        cwd = os.getcwd()  # Current working directory, to avoid calling os.getcwd() multiple times

        self.config.update(selected_model)
        if "ngc_model_name" in selected_model:
            self.config.update({
                "model_path": os.path.join(cwd, "model", selected_model["id"], "engine") if "id" in selected_model else None,
                "engine": metadata.get("engine", None),
                "tokenizer_path": os.path.join(cwd, "model", selected_model["id"] ,selected_model["prerequisite"]["tokenizer_local_dir"] ) if "tokenizer_local_dir" in selected_model["prerequisite"] else None,
                "vocab": os.path.join(cwd, "model", selected_model["id"] ,selected_model["prerequisite"]["vocab_local_dir"], selected_model["prerequisite"]["tokenizer_files"]["vocab_file"]) if "vocab_local_dir" in selected_model["prerequisite"] else None,
                "max_new_tokens": metadata.get("max_new_tokens", None),
                "max_input_token": metadata.get("max_input_token", None),
                "temperature": metadata.get("temperature", None),
                "prompt_template": metadata.get("prompt_template", None)
            })
        elif "hf_model_name" in selected_model:
            self.config.update({
                "model_path": os.path.join(cwd, "model", selected_model["id"]) if "id" in selected_model else None,
                "tokenizer_path": os.path.join(cwd, "model", selected_model["id"]) if "id" in selected_model else None,
                "prompt_template": metadata.get("prompt_template", None)
            })

        if not self.config["model_path"] or not self.config["engine"]:
            print("Model path or engine not provided in metadata")
    
    def __getitem__(self, key):
        return self.config[key]
    
class WhisperConfig:
    def __init__(self, config : ModelConfig, model_name=None):
        self.asr_model_name = DefaultModels.WHISPER if model_name == None else model_name
        self.asr_model_config = self._get_asr_model_config(config, self.asr_model_name)
        self.asr_engine_path = self.asr_model_config["model_path"]
        self.asr_assets_path = self.asr_model_config["assets_path"]
        self.whisper_model_loaded = False
        self.enable_asr = config["models"]["enable_asr"]

    def _get_asr_model_config(self, config, model_name):
        models = config["models"]["supported_asr"]
        selected_model = next((model for model in models if model["name"] == model_name), models[0])
        return {
            "model_path": os.path.join(os.getcwd(), selected_model["metadata"]["model_path"]),
            "assets_path": os.path.join(os.getcwd(), selected_model["metadata"]["assets_path"])
        }

class StorageConfig:
    def __init__(self, app_config):
        self.embedded_model_name = app_config["embedded_model"]
        self.embedded_model_name = os.path.join(os.getcwd(), "model", self.embedded_model_name)
        self.embedded_dimension = app_config["embedded_dimension"]
        self.similarity_top_k = app_config["similarity_top_k"]
