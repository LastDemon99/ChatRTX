from core.utils.llm_prompt_templates import LLMPromptTemplate
from llama_index.llms.llama_utils import messages_to_prompt
from core.utils.trt_llama_api import TrtLlmAPI

from core.config.app import AppConfig
from core.config.models import ModelConfig
from core.utils.utils import read_model_name

class LLM:
    def __init__(self):
        self.model: TrtLlmAPI = None

    def is_loaded(self):
        return self.model is not None

    def load(self, model_config: ModelConfig, app_config: AppConfig):
        model_name, _ = read_model_name(model_config["model_path"])
        prompt_template_obj = LLMPromptTemplate()
        text_qa_template_str = prompt_template_obj.model_context_template(model_name)
        selected_completion_to_prompt =  text_qa_template_str

        self.model = TrtLlmAPI(
            model_path=model_config["model_path"],
            engine_name=model_config["engine"],
            tokenizer_dir=model_config["tokenizer_path"],
            temperature=model_config["temperature"],
            max_new_tokens=model_config["max_new_tokens"],
            context_window=model_config["max_input_token"],
            vocab_file=model_config["vocab"],
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=selected_completion_to_prompt,
            use_py_session=app_config["use_py_session"],
            add_special_tokens=app_config["add_special_tokens"],
            trtLlm_debug_mode=app_config["trtLlm_debug_mode"],
            verbose=app_config["verbose"]
        )

    def unload(self):
        self.model.unload_model()
        del self.model
        self.model = None