import json
from dataclasses import dataclass

APP_CONFIG_FILE = "config\\app_config.json"
PREFERENCE_CONFIG_FILE = "config\\preferences.json"
MODEL_CONFIG_FILE = "config\\config.json"

@dataclass(frozen=True)
class DefaultModels:
    MISTRAL: str = ""
    CLIP: str = "CLIP"
    ChatGLM: str = "ChatGLM 3 6B int4 (Supports Chinese)"
    WHISPER: str = "Whisper Medium Int8"

def read_config(file_name):
    try:
        with open(file_name, 'r', encoding='utf8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except json.JSONDecodeError:
        print(f"There was an error decoding the JSON from the file {file_name}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None