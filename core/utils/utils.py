# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, T5Tokenizer
import win32api
import win32security
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from tensorrt_llm.builder import get_engine_version

# TODO(enweiz): Update for refactored models
DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPTForCausalLM': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
}

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'QWenForCausalLM':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
}

def validate_privileges():
    # Use GetCurrentProcess to get a handle to the current process
    hproc = win32api.GetCurrentProcess()
    # Use GetCurrentProcessToken to get the token of the current process
    htok = win32security.OpenProcessToken(hproc, win32security.TOKEN_QUERY)

    # Retrieve the list of privileges enabled
    privileges = win32security.GetTokenInformation(htok, win32security.TokenPrivileges)

    # Iterate over privileges and output the ones that are enabled
    priv_list = []
    for priv_id, priv_flags in privileges:
        # Check if privilege is enabled
        if priv_flags == win32security.SE_PRIVILEGE_ENABLED or win32security.SE_PRIVILEGE_ENABLED_BY_DEFAULT:
            # Lookup the name of the privilege
            priv_name = win32security.LookupPrivilegeName(None, priv_id)
            priv_list.append(priv_name)

    print(f"Privileges of app process: {priv_list}")

def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    return model_arch, model_version

def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left',
                                legacy=False)

    if model_name == 'QWenForCausalLM':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == 'raw' or chat_format == 'chatml':
            pad_id = gen_config['pad_token_id']
            end_id = gen_config['eos_token_id']
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id

def display_free_gpu_memory():
    vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    free_vid_mem = vid_mem_info.free / (1024*1024)
    print("free video memory in MB = ", free_vid_mem)