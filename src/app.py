# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import calendar
import time
import os
import sys
import shutil
import gc
import torch
import scipy.signal as sps
import numpy as np

from pathlib import Path
from scipy.io import wavfile
from ui.user_interface import MainInterface
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from llama_index import set_global_service_context, ServiceContext

from core.utils.model_setup_manager import download_model_by_name, build_engine_by_name
from core.utils.utils import validate_privileges

from core.models.clip import Clip
from core.models.llm import LLM
from core.models.storage import Storage
from core.models.whisper import Whisper

from core.config.app import AppConfig

def set_model(model_name, model_config, session_id=None):
    app_config.selected_model = model_name
    storage.config = app_config.get_storage_config()

    if app_config.is_selected_clip():
        clip.load(model_config, app_config.data_dir) # Initialize model and processor
    else:
        llm.load(model_config, app_config) # create trt_llm engine object   
        storage.load(app_config) # create embeddings model object

        service_context = ServiceContext.from_defaults(llm=llm.model, embed_model=storage.embed_model,
                                                context_window=llm.model.context_window, chunk_size=512,
                                                chunk_overlap=200)
        set_global_service_context(service_context)

        storage.generate_inferance_engine(app_config) # load the vectorstore index

def call_llm_streamed(query):
    partial_response = ""
    response = llm.model.stream_complete(query, formatted=False)
    for token in response:
        partial_response += token.delta
        yield partial_response

def clip_chat(query):
    ts = calendar.timegm(time.gmtime())
    temp_image_folder_name = "Temp/Temp_Images"
    if os.path.isdir(temp_image_folder_name):                
        try:
            shutil.rmtree(os.path.join(os.getcwd(), temp_image_folder_name))
        except Exception as e:
            print("Exception during folder delete", e)
    image_results_path = os.path.join(os.getcwd(), temp_image_folder_name, str(ts))
    res_im_paths = clip.engine.query(query, image_results_path)
    if len(res_im_paths) == 0:
        yield "No supported images found in the selected folder"
        torch.cuda.empty_cache()
        gc.collect()
        return

    div_start = '<div class="chat-output-images">'
    div_end = '</div>'
    im_elements = ''
    for i, im in enumerate(res_im_paths):
        if i>2 : break # display atmost 3 images.
        cur_data_link_src = temp_image_folder_name +"/" + str(ts) + "/" + os.path.basename(im)
        cur_src = "file/" + temp_image_folder_name +"/" + str(ts) + "/" + os.path.basename(im)
        im_elements += '<img data-link="{data_link_src}" src="{src}"/>'.format(src=cur_src, data_link_src=cur_data_link_src)
    full_div = (div_start + im_elements + div_end)
    folder_link = f'<a data-link="{image_results_path}">{"See all matches"}</a>'
    prefix = ""
    if(len(res_im_paths)>1): 
        prefix = "Here are the top matching pictures from your dataset"
    else:
        prefix = "Here is the top matching picture from your dataset"
    response = prefix + "<br>"+ full_div + "<br>"+ folder_link

    torch.cuda.empty_cache()
    gc.collect()

    return response

def get_lowest_score(response):
    # Aggregate scores by file
    lowest_score_file = None
    lowest_score = sys.float_info.max

    for node in response.source_nodes:
        if 'filename' in node.metadata:
            if node.score < lowest_score:
                lowest_score = node.score
                lowest_score_file = node.metadata['filename']
    return lowest_score_file

def create_temp_docs_directory():
    ts = calendar.timegm(time.gmtime())
    temp_docs_folder_name = "Temp/Temp_Docs"
    docs_path = os.path.join(os.getcwd(), temp_docs_folder_name, str(ts))
    os.makedirs(docs_path, exist_ok=True)
    return docs_path

def generate_links_file(lowest_score_file, file_links, docs_path):
    seen_files = set()  # Set to track unique file names

    # Generate links for the file with the highest aggregated score
    if lowest_score_file:
        abs_path = Path(os.path.join(os.getcwd(), lowest_score_file.replace('\\', '/')))
        file_name = os.path.basename(abs_path)
        doc_path = os.path.join(docs_path, file_name)
        shutil.copy(abs_path, doc_path)

        if file_name not in seen_files:  # Ensure the file hasn't already been processed
            if app_config.dataset_is_allowed:
                file_link = f'<a data-link="{doc_path}">{file_name}</a>'
            else:
                exit("Wrong data_source type")
            file_links.append(file_link)
            seen_files.add(file_name)  # Mark file as processed

def chatbot(query, chat_history, session_id):
    if app_config.is_selected_clip():
        yield clip_chat(query)
        return

    if not app_config.dataset_is_allowed:
        yield llm.model.complete(query, formatted=False).text
        return

    if app_config.is_chat_engine:
        response = storage.engine.chat(query)
    else:
        response = storage.engine.query(query)

    file_links = []
    lowest_score_file = get_lowest_score(response)
    docs_path = create_temp_docs_directory()
    generate_links_file(lowest_score_file, file_links, docs_path)

    response_txt = str(response)
    if file_links:
        response_txt += "<br>Reference files:<br>" + "<br>".join(file_links)
    if not lowest_score_file:  # If no file with a high score was found
        response_txt = llm.model.complete(query).text
    yield response_txt

def stream_chatbot(query, chat_history, session_id):
    
    if app_config.is_selected_clip():
        yield clip_chat(query)
        return

    if not app_config.dataset_is_allowed:
        for response in call_llm_streamed(query):
            yield response
        return

    if app_config.is_chat_engine:
        response = storage.engine.stream_chat(query)
    else:
        response = storage.engine.query(query)

    partial_response = ""
    if len(response.source_nodes) == 0:
        response = llm.stream_complete(query, formatted=False)
        for token in response:
            partial_response += token.delta
            yield partial_response
    else:
        lowest_score_file = get_lowest_score(response)

        file_links = []
        seen_files = set()

        for token in response.response_gen:
            partial_response += token
            yield partial_response

        docs_path = create_temp_docs_directory()

        if lowest_score_file:
            abs_path = Path(os.path.join(os.getcwd(), lowest_score_file.replace('\\', '/')))
            file_name = os.path.basename(abs_path)
            doc_path = os.path.join(docs_path, file_name)
            shutil.copy(abs_path, doc_path)
            if file_name not in seen_files:  # Check if file_name is already seen
                if app_config.dataset_is_allowed:
                    file_link = f'<a data-link="{doc_path}">{file_name}</a>'
                else:
                    exit("Wrong data_source type")
                file_links.append(file_link)
                seen_files.add(file_name)  # Add file_name to the set

        if seen_files:
            partial_response += "<br>Reference files:<br>" + "<br>".join(seen_files)
        
        yield partial_response

    # call garbage collector after inference
    torch.cuda.empty_cache()
    gc.collect()

def shutdown_handler(session_id):
    if whisper.is_loaded():
        whisper.unload()

    if llm.is_loaded():
        llm.unload()

    if clip.is_loaded():
        clip.unload()

    temp_data_folder_name = "Temp"
    if os.path.isdir(temp_data_folder_name):                
        try:
            shutil.rmtree(os.path.join(os.getcwd(), temp_data_folder_name))
        except Exception as e:
            print("Exception during temp folder delete", e)
    # Force a garbage collection cycle
    gc.collect()

def reset_chat_handler(session_id):
    print('reset chat called', session_id)
    if app_config.is_selected_clip():
        return
    if app_config.is_chat_engine:
        storage.faiss_storage.reset_engine(storage.engine)

def mic_init_handler():
    if not whisper_config.enable_asr:
        return False
    vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    free_vid_mem = vid_mem_info.free / (1024*1024)
    print("free video memory in MB = ", free_vid_mem)
    if whisper.is_loaded():
        whisper.unload()
    whisper.load(whisper_config)
    return True

def mic_recording_done_handler(audio_path):
    transcription = ""
    if not whisper_config.enable_asr:
        return ""
    
    # Check and wait until model is loaded before running it.
    checks_left_for_model_loading = 40
    sleep_time = 0.2
    while checks_left_for_model_loading>0 and not whisper.is_loaded():
        time.sleep(sleep_time)
        checks_left_for_model_loading -= 1
    assert checks_left_for_model_loading>0, f"Whisper model loading not finished even after {(checks_left_for_model_loading*sleep_time)} seconds"
    if checks_left_for_model_loading == 0:
        return ""

    # Covert the audio file into required sampling rate
    current_sampling_rate, data = wavfile.read(audio_path)
    new_sampling_rate = 16000
    number_of_samples = round(len(data) * float(new_sampling_rate) / current_sampling_rate)
    data = sps.resample(data, number_of_samples)
    new_file_path = os.path.join( os.path.dirname(audio_path), "whisper_audio_input.wav" )
    wavfile.write(new_file_path, new_sampling_rate, data.astype(np.int16))
    language = "english"
    if app_config.is_selected_chatGLM(): language = "chinese"
    transcription = whisper.decode_audio_file(new_file_path, whisper_config.asr_assets_path, language)

    if whisper.is_loaded():
        whisper.unload()        
    return transcription

def model_download_handler(model_info):
    status = download_model_by_name(model_info=model_info,  download_path=app_config.download_path)
    print(f"Model download status: {status}")
    return status

def model_install_handler(model_info):
    #unload the current model
    if llm.is_loaded():
        llm.unload()

    if clip.is_loaded():
        clip.unload()
    # build the engine
    status = build_engine_by_name(model_info=model_info , download_path=app_config.download_path)
    print(f"Engine build status: {status}")
    return status

def model_delete_handler(model_info):
    print("Model deleting ", model_info)
    model_dir = os.path.join(os.getcwd(), "model", model_info['id'])
    isSuccess = True
    if os.path.isdir(model_dir): 
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print("Exception during temp folder delete", e)
            isSuccess = False
    return isSuccess

def on_dataset_path_updated_handler(source, new_directory, video_count, session_id):
    print('data set path updated to ', source, new_directory, video_count, session_id)
    if app_config.is_selected_clip():
        app_config.data_dir = new_directory
        clip.generate_engine(new_directory)
        return
    if source == 'directory':
        if app_config.data_dir != new_directory:
            app_config.data_dir = new_directory
            storage.generate_inferance_engine(app_config)

def on_dataset_source_change_handler(source, path, session_id):
    app_config.dataset_is_allowed = source == "directory"

    if not app_config.dataset_is_allowed:
        print(' No dataset source selected', session_id)
        return
    
    print('dataset source updated ', source, path, session_id)
    
    app_config.data_dir = path
    if app_config.dataset_is_allowed:
        storage.generate_inferance_engine(app_config)
    else:
        print("Wrong data type selected")

def handle_regenerate_index(source, path, session_id):
    app_config.data_dir = path
    if app_config.is_selected_clip():
        clip.generate_engine(path, force_rewrite=True)
    else:
        storage.generate_inferance_engine(app_config, force_rewrite=True)
    print("on regenerate index", source, path, session_id)

if __name__ == "__main__":
    validate_privileges()
    nvmlInit()

    app_config = AppConfig()
    model_config = app_config.get_model_config()    
    whisper_config = app_config.get_whisper_config(model_config)

    llm: LLM = LLM()
    clip: Clip = Clip()
    whisper: Whisper = Whisper()
    storage: Storage = Storage()

    set_model(app_config.selected_model, model_config.config)

    interface = MainInterface(chatbot=stream_chatbot if app_config.streaming else chatbot, streaming=app_config.streaming)
    interface.on_shutdown(shutdown_handler)
    interface.on_reset_chat(reset_chat_handler)
    interface.on_mic_button_click(mic_init_handler)
    interface.on_mic_recording_done(mic_recording_done_handler)
    interface.on_model_change(set_model)
    interface.on_model_downloaded(model_download_handler)
    interface.on_model_installed(model_install_handler)
    interface.on_model_delete(model_delete_handler)
    interface.on_dataset_path_updated(on_dataset_path_updated_handler)
    interface.on_dataset_source_updated(on_dataset_source_change_handler)
    interface.on_regenerate_index(handle_regenerate_index)

    interface.render()
    