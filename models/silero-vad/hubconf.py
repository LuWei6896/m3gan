dependencies = ['torch']
import torch
import json
from utils_vad import (init_jit_model,
                       get_speech_timestamps,
                       get_number_ts,
                       get_language,
                       get_language_and_group,
                       save_audio,
                       load_audio,
                       VADIterator,
                       collect_chunks,
                       drop_chunks,
                       Validator,
                       OnnxWrapper)


def versiontuple(v):
    return tuple(map(int, (v.split('+')[0].split("."))))


def silero_vad(onnx=False, force_onnx_cpu=False):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """

    if not onnx:
        installed_version = torch.__version__
        supported_version = '1.12.0'
        if versiontuple(installed_version) < versiontuple(supported_version):
            raise Exception(f'Please install torch {supported_version} or greater ({installed_version} installed)')

    hub_dir = torch.hub.get_dir()
    if onnx:
        model = OnnxWrapper(f'{hub_dir}/snakers4_silero-vad_master/files/silero_vad.onnx', force_onnx_cpu)
    else:
        model = init_jit_model(model_path=f'{hub_dir}/snakers4_silero-vad_master/files/silero_vad.jit')
    utils = (get_speech_timestamps,
             save_audio,
             load_audio,
             VADIterator,
             collect_chunks)

    return model, utils

def silero_vad_local(onnx=False, force_onnx_cpu=False):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    
    Local version that provides path internally rather than from .cache/torch/hub
    """

    if not onnx:
        installed_version = torch.__version__
        supported_version = '1.12.0'
        if versiontuple(installed_version) < versiontuple(supported_version):
            raise Exception(f'Please install torch {supported_version} or greater ({installed_version} installed)')

    import pathlib
    import os
    hub_dir = str(pathlib.Path(os.getcwd())) # should be C:/Users/username/documents/transcription_dev
    if onnx:
        model = OnnxWrapper(f'{hub_dir}/models/silero-vad/files/silero_vad.onnx', force_onnx_cpu)
    else:
        model = init_jit_model(model_path=f'{hub_dir}/models/silero-vad/files/silero_vad.jit')
    utils = (get_speech_timestamps,
             save_audio,
             load_audio,
             collect_chunks)

    return model, utils


def silero_number_detector(onnx=False, force_onnx_cpu=False):
    """Silero Number Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    if onnx:
        url = 'https://models.silero.ai/vad_models/number_detector.onnx'
    else:
        url = 'https://models.silero.ai/vad_models/number_detector.jit'
    model = Validator(url, force_onnx_cpu)
    utils = (get_number_ts,
             save_audio,
             load_audio,
             collect_chunks,
             drop_chunks)

    return model, utils


def silero_lang_detector(onnx=False, force_onnx_cpu=False):
    """Silero Language Classifier
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    if onnx:
        url = 'https://models.silero.ai/vad_models/number_detector.onnx'
    else:
        url = 'https://models.silero.ai/vad_models/number_detector.jit'
    model = Validator(url, force_onnx_cpu)
    utils = (get_language,
             load_audio)

    return model, utils


def silero_lang_detector_95(onnx=False, force_onnx_cpu=False):
    """Silero Language Classifier (95 languages)
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """

    hub_dir = torch.hub.get_dir()
    if onnx:
        url = 'https://models.silero.ai/vad_models/lang_classifier_95.onnx'
    else:
        url = 'https://models.silero.ai/vad_models/lang_classifier_95.jit'
    model = Validator(url, force_onnx_cpu)

    with open(f'{hub_dir}/snakers4_silero-vad_master/files/lang_dict_95.json', 'r') as f:
        lang_dict = json.load(f)

    with open(f'{hub_dir}/snakers4_silero-vad_master/files/lang_group_dict_95.json', 'r') as f:
        lang_group_dict = json.load(f)

    utils = (get_language_and_group, load_audio)

    return model, lang_dict, lang_group_dict, utils
