import torch
from typing import List
import torch.nn.functional as F
import warnings

import ffmpeg
import numpy as np
import soundfile as sf
languages = ['ru', 'en', 'de', 'es']


class OnnxWrapper():

    def __init__(self, path, force_onnx_cpu=False):
        import numpy as np
        global np
        import onnxruntime
        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(path)
        self.session.intra_op_num_threads = 1
        self.session.inter_op_num_threads = 1

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._h = np.zeros((2, batch_size, 64)).astype('float32')
        self._c = np.zeros((2, batch_size, 64)).astype('float32')
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):

        x, sr = self._validate_input(x, sr)
        batch_size = x.shape[0]

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if sr in [8000, 16000]:
            ort_inputs = {'input': x.numpy(), 'h': self._h, 'c': self._c, 'sr': np.array(sr, dtype='int64')}
            ort_outs = self.session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
        else:
            raise ValueError()

        self._last_sr = sr
        self._last_batch_size = batch_size

        out = torch.tensor(out)
        return out

    def audio_forward(self, x, sr: int, num_samples: int = 512):
        outs = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), 'constant', value=0.0)

        self.reset_states(x.shape[0])
        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i+num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1)
        return stacked.cpu()


class Validator():
    def __init__(self, url, force_onnx_cpu):
        self.onnx = True if url.endswith('.onnx') else False
        torch.hub.download_url_to_file(url, 'inf.model')
        if self.onnx:
            import onnxruntime
            if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
                self.model = onnxruntime.InferenceSession('inf.model', providers=['CPUExecutionProvider'])
            else:
                self.model = onnxruntime.InferenceSession('inf.model')
        else:
            self.model = init_jit_model(model_path='inf.model')

    def __call__(self, inputs: torch.Tensor):
        with torch.no_grad():
            if self.onnx:
                ort_inputs = {'input': inputs.cpu().numpy()}
                outs = self.model.run(None, ort_inputs)
                outs = [torch.Tensor(x) for x in outs]
            else:
                outs = self.model(inputs)

        return outs

# This is from whisper_module/whisper/audio.py
# We will use this instead of torchaudio
def load_audio(file: str, ffmpeg_path: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=[ffmpeg_path, "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    # import numpy as np
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def _get_subtype(dtype: torch.dtype, format: str, encoding: str, bits_per_sample: int):
    if format == "wav":
        return _get_subtype_for_wav(dtype, encoding, bits_per_sample)
    if format == "flac":
        if encoding:
            raise ValueError("flac does not support encoding.")
        if not bits_per_sample:
            return "PCM_16"
        if bits_per_sample > 24:
            raise ValueError("flac does not support bits_per_sample > 24.")
        return "PCM_S8" if bits_per_sample == 8 else f"PCM_{bits_per_sample}"
    if format in ("ogg", "vorbis"):
        if encoding or bits_per_sample:
            raise ValueError("ogg/vorbis does not support encoding/bits_per_sample.")
        return "VORBIS"
    if format in ("nis", "nist"):
        return "PCM_16"
    raise ValueError(f"Unsupported format: {format}")

def _get_subtype_for_wav(dtype: torch.dtype, encoding: str, bits_per_sample: int):
    if not encoding:
        if not bits_per_sample:
            subtype = {
                torch.uint8: "PCM_U8",
                torch.int16: "PCM_16",
                torch.int32: "PCM_32",
                torch.float32: "FLOAT",
                torch.float64: "DOUBLE",
            }.get(dtype)
            if not subtype:
                raise ValueError(f"Unsupported dtype for wav: {dtype}")
            return subtype
        if bits_per_sample == 8:
            return "PCM_U8"
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_S":
        if not bits_per_sample:
            return "PCM_32"
        if bits_per_sample == 8:
            raise ValueError("wav does not support 8-bit signed PCM encoding.")
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_U":
        if bits_per_sample in (None, 8):
            return "PCM_U8"
        raise ValueError("wav only supports 8-bit unsigned PCM encoding.")
    if encoding == "PCM_F":
        if bits_per_sample in (None, 32):
            return "FLOAT"
        if bits_per_sample == 64:
            return "DOUBLE"
        raise ValueError("wav only supports 32/64-bit float PCM encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("wav only supports 8-bit mu-law encoding.")
    if encoding == "ALAW":
        if bits_per_sample in (None, 8):
            return "ALAW"
        raise ValueError("wav only supports 8-bit a-law encoding.")
    raise ValueError(f"wav does not support {encoding}.")

def save_wav(
    filepath: str,
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: float = None,
    format: str = None,
    encoding: str = None,
    bits_per_sample: int = None,
):
    """Save audio data to file.

    Note:
        The formats this function can handle depend on the soundfile installation.
        This function is tested on the following formats;

        * WAV

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * FLAC
        * OGG/VORBIS
        * SPHERE

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,
        which has a restriction on type annotation due to TorchScript compiler compatiblity.

    Args:
        filepath (str or pathlib.Path): Path to audio file.
        src (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float of None, optional): Not used.
            It is here only for interface compatibility reson with "sox_io" backend.
        format (str or None, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is
            inferred from file extension. If the file extension is missing or
            different, you can specify the correct format with this argument.

            When ``filepath`` argument is file-like object,
            this argument is required.

            Valid values are ``"wav"``, ``"ogg"``, ``"vorbis"``,
            ``"flac"`` and ``"sph"``.
        encoding (str or None, optional): Changes the encoding for supported formats.
            This argument is effective only for supported formats, sush as
            ``"wav"``, ``""flac"`` and ``"sph"``. Valid values are;

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

        bits_per_sample (int or None, optional): Changes the bit depth for the
            supported formats.
            When ``format`` is one of ``"wav"``, ``"flac"`` or ``"sph"``,
            you can change the bit depth.
            Valid values are ``8``, ``16``, ``24``, ``32`` and ``64``.

    Supported formats/encodings/bit depth/compression are:

    ``"wav"``
        - 32-bit floating-point PCM
        - 32-bit signed integer PCM
        - 24-bit signed integer PCM
        - 16-bit signed integer PCM
        - 8-bit unsigned integer PCM
        - 8-bit mu-law
        - 8-bit a-law

        Note:
            Default encoding/bit depth is determined by the dtype of
            the input Tensor.

    ``"flac"``
        - 8-bit
        - 16-bit (default)
        - 24-bit

    ``"ogg"``, ``"vorbis"``
        - Doesn't accept changing configuration.

    ``"sph"``
        - 8-bit signed integer PCM
        - 16-bit signed integer PCM
        - 24-bit signed integer PCM
        - 32-bit signed integer PCM (default)
        - 8-bit mu-law
        - 8-bit a-law
        - 16-bit a-law
        - 24-bit a-law
        - 32-bit a-law

    """
    if src.ndim != 2:
        raise ValueError(f"Expected 2D Tensor, got {src.ndim}D.")
    if compression is not None:
        warnings.warn(
            '`save` function of "soundfile" backend does not support "compression" parameter. '
            "The argument is silently ignored."
        )
    if hasattr(filepath, "write"):
        if format is None:
            raise RuntimeError("`format` is required when saving to file object.")
        ext = format.lower()
    else:
        ext = str(filepath).split(".")[-1].lower()

    if bits_per_sample not in (None, 8, 16, 24, 32, 64):
        raise ValueError("Invalid bits_per_sample.")
    if bits_per_sample == 24:
        warnings.warn(
            "Saving audio with 24 bits per sample might warp samples near -1. "
            "Using 16 bits per sample might be able to avoid this."
        )
    subtype = _get_subtype(src.dtype, ext, encoding, bits_per_sample)

    # sph is a extension used in TED-LIUM but soundfile does not recognize it as NIST format,
    # so we extend the extensions manually here
    if ext in ["nis", "nist", "sph"] and format is None:
        format = "NIST"

    if channels_first:
        src = src.t()

    sf.write(file=filepath, data=src, samplerate=sample_rate, subtype=subtype, format=format)


def save_audio(
    path:str,
    tensor: torch.Tensor,
    sampling_rate: int = 16000
):
    save_wav(path, tensor.unsqueeze(0), sampling_rate, bits_per_sample=16)

def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def make_visualization(probs, step):
    import pandas as pd
    pd.DataFrame({'probs': probs},
                 index=[x * step for x in range(len(probs))]).plot(figsize=(16, 8),
                 kind='area', ylim=[0, 1.05], xlim=[0, len(probs) * step],
                 xlabel='seconds',
                 ylabel='speech probability',
                 colormap='tab20')


def get_speech_timestamps(audio: torch.Tensor,
                          model,
                          threshold: float = 0.5,
                          sampling_rate: int = 16000,
                          min_speech_duration_ms: int = 250,
                          max_speech_duration_s: float = float('inf'),
                          min_silence_duration_ms: int = 100,
                          window_size_samples: int = 512,
                          speech_pad_ms: int = 30,
                          return_seconds: bool = False,
                          visualize_probs: bool = False):

    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded .jit silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100s (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    visualize_probs: bool (default - False)
        whether draw prob hist or not

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn('window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!')
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn('Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate')

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0 # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0 # to save potential segment limits in case of maximum segment size reached

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
               next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue
        
        if triggered and (window_size_samples * i) - current_speech['start'] > max_speech_samples:
            if prev_end:
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end: # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech['start'] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech['end'] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue
                

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if ((window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech : # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 1)
            speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


def get_number_ts(wav: torch.Tensor,
                  model,
                  model_stride=8,
                  hop_length=160,
                  sample_rate=16000):
    wav = torch.unsqueeze(wav, dim=0)
    perframe_logits = model(wav)[0]
    perframe_preds = torch.argmax(torch.softmax(perframe_logits, dim=1), dim=1).squeeze()   # (1, num_frames_strided)
    extended_preds = []
    for i in perframe_preds:
        extended_preds.extend([i.item()] * model_stride)
    # len(extended_preds) is *num_frames_real*; for each frame of audio we know if it has a number in it.
    triggered = False
    timings = []
    cur_timing = {}
    for i, pred in enumerate(extended_preds):
        if pred == 1:
            if not triggered:
                cur_timing['start'] = int((i * hop_length) / (sample_rate / 1000))
                triggered = True
        elif pred == 0:
            if triggered:
                cur_timing['end'] = int((i * hop_length) / (sample_rate / 1000))
                timings.append(cur_timing)
                cur_timing = {}
                triggered = False
    if cur_timing:
        cur_timing['end'] = int(len(wav) / (sample_rate / 1000))
        timings.append(cur_timing)
    return timings


def get_language(wav: torch.Tensor,
                 model):
    wav = torch.unsqueeze(wav, dim=0)
    lang_logits = model(wav)[2]
    lang_pred = torch.argmax(torch.softmax(lang_logits, dim=1), dim=1).item()   # from 0 to len(languages) - 1
    assert lang_pred < len(languages)
    return languages[lang_pred]


def get_language_and_group(wav: torch.Tensor,
                           model,
                           lang_dict: dict,
                           lang_group_dict: dict,
                           top_n=1):
    wav = torch.unsqueeze(wav, dim=0)
    lang_logits, lang_group_logits = model(wav)

    softm = torch.softmax(lang_logits, dim=1).squeeze()
    softm_group = torch.softmax(lang_group_logits, dim=1).squeeze()

    srtd = torch.argsort(softm, descending=True)
    srtd_group = torch.argsort(softm_group, descending=True)

    outs = []
    outs_group = []
    for i in range(top_n):
        prob = round(softm[srtd[i]].item(), 2)
        prob_group = round(softm_group[srtd_group[i]].item(), 2)
        outs.append((lang_dict[str(srtd[i].item())], prob))
        outs_group.append((lang_group_dict[str(srtd_group[i].item())], prob_group))

    return outs, outs_group


class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None


def collect_chunks(tss: List[dict],
                   wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return torch.cat(chunks)


def drop_chunks(tss: List[dict],
                wav: torch.Tensor):
    chunks = []
    cur_start = 0
    for i in tss:
        chunks.append((wav[cur_start: i['start']]))
        cur_start = i['end']
    return torch.cat(chunks)
