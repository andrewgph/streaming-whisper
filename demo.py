import argparse
import numpy as np
import time
import mlx.core as mx
from webrtcvad import Vad

import audio
from audio_input import AudioInput
from whisper_mlx import load_model
from decoding import StreamingDecoder, DecodingOptions


def audio_bytes_to_np_array(bytes_data):
    # https://github.com/WarrenWeckesser/wavio/blob/master/wavio.py#L67
    arr = np.frombuffer(bytes_data, dtype='<i2')
    # Modifying 16 bit data using above approach
    arr = arr.astype('float32') / 32768.0
    return arr


class IncrementalTranscriber:

    def __init__(self, streaming_decoder: StreamingDecoder, vad: Vad, n_mels: int, no_streaming: bool = False):
        self.streaming_decoder = streaming_decoder
        self.vad = vad
        self.n_mels = n_mels
        self.no_streaming = no_streaming
        self.audio_bytes_buffer = b''

    def update(self, audio_bytes: bytes):
        processing_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        if self.no_streaming:
            # Reset the streaming decoder to remove cached state
            self.streaming_decoder = StreamingDecoder(self.streaming_decoder.model, self.streaming_decoder.options)
            self.audio_bytes_buffer += audio_bytes
            # Maintain buffer with 30 seconds of audio
            # 16000 hz * 2 bytes per sample * 30 seconds
            if len(self.audio_bytes_buffer) >= 16000 * 2 * 30:
                self.audio_bytes_buffer = self.audio_bytes_buffer[-(16000 * 2 * 30):]
            audio_bytes = self.audio_bytes_buffer

        contains_speech = False
        for i in range(0, len(audio_bytes), 320 * 3):
            contains_speech = contains_speech or self.vad.is_speech(audio_bytes[i:i+(320 * 3)], 16000)

        if not contains_speech:
            num_mel_frames = len(audio_bytes) // 320
            self.streaming_decoder.add_empty_window(num_mel_frames)
            print(f"{processing_time} : [no speech detected]")
            return

        start_time = time.time()
        audio_arr = audio_bytes_to_np_array(audio_bytes)
        mel_arr = audio.log_mel_spectrogram(audio_arr, self.n_mels)
        mel_arr = mel_arr.reshape(1, *mel_arr.shape)
        result = self.streaming_decoder.incremental_decode(mel_arr)
        mx.eval()
        end_time = time.time()
        decoding_time_ms = int(1000 * (end_time - start_time))
        print(f"{processing_time} : {decoding_time_ms}ms : {result.new_text}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio file.')
    parser.add_argument('--model_path', type=str, help='Path to the whisper mlx model.')
    parser.add_argument('--language_code', type=str, default='en', help='Language in the input audio, default "en".')
    parser.add_argument('--task', type=str, default='transcribe', help='Either "transcribe" or "translate".')
    parser.add_argument('--device_name', type=str, default=None, help='Name for the input audio device.')
    parser.add_argument('--no_streaming', action='store_true', help='Disable streaming if flag is present.')
    args = parser.parse_args()
    return args


def demo():
    args = parse_args()
    
    model = load_model(args.model_path)
    options = DecodingOptions(
        task=args.task,
        language=args.language_code)
    streaming_decoder = StreamingDecoder(model, options)
    
    vad = Vad()
    vad.set_mode(3)

    incremental_transcriber = IncrementalTranscriber(streaming_decoder, vad, model.dims.n_mels, args.no_streaming)

    mic_input = AudioInput(device_name_like=args.device_name)
    mic_input.run(incremental_transcriber.update)


if __name__ == '__main__':
    demo()
