import argparse
import numpy as np
import time
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

    def __init__(self, streaming_decoder: StreamingDecoder, vad: Vad, n_mels: int):
        self.streaming_decoder = streaming_decoder
        self.vad = vad
        self.n_mels = n_mels

    def update(self, audio_bytes: bytes):
        processing_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        contains_speech = False
        for i in range(0, len(audio_bytes), 320 * 3):
            contains_speech = contains_speech or self.vad.is_speech(audio_bytes[i:i+(320 * 3)], 16000)

        if not contains_speech:
            num_mel_frames = len(audio_bytes) // 320
            self.streaming_decoder.add_empty_window(num_mel_frames)
            print(f"{processing_time} : [no speech detected]")
            return

        audio_arr = audio_bytes_to_np_array(audio_bytes)
        mel_arr = audio.log_mel_spectrogram(audio_arr, self.n_mels)
        mel_arr = mel_arr.reshape(1, *mel_arr.shape)
        result_streaming, _ = self.streaming_decoder.incremental_decode(mel_arr)
        print(f"{processing_time} : {result_streaming}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio file.')
    parser.add_argument('--model_path', type=str, help='Path to the whisper mlx model.')
    args = parser.parse_args()
    return args


def demo():
    args = parse_args()
    
    model = load_model(args.model_path)
    streaming_decoder = StreamingDecoder(model, DecodingOptions())
    
    vad = Vad()
    vad.set_mode(3)

    incremental_transcriber = IncrementalTranscriber(streaming_decoder, vad, model.dims.n_mels)

    mic_input = AudioInput()
    mic_input.run(incremental_transcriber.update)


if __name__ == '__main__':
    demo()
