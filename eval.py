import argparse
import time
import numpy as np

import audio
import mlx.core as mx
from decoding import StreamingDecoder, DecodingOptions
from whisper_mlx import load_model


def eval(audio_path: str, model_path: str):
    model = load_model(model_path)
    options = DecodingOptions()
    decoder_streaming = StreamingDecoder(model, options)

    print("Loaded model")

    audio_data = audio.load_audio(audio_path)
    mel_data = audio.log_mel_spectrogram(audio_data, n_mels=model.dims.n_mels)
    mel_data = mel_data.reshape(1, *mel_data.shape)

    print("Loaded audio")

    latencies_all = []
    latencies_streaming = []
    latencies_diff = []

    print("Starting eval")

    for start_idx in range(0, mel_data.shape[1], 100):
        end_idx = start_idx + 3000
        print(f"Start index: {start_idx}, end index: {end_idx}")

        mel_diff =  mel_data[:, end_idx-100:end_idx, :]
        mel_segment = mel_data[:, start_idx:end_idx, :]
        if mel_segment.shape[1] < 3000:
            padding = mx.zeros((mel_segment.shape[0], 3000 - mel_segment.shape[1], mel_segment.shape[2]))
            mel_segment = mx.concatenate([mel_segment, padding], axis=1)

        decoder_all = StreamingDecoder(model, options)
        start_time = time.time()
        result_all = decoder_all.incremental_decode(mel_segment)
        mx.eval()
        print(f"Result all: {result_all}", flush=True)
        end_time = time.time()
        latency_all_ms = 1000 * (end_time - start_time)
        latencies_all.append(latency_all_ms)

        mel_segment_streaming = mel_segment if start_idx == 0 else mel_diff
        start_time = time.time()
        result_streaming = decoder_streaming.incremental_decode(mel_segment_streaming)
        mx.eval()
        print(f"Result streaming: {result_streaming}", flush=True)
        end_time = time.time()
        latency_streaming_ms = 1000 * (end_time - start_time)
        latencies_streaming.append(latency_streaming_ms)
        
        latencies_diff.append(latency_all_ms - latency_streaming_ms)

        print(f"Latency all: {latency_all_ms:.2f}, latency streaming: {latency_streaming_ms:.2f}", flush=True)
        print(f"Latency all mean: {np.mean(latencies_all):.2f} +/- {np.std(latencies_all):.2f}", flush=True)
        print(f"Latency streaming mean: {np.mean(latencies_streaming):.2f} +/- {np.std(latencies_streaming):.2f}", flush=True)
        
        # TODO: compare accuracy of transcription for all vs streaming
    
    print(f"Average latency all: {sum(latencies_all) / len(latencies_all)}")
    print(f"Average latency streaming: {sum(latencies_streaming) / len(latencies_streaming)}")
    print(f"Average latency diff: {sum(latencies_diff) / len(latencies_diff)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio file.')
    parser.add_argument('--audio_path', type=str, help='Path to the audio file to evaluate.')
    parser.add_argument('--model_path', type=str, help='Path to the whisper mlx model.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval(args.audio_path, args.model_path)
