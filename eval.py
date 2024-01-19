import argparse
import time
import numpy as np

import audio
import mlx.core as mx
from decoding import StreamingDecoder, DecodingOptions
from whisper_mlx import load_model


def levenstein_distance(list1: list, list2: list):
    """Compute the Levenshtein distance between two lists."""
    if len(list1) < len(list2):
        return levenstein_distance(list2, list1)

    if len(list2) == 0:
        return len(list1)

    previous_row = np.arange(len(list2) + 1)
    for i, token1 in enumerate(list1):
        current_row = np.zeros(len(list2) + 1, dtype=int)
        current_row[0] = i + 1  # Incremental number of deletions for each row
        for j, token2 in enumerate(list2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1
            substitutions = previous_row[j] + (token1 != token2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


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

    num_tokens_all = []
    num_tokens_streaming = []
    levenstein_distances = []

    print("Starting eval")

    # Windows are size 3000 (30 seconds), stride 100 (1 second)
    for start_idx in range(0, mel_data.shape[1] - 2900, 100):
        end_idx = start_idx + 3000
        print(f"Start index: {start_idx}, end index: {end_idx}")

        mel_diff =  mel_data[:, end_idx-100:end_idx, :]
        mel_segment = mel_data[:, start_idx:end_idx, :]
        if mel_segment.shape[1] < 3000:
            padding = mx.zeros((mel_segment.shape[0], 3000 - mel_segment.shape[1], mel_segment.shape[2]))
            mel_segment = mx.concatenate([mel_segment, padding], axis=1)

        decoder_all = StreamingDecoder(model, options)
        
        start_time = time.time()
        result_all, tokens_all = decoder_all.incremental_decode(mel_segment)
        mx.eval()
        end_time = time.time()

        latency_all_ms = 1000 * (end_time - start_time)
        latencies_all.append(latency_all_ms)
        num_tokens_all.append(len(tokens_all))

        print(f"Result all: {result_all}")

        mel_segment_streaming = mel_segment if start_idx == 0 else mel_diff

        start_time = time.time()
        result_streaming, tokens_streaming = decoder_streaming.incremental_decode(mel_segment_streaming)
        mx.eval()
        end_time = time.time()
        
        latency_streaming_ms = 1000 * (end_time - start_time)
        latencies_streaming.append(latency_streaming_ms)
        num_tokens_streaming.append(len(tokens_streaming))

        print(f"Result streaming: {result_streaming}")
        
        latencies_diff.append(latency_all_ms - latency_streaming_ms)
        levenstein_distances.append(levenstein_distance(tokens_all.tolist(), tokens_streaming.tolist()))
        print(f"Levenstein distance: {levenstein_distances[-1]}")

    print(f"Latency all: {latency_all_ms:.2f}, latency streaming: {latency_streaming_ms:.2f}")
    print(f"Latency all mean: {np.mean(latencies_all):.2f} +/- {np.std(latencies_all):.2f}")
    print(f"Latency streaming mean: {np.mean(latencies_streaming):.2f} +/- {np.std(latencies_streaming):.2f}")
    print(f"Latency diff mean: {np.mean(latencies_diff):.2f} +/- {np.std(latencies_diff):.2f}")

    print(f"Num tokens all: {np.mean(num_tokens_all):.2f} +/- {np.std(num_tokens_all):.2f}")
    print(f"Num tokens streaming: {np.mean(num_tokens_streaming):.2f} +/- {np.std(num_tokens_streaming):.2f}")
    print(f"Levenstein distance mean: {np.mean(levenstein_distances):.2f} +/- {np.std(levenstein_distances):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio file.')
    parser.add_argument('--audio_path', type=str, help='Path to the audio file to evaluate.')
    parser.add_argument('--model_path', type=str, help='Path to the whisper mlx model.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval(args.audio_path, args.model_path)
