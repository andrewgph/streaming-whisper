# Streaming Whisper (in MLX)

This project is an experiment in making the Whisper model faster for streaming use cases. Most of the [examples](https://github.com/openai/whisper/discussions/2) I've seen of using running Whisper in streaming mode involve continually re-running transcription over the most recent 30 second window with a small shift. Intuitively there are overlapping transcripts between windows so there should be some re-usable computation.

## Overview of Whisper model and streaming issues

![Whisper Model Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

The main problem for faster streaming is how you can re-use artifacts generated when processing a previous window in the current window. For example if you're re-running transcription of a 30 second window every 1 second then there is a 29 second overlap between consecutive windows. For real-time use cases you'd ideally re-run transcription on a very short interval (say 500ms) so you can transcribe and react to speech as soon as possible.

Log-Mel Spectrogram: The conversion is independent for each frame, so you can incrementally convert frames of audio as you receive them.

Encoder: The encoder portion can't be updated incrementally. The encoding of one frame can depend upon all the others in the 30 second window.

Decoder: This is the most expensive part of the model as you repeatedly run the decoder to get output tokens. The output for the next window should overlap with the previous one. How can the decoding be re-used?

## Implementation

The idea implemented here (see [decoding.py](decoding.py#L210)) is to:
1. Run the decoder with the last window's tokens.
2. Take the highest probability next tokens over this decoder result.
3. Check if these tokens have an overlapping segment with the previous tokens.
4. Extract this segment of tokens and use them as an initial token sequence for decoding the current window.

The intuition is that if you're re-running transcription with only a small shift (say 500ms to 3 seconds) that only a small portion of the previous transcription would no longer be in the window. Re-running the previous text decoding with the new audio encoding can verify which tokens should be dropped and how many tokens are still in the window.

This is similar to speculative decoding, where a small draft model is used to generate token sequences which are verified by a larger model. Except in this case the draft model is just the same model over the previous window.

The implementation uses MLX as this was a simple way to experiment with ideas on a macbook. The same implementation could be done in the PyTorch version of the model.

### Alternative ideas considered

Initially I used some simple heuristics for dropping tokens from the beginning and end of the previous transcription. The main problem here is identifying which tokens should still be present in the window. You know how much audio was added or dropped so you could use [word-level timestamps](https://github.com/openai/whisper/discussions/813) to identify which tokens are no longer in the window, or which are are close to the end and so might have been misheard in the previous window. The implementation of word-level timestamps uses cross-attention activations and Dynamic Time Warping. I considered just using the cross-attention activations and a simpler heuristic to identify tokens in the initial X seconds of audio from previous window. Although this seemed unnecessary as the  technique described above seems robust enough to handle a few initial tokens which are no longer in the window. It might still be useful if the gaps between windows are larger (maybe 5+ seconds).

## Usage

First install dependencies:

```
pip install -r requirements.txt
```

You also need to setup a MLX version of the whisper models. See the [mlx/examples repo](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for instructions.

### Eval

Example usage of eval.py script:

```
$ wget https://upload.wikimedia.org/wikipedia/commons/transcoded/6/6f/Apollo13-wehaveaproblem.ogg/Apollo13-wehaveaproblem.ogg.mp3 
$ python eval.py --audio_path Apollo13-wehaveaproblem.ogg.mp3 --model_path ~/mlx_models/whisper-large-v3
...
Latency all: 3820.00, latency streaming: 2212.32
Latency all mean: 3055.68 +/- 1355.68
Latency streaming mean: 2102.05 +/- 1106.31
Latency diff mean: 693.04 +/- 468.92
Num tokens all: 61.75 +/- 32.24
Num tokens streaming: 56.50 +/- 29.95
Word Error Rate mean: 9.47% +/- 5.66
```

This goes over the audio file in 30 second windows with 1 second gap between them. It compares running decoding over the entire window from scratch with running incremental decoding.

In general the latency for incremental decoding is lower, both in mean and variance, which is good for streaming use cases.

The word error rate is used to compare the results of the two decodings. Ideally the two decodings would be the same, so the incremental decoding is giving you the same result as running a full decoding from scratch. In practice there are differences. I'm guessing this is because of differences in the greedy decoding outcomes given different initial token sequences. The quality of transcription seems reasonable when manually reviewing them.

## Demo scripts

Example usage of demo.py script:

```
$ python demo.py --model_path ~/mlx_models/whisper-large-v3
```

Can also use it for translation:

```
$ python demo.py --model_path /Users/andrew/mlx_models/whisper-large-v3 --task translate --language_code zh
```

Or turn off the streaming decoding to experience the difference in transcription speed:

```
$ python demo.py --model_path ~/mlx_models/whisper-large-v3 --no_streaming
```

## Ideas for further work

Beam search: The MLX example implementation didn't include beam search. It also makes the code more complicated and I wanted a small codebase to experiment with, so I didn't include it.

Speculative decoding using a smaller model seems like a nice improvement that could stack with the implementation here. There is an example in [distil-whisper](https://github.com/huggingface/distil-whisper).

The cut and merge technique used in [WhisperX](https://github.com/m-bain/whisperX) could be used here. That might help with hallucination issues when there is little speech in the recent audio.

There doesn't seem to be a technical restriction to using a window shorter than 30 seconds. You can modify the sinusoids positional encoder for the AudioEncoder to have a shorter length, then run the audio encoder for a window of that length. The decoder can operate on a shorter audio window, although I haven't investigated how this impacts accuracy.

One observation is that the Whisper model is designed to operate over a longer window. This is useful when the word positions shouldn't necessarily be transcribed in their spoken order (see [discussion](https://github.com/openai/whisper/discussions/943#discussion-4836042)). For example "30 million dollars" is transcribed as "$30 million". Also the model was designed to perform translation, where the word order can be different between languages. Other model architectures would be better for streaming.