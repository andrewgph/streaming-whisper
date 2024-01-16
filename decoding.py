# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from tokenizer import Tokenizer, get_tokenizer

from audio import N_FRAMES


@dataclass(frozen=True)
class DecodingOptions:
    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # implementation details
    fp16: bool = False  # use fp16 for most of the calculation


class Inference:
    def __init__(self, model: "Whisper"):
        self.model: "Whisper" = model
        self.num_inference_calls = 0
        self.kv_cache = None

    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Perform a forward pass on the decoder and return per-token logits"""
        # only need to use the last token except in the first forward pass
        if self.num_inference_calls > 0:
            tokens = tokens[:, -1:]
        self.num_inference_calls += 1

        logits, self.kv_cache, _ = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        return logits.astype(mx.float32)

    def reset(self):
        self.num_inference_calls = 0
        self.kv_cache = None


class LogitFilter:
    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply any filtering or masking to logits

        Parameters
        ----------
        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int, n_vocab: int):
        self.sample_begin = sample_begin
        mask = np.zeros(n_vocab, np.float32)
        mask[tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if tokens.shape[1] == self.sample_begin:
            return logits + self.mask
        return logits


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int], n_vocab: int):
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppress_tokens)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        return logits + self.mask


class StreamingDecoder:
    logit_filters: List[LogitFilter]
    inference: Inference


    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model
        self.options = options

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task="transcribe",
        )
        self.tokenizer: Tokenizer = tokenizer

        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2
        self.temperature = options.temperature

        # Only supports notimestamps, intention is that streaming would determine timestamps as it goes
        self.sot_sequence = tokenizer.sot_sequence_including_notimestamps
        self.eot = tokenizer.eot

        self.initial_tokens: Tuple[int] = list(self.sot_sequence)
        self.sample_begin: int = len(self.initial_tokens)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = Inference(model)

        # Only support a batch size of 1
        self.current_tokens = mx.array(list(self.sot_sequence)).reshape(1, -1)
        self.mel = mx.zeros((1, N_FRAMES, model.dims.n_mels),
                            dtype=mx.float16 if self.options.fp16 else mx.float32)
        self.num_inference_calls = 0

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(self.tokenizer, self.sample_begin, model.dims.n_vocab)
            )
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens(), model.dims.n_vocab)
            )


    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))


    def _get_audio_features(self, mel: mx.array) -> mx.array:
        if self.options.fp16:
            mel = mel.astype(mx.float16)

        audio_features = self.model.encoder(mel)

        if audio_features.dtype != (mx.float16 if self.options.fp16 else mx.float32):
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features


    def _main_loop(self, audio_features: mx.array, tokens: mx.array) -> mx.array:
        try: 
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                # Only consider the logits at the last token
                logits = logits[:, -1]

                # Apply the logit filters, e.g. for suppressing or applying penalty
                for logit_filter in self.logit_filters:
                    logits = logit_filter.apply(logits, tokens)

                # Greedy decoding of next token
                if self.temperature == 0:
                    next_tokens = logits.argmax(axis=-1)
                else:
                    next_tokens = mx.random.categorical(logits=logits / self.temperature)
                next_tokens = mx.argmax(logits, axis=-1)

                completed = mx.any(next_tokens == self.eot)
                # Don't append eot as we'll continually append to the transcript
                if not completed:
                    tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.reset()

        return tokens
    

    def incremental_decode(self, new_mel: mx.array) -> str:
        assert new_mel.shape[0] == 1, "streaming only a single audio stream is supported"

        # Append new audio to existing audio
        self.mel = mx.concatenate([self.mel, new_mel], axis=1)

        # Trim to the most recent 30 seconds
        if self.mel.shape[1] > N_FRAMES:
            self.mel = self.mel[:, -N_FRAMES:, :]
        
        # Encoder forward pass on new audio
        audio_features: mx.array = self._get_audio_features(self.mel)

        # Drop old tokens
        # TODO: need to align this with the dropped audio
        if self.current_tokens.shape[-1] > 60:
            self.current_tokens = mx.concatenate([
                 mx.array(list(self.sot_sequence)).reshape(1, -1),
                 self.current_tokens[:, self.sample_begin:][:, -60:]
            ], axis=1)

        # Call the main sampling loop to iteratively decode new tokens from current audio
        # TODO: redo the last few tokens, in case there is a better decoding for them now
        # TODO: identify recent tokens that might have been misheard
        last_token_idx = max(self.sample_begin, self.current_tokens.shape[1] - 20)
        inference_tokens = self.current_tokens[:, :last_token_idx]
        tokens = self._main_loop(audio_features, inference_tokens)

        # Update list of tokens
        self.current_tokens = tokens

        return [self.tokenizer.decode(tokens.tolist()[0]).strip()]

