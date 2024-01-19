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
        self.current_tokens = mx.array([[]], dtype=mx.int32)
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
    

    def _find_next_tokens_segment(self, tokens: mx.array, next_tokens: mx.array) -> Tuple[int, int]:
        """Looks for a matching initial segment of tokens"""

        for i in range(min(10, next_tokens.shape[1] - 5)):
            for j in range(min(10, tokens.shape[1] - 5)):
                if mx.all(next_tokens[0,i:i+5] == tokens[0,j:j+5]).item(): 
                    l = min(next_tokens.shape[1] - i, tokens.shape[1] - j)
                    matches = (next_tokens[0,i:i+l] == tokens[0,j:j+l]).tolist()
                    matching_length = matches.index(False) if False in matches else l
                    return i, i + matching_length
    
        return None, None
    

    def _check_previous_decode(self, audio_features: mx.array, tokens: mx.array) -> mx.array:
        assert tokens.shape[0] == 1, "only supports a batch size of 1"

        if tokens.size == 0:
            return tokens

        # Run model with the new audio features and the previous tokens
        inference_tokens =  mx.concatenate([
            mx.array(self.initial_tokens).reshape(1, -1),
            tokens
        ], axis=1)
        logits = self.inference.logits(inference_tokens, audio_features)
        self.inference.reset()

        logits = logits[:, self.sample_begin:]
        next_tokens = mx.argmax(logits, axis=-1)

        start_idx, end_idx = self._find_next_tokens_segment(tokens, next_tokens)

        if start_idx is None:
            return mx.array([[]], dtype=mx.int32)
        
        return next_tokens[:, start_idx:end_idx]


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
                    next_token = logits.argmax(axis=-1)
                else:
                    next_token = mx.random.categorical(logits=logits / self.temperature)

                completed = mx.any(next_token == self.eot)
                # Don't append eot as we'll continually update the tokens
                if not completed:
                    tokens = mx.concatenate([tokens, next_token[:, None]], axis=-1)

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

        # Call the main sampling loop to iteratively decode new tokens from current audio
        # First identify the tokens which can be reused from the previous decode
        verified_tokens = self._check_previous_decode(
            audio_features, self.current_tokens)
        inference_tokens =  mx.concatenate([
                 mx.array(self.initial_tokens).reshape(1, -1),
                 verified_tokens
        ], axis=1)
        tokens = self._main_loop(audio_features, inference_tokens)

        # Update list of tokens
        self.current_tokens = tokens[:, self.sample_begin:]

        return [self.tokenizer.decode(self.current_tokens.tolist()[0]).strip()], self.current_tokens[0]
