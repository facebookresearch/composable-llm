"""
Tokenizers

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import abc
import logging
import os
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from sentencepiece import SentencePieceProcessor

    has_sp = True
except ImportError:
    has_sp = False

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    has_tiktoken = True
except ImportError:
    has_tiktoken = False

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, sentence: str, add_bos: bool, add_eos: bool) -> list[int]:
        """
        Encode a sentence into a list of token IDs.

        Parameters
        ----------
        sentence:
            The sentence to encode.
        add_bos:
            Whether to add a BOS token at the beginning.
        add_eos:
            Whether to add an EOS token at the end.

        Returns
        -------
        The list of token IDs.
        """
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a sentence.

        Parameters
        ----------
        tokens:
            The list of token IDs to decode.

        Returns
        -------
        The decoded sentence.
        """
        pass

    @abc.abstractmethod
    def get_token_offsets(self, text: str, tokens: list[int] = None) -> tuple[list[str], list[int]]:
        """
        Return the offsets of the tokens in the original text.
        This is useful for evaluation.

        Parameters
        ----------
        text:
            The original text.
        tokens:
            The list of token IDs.

        Returns
        -------
        A tuple containing the list of substrings and the list of offsets.
        """
        pass


class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens: str, add_bos: bool, add_eos: bool) -> list[int]:
        return tokens


class ByteTokenizer(Tokenizer):
    encoding = "utf-8"

    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, sentence: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = [self.bos_id] * add_bos + list(sentence.encode(self.encoding)) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode(self.encoding, errors="backslashreplace")

    def get_token_offsets(self, text: str, tokens: list[int] = None) -> tuple[list[str], list[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool) -> list[int]:
        assert type(s) is str
        tokens = [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.sp_model.decode(tokens)

    def get_token_offsets(self, text: str, tokens: list[int] = None) -> tuple[list[str], list[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]"""
    r"""+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class TikTokenTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
        for id in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, add_bos: bool, add_eos: bool) -> list[int]:
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)

    def get_token_offsets(self, text: str, tokens: list[int] = None) -> tuple[list[str], list[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(self.tkt_model.encode(text, allowed_special="all"))

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets


def build_tokenizer(name: str, path: str = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        assert has_sp, "sentencepiece not installed"
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        assert has_tiktoken, "tiktoken not installed"
        return TikTokenTokenizer(path)
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
