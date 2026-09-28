"""Tiny, offline, deterministic model builder for behavioral-pipeline tests.

Uses a randomly-initialized (seeded) GPT2-architecture model with a minimal
hand-rolled character-level tokenizer -- no network access or pretrained-
weight download required. This is intentionally not a good language model;
it exists to exercise the generate -> decode -> compare pipeline end-to-end
with real torch/transformers objects.
"""

from __future__ import annotations

import string

import torch


class _BatchEncoding(dict):
    def to(self, device: str) -> _BatchEncoding:
        return _BatchEncoding({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class SimpleCharTokenizer:
    """Character-level tokenizer over a fixed, small ASCII vocabulary."""

    def __init__(self) -> None:
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        chars = list(string.printable)
        vocab_list = specials + [c for c in chars if c not in specials]
        self.char_to_id = {c: i for i, c in enumerate(vocab_list)}
        self.id_to_char = dict(enumerate(vocab_list))
        self.pad_token_id = self.char_to_id["<pad>"]
        self.bos_token_id = self.char_to_id["<bos>"]
        self.eos_token_id = self.char_to_id["<eos>"]
        self.unk_token_id = self.char_to_id["<unk>"]
        self.vocab_size = len(vocab_list)

    def __call__(self, text: str, return_tensors: str = "pt") -> _BatchEncoding:
        ids = [self.char_to_id.get(c, self.unk_token_id) for c in text]
        ids = [self.bos_token_id, *ids]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return _BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        chars = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in special_ids:
                continue
            chars.append(self.id_to_char.get(i, ""))
        return "".join(chars)


def build_tiny_model(seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    tokenizer = SimpleCharTokenizer()
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, tokenizer
