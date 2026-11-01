#!/usr/bin/env python
"""Генерация. По умолчанию чат. Для сырого текста: --mode raw."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from util import read_jsonl, write_jsonl
from net import STOP_STRINGS, BelTokenizer, SamplingConfig, format_prompt, generate, load_model, strip_stop_strings


def complete(
    model,
    tok: BelTokenizer,
    prompt: str,
    sc: SamplingConfig,
    seed: int,
    prepend_eos: bool = True,
    chat: bool = True,
    history=None,
) -> str:
    text = format_prompt(prompt, history) if chat else prompt
    prompt_ids = tok.encode(text)
    ctx = ([tok.eos_id] if prepend_eos else []) + prompt_ids
    gen = torch.Generator().manual_seed(seed)
    out = generate(
        model,
        ctx,
        sc,
        eos_id=tok.eos_id,
        stop_strings=list(STOP_STRINGS) if chat else None,
        decode_fn=tok.decode if chat else None,
        generator=gen,
    )
    raw = tok.decode(out)
    return strip_stop_strings(raw) if chat else raw


def load_prompts(args) -> list[dict]:
    prompts = [{"id": f"cli{i}", "prompt": p} for i, p in enumerate(args.prompt or [])]
    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".jsonl":
            prompts += list(read_jsonl(path))
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            prompts += [{"id": f"line{i}", "prompt": ln} for i, ln in enumerate(lines) if ln.strip()]
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--prompts-file")
    ap.add_argument("--mode", choices=["chat", "raw"], default="chat")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--no-repeat-ngram", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    chat = args.mode == "chat"
    if chat:
        sc = SamplingConfig(
            max_new_tokens=args.max_new_tokens or 80,
            temperature=0.4 if args.temperature is None else args.temperature,
            top_k=40 if args.top_k is None else args.top_k,
            top_p=0.85 if args.top_p is None else args.top_p,
            repetition_penalty=1.15 if args.repetition_penalty is None else args.repetition_penalty,
            no_repeat_ngram_size=3 if args.no_repeat_ngram is None else args.no_repeat_ngram,
            use_cache=not args.no_cache,
        )
    else:
        sc = SamplingConfig(
            max_new_tokens=args.max_new_tokens or 120,
            temperature=0.8 if args.temperature is None else args.temperature,
            top_k=50 if args.top_k is None else args.top_k,
            top_p=0.95 if args.top_p is None else args.top_p,
            repetition_penalty=1.1 if args.repetition_penalty is None else args.repetition_penalty,
            no_repeat_ngram_size=0 if args.no_repeat_ngram is None else args.no_repeat_ngram,
            use_cache=not args.no_cache,
        )

    model, ck = load_model(args.ckpt, args.device)
    tok = BelTokenizer(args.tokenizer or ck.get("tokenizer"))
    if tok.vocab_size != model.cfg.vocab_size:
        raise ValueError("Токенизатор не соответствует модели")
    prompts = load_prompts(args)
    if not prompts:
        ap.error("Нужен --prompt или --prompts-file")

    results = []
    for pi, p in enumerate(prompts):
        user = p.get("prompt") or p.get("question") or ""
        for s in range(args.num_samples):
            text = complete(model, tok, user, sc, seed=args.seed + 1000 * pi + s, chat=chat)
            results.append({"id": p.get("id", f"p{pi}"), "sample": s, "prompt": user, "completion": text, "mode": args.mode})
            print(f"### [{p.get('id', pi)}#{s}]\n{user}\u2502{text}\n")
    if args.out:
        write_jsonl(args.out, results)
        print(f"Сохранено: {args.out}")
    else:
        sys.stdout.flush()


if __name__ == "__main__":
    main()
