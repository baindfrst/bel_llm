"""Токенизатор. Команды: train, encode, chat."""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from net import BelTokenizer, encode_supervised, pad_supervised
from util import (
    IGNORE_INDEX,
    LABEL_DTYPE,
    SFT_LAYOUT,
    TOKEN_DTYPE,
    batched,
    bounded_imap,
    read_jsonl,
    words,
    write_json,
)

def build_sample(train_path: Path, out_txt: Path, target_lines: int, seed: int, max_chars: int) -> int:
    total = sum(len([p for p in rec["text"].split("\n") if p]) for rec in read_jsonl(train_path))
    prob = min(1.0, target_lines / max(total, 1))
    rng = random.Random(seed)
    n = 0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for rec in read_jsonl(train_path):
            for para in rec["text"].split("\n"):
                if para and len(para) <= max_chars and rng.random() < prob:
                    f.write(para + "\n")
                    n += 1
    return n


def train_one(sample_txt: Path, out_prefix: Path, vocab_size: int, args) -> None:
    import sentencepiece as spm

    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(sample_txt),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=args.character_coverage,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=True,
        max_sentence_length=args.max_chars * 4,
        input_sentence_size=args.sample_lines,
        shuffle_input_sentence=True,
        num_threads=args.threads,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        train_extremely_large_corpus=False,
    )


def cmd_train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train.jsonl.gz")
    ap.add_argument("--out-dir", default="tokenizers")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000])
    ap.add_argument("--sample-lines", type=int, default=4_000_000)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reuse-sample", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sample = out_dir / "spm_train_sample.txt"
    if not (args.reuse_sample and sample.exists()):
        build_sample(Path(args.train), sample, args.sample_lines, args.seed, args.max_chars)
    for v in args.vocab_sizes:
        tag = f"{v // 1000}k" if v % 1000 == 0 else str(v)
        train_one(sample, out_dir / f"bel_bpe_{tag}", v, args)
    write_json(out_dir / "train_params.json", vars(args))


_TOK = None


def _init(path: str) -> None:
    global _TOK
    _TOK = BelTokenizer(path)


def _encode_batch(texts: list[str]):
    arrs, chars, n_words = [], 0, 0
    for t in texts:
        arrs.append(np.asarray(_TOK.encode_document(t), dtype=np.uint16))
        chars += len(t) + 1  # +1 за разделитель документа
        n_words += len(words(t))
    return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.uint16), chars, n_words, len(texts)


def cmd_encode() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tok = BelTokenizer(args.tokenizer)
    assert tok.vocab_size < 2 ** 16, "uint16 поддерживает словарь до 65535"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "eos_id": tok.eos_id,
            "nl_id": tok.nl_id, "dtype": "uint16", "splits": {}}

    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for split in args.splits:
            t0 = time.time()
            src = Path(args.splits_dir) / f"{split}.jsonl.gz"
            dst = out_dir / f"{split}.bin"
            n_tok = n_chars = n_words = n_docs = 0
            texts = (rec["text"] for rec in read_jsonl(src))
            with open(dst, "wb") as f:
                for arr, c, w, d in bounded_imap(pool, _encode_batch, batched(texts, 512), 4 * args.workers):
                    f.write(arr.tobytes())
                    n_tok += len(arr)
                    n_chars += c
                    n_words += w
                    n_docs += d
            meta["splits"][split] = {"tokens": n_tok, "chars": n_chars, "words": n_words, "docs": n_docs}
    write_json(out_dir / "meta.json", meta)


def _pad_id(tok: BelTokenizer) -> int:
    pid = tok.pad_id
    if pid is None or pid < 0:
        raise SystemExit("нет pad_id")
    return int(pid)


def encode_split(tok: BelTokenizer, src: Path, dst_bin: Path, max_len: int) -> dict:
    pad_id = _pad_id(tok)
    ids_all, lab_all = [], []
    n_docs = n_skip = n_ans = n_pad = n_chars = n_words = 0
    for rec in read_jsonl(src):
        q, a = rec.get("question"), rec.get("answer")
        if not q or not a:
            continue
        ids, labels = encode_supervised(tok, q, a)
        padded = pad_supervised(ids, labels, max_len, pad_id)
        if padded is None:
            n_skip += 1
            continue
        ids, labels = padded
        ids_all.append(np.asarray(ids, dtype=TOKEN_DTYPE))
        lab_all.append(np.asarray(labels, dtype=LABEL_DTYPE))
        n_docs += 1
        n_ans += sum(1 for x in labels if x != IGNORE_INDEX)
        n_pad += sum(1 for t in ids if t == pad_id)
        n_chars += len(q) + len(a)
        n_words += len(words(q)) + len(words(a))
    if not ids_all:
        raise SystemExit(f"{src}: пусто")
    tok_arr = np.concatenate(ids_all)
    lab_arr = np.concatenate(lab_all)
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    tok_arr.tofile(dst_bin)
    lab_arr.tofile(dst_bin.with_name(dst_bin.stem + ".labels.bin"))
    return {
        "docs": n_docs,
        "skipped": n_skip,
        "tokens": int(len(tok_arr)),
        "answer_tokens": n_ans,
        "pad_tokens": n_pad,
        "answer_share": round(n_ans / max(len(tok_arr), 1), 4),
        "chars": n_chars,
        "words": n_words,
    }


def cmd_encode_chat() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--splits-dir", default="data/splits/chat")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    tok = BelTokenizer(args.tokenizer)
    if tok.vocab_size >= 2 ** 16:
        raise SystemExit("словарь не влезает в uint16")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_id,
        "pad_id": _pad_id(tok),
        "nl_id": tok.nl_id,
        "dtype": "uint16",
        "labels_dtype": "int32",
        "has_labels": True,
        "chat": True,
        "layout": SFT_LAYOUT,
        "seq_len": args.max_len,
        "ignore_index": IGNORE_INDEX,
        "splits": {},
    }
    for split in args.splits:
        src = Path(args.splits_dir) / f"{split}.jsonl.gz"
        if not src.exists():
            continue
        stats = encode_split(tok, src, out / f"{split}.bin", args.max_len)
        meta["splits"][split] = stats
    n_docs = int(meta["splits"].get("train", {}).get("docs") or 0)
    if n_docs:
        meta["recommended_max_steps_2ep"] = max(1, 2 * n_docs // MID_EXAMPLES_PER_STEP)
        meta["examples_per_step_mid"] = MID_EXAMPLES_PER_STEP
    write_json(out / "meta.json", meta)


def main() -> None:
    cmds = {"train": cmd_train, "encode": cmd_encode, "chat": cmd_encode_chat}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(2)
    fn = cmds[sys.argv[1]]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    fn()


if __name__ == "__main__":
    main()
