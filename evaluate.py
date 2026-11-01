#!/usr/bin/env python
"""ppl — loss/bpc на bin. gen — ответы по eval_prompts.jsonl. all — оба."""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml

from util import aggregate_generation_metrics, evaluate_bin, read_json, read_jsonl, write_json, write_jsonl
from generate import complete
from net import BelTokenizer, SamplingConfig, format_prompt, load_model


def run_ppl(model, ecfg: dict, device, split_paths: dict) -> dict:
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    out = {}
    for split in ecfg["ppl"]["splits"]:
        path = Path(split_paths[split]) if split in split_paths else None
        if path is None or not path.exists():
            print(f"[warn] нет {split}: {path}")
            continue
        res = evaluate_bin(model, path, model.cfg.max_seq_len, ecfg["ppl"]["batch_size"], device, amp, max_tokens=ecfg["ppl"].get("max_tokens"), split_name=split)
        out[split] = res
        if "bits_per_char" in res:
            print(f"{split} loss {res['loss']:.4f} ppl {res['ppl']:.2f} bpc {res['bits_per_char']:.4f}")
        else:
            print(f"{split} loss {res['loss']:.4f} ppl {res['ppl']:.2f}")
    return out


def _qa_hit(completion: str, patterns: list[str], window: int) -> bool:
    head = completion[:window].lower()
    return any(re.search(p, head) for p in patterns)


def run_generation(model, tok: BelTokenizer, ecfg: dict) -> tuple[list[dict], dict]:
    prompts = list(read_jsonl(ROOT / ecfg["prompts"]))
    g, q = ecfg["generation"], ecfg["qa"]
    chat = bool(ecfg.get("chat", True))
    sc_gen = SamplingConfig(max_new_tokens=g["max_new_tokens"], temperature=g["temperature"], top_k=g["top_k"],
                            top_p=g["top_p"], repetition_penalty=g["repetition_penalty"],
                            no_repeat_ngram_size=g.get("no_repeat_ngram_size", 0 if not chat else 3))
    sc_qa = SamplingConfig(max_new_tokens=q["max_new_tokens"], temperature=q["temperature"],
                           repetition_penalty=q.get("repetition_penalty", 1.0),
                           no_repeat_ngram_size=q.get("no_repeat_ngram_size", 0))
    base_seed = int(ecfg.get("seed", 1234))
    records = []
    for pi, p in enumerate(prompts):
        if p["category"] == "qa":
            user = p["prompt"]
            shown = format_prompt(user) if chat else q["question_template"].format(prompt=user)
            comp = complete(model, tok, user if chat else shown, sc_qa, seed=base_seed + pi, chat=chat)
            records.append({**p, "input": shown, "sample": 0, "completion": comp,
                            "hit": _qa_hit(comp, p["answers"], q["answer_window_chars"])})
        else:
            user = p["prompt"] if chat else p["prompt"]
            if chat and p.get("style") == "completion":
                user = f"Працягні: {p['prompt']}"
            for s in range(g["samples_per_prompt"]):
                comp = complete(model, tok, user, sc_gen, seed=base_seed + 1000 * pi + s, chat=chat)
                records.append({**p, "input": format_prompt(user) if chat else user, "sample": s, "completion": comp})

    free = [r for r in records if r["category"] != "qa"]
    metrics = {"all": aggregate_generation_metrics([r["completion"] for r in free])}
    by_cat = defaultdict(list)
    for r in free:
        by_cat[r["category"]].append(r["completion"])
    metrics["by_category"] = {c: aggregate_generation_metrics(v) for c, v in by_cat.items()}

    qa = [r for r in records if r["category"] == "qa"]
    groups = defaultdict(list)
    for r in qa:
        groups[r["group"]].append(r["hit"])
    metrics["qa"] = {
        "accuracy": sum(r["hit"] for r in qa) / len(qa) if qa else 0.0,
        "groups": len(groups),
        "group_all_correct": sum(all(v) for v in groups.values()) / len(groups) if groups else 0.0,
        "group_any_correct": sum(any(v) for v in groups.values()) / len(groups) if groups else 0.0,
        "group_consistency": sum(len(set(v)) == 1 for v in groups.values()) / len(groups) if groups else 0.0,
        "per_group": {k: sum(v) / len(v) for k, v in groups.items()},
    }
    return records, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["ppl", "gen", "all"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.eval_config, encoding="utf-8") as f:
        ecfg = yaml.safe_load(f)
    ckpt = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ck = load_model(ckpt, device)
    name = ck.get("name", ckpt.parent.name)
    print(f"{name} {model.num_parameters() / 1e6:.2f}M {model.cfg.describe()}")

    summary_path = out_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary.update({"name": name, "params": model.num_parameters(), "describe": model.cfg.describe()})

    if args.mode in ("ppl", "all"):
        if args.data_dir:
            data_dir = Path(args.data_dir)
            split_paths = {s: data_dir / f"{s}.bin" for s in ecfg["ppl"]["splits"]}
        else:
            run_cfg = ckpt.parent / "config.yaml"
            if not run_cfg.exists():
                ap.error("Укажите --data-dir")
            with open(run_cfg, encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)["data"]
            split_paths = {"validation": Path(data_cfg["val_bin"]), "test": Path(data_cfg["test_bin"])}
        summary["ppl"] = run_ppl(model, ecfg, device, split_paths)

    if args.mode in ("gen", "all"):
        tok = BelTokenizer(args.tokenizer or ck.get("tokenizer"))
        records, metrics = run_generation(model, tok, ecfg)
        write_jsonl(out_dir / "generations.jsonl", records)
        summary["generation"] = metrics
        m, qa = metrics["all"], metrics["qa"]
        print(f"QA {qa['accuracy']:.3f} bel {m['mean_bel_word_ratio']:.3f}")

    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
