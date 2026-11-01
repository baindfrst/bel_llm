#!/usr/bin/env python
"""Обучение. python train.py --config configs/base.yaml"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from util import (
    ChunkSampler,
    PackedTokens,
    evaluate_bin,
    labels_path_for,
    load_config,
    looks_like_sft,
    read_token_meta,
    save_config,
    write_json,
)
from net import (
    BelarusianLM,
    ModelConfig,
    assert_init_compatible,
    export_model,
    resolve_weight_path,
    unwrap_state_dict,
)


def lr_at(step: int, t: dict) -> float:
    base, warmup, total = float(t["lr"]), int(t["warmup_steps"]), int(t["max_steps"])
    min_lr = base * float(t.get("min_lr_ratio", 0.1))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module, t: dict, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": float(t["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=float(t["lr"]), betas=(float(t["beta1"]), float(t["beta2"])), eps=float(t.get("eps", 1e-8)))
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


class Logger:
    def __init__(self, run_dir: Path, enabled: bool):
        self.enabled = enabled
        self.f = open(run_dir / "metrics.jsonl", "a", encoding="utf-8") if enabled else None

    def log(self, record: dict) -> None:
        if not self.enabled:
            return
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        if self.f:
            self.f.close()


def save_checkpoint(path: Path, raw_model, optimizer, scaler, step, tokens_seen, sampler, best_val, cfg) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": raw_model.state_dict(),
        "model_config": raw_model.cfg.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "sampler": sampler.state_dict(),
        "best_val_loss": best_val,
        "config": cfg,
        "tokenizer": cfg["data"].get("tokenizer"),
    }, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    t, d = cfg["train"], cfg["data"]

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device(t.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    master = rank == 0
    torch.manual_seed(int(cfg.get("seed", 1337)) + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = resolve_dtype(t.get("dtype", "auto"), device)
    amp_ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype != torch.float32 else nullcontext()
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    run_dir = Path(cfg.get("out_dir", "runs")) / cfg["name"]
    if master:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")

    mcfg = ModelConfig.from_dict(cfg["model"])
    meta = read_token_meta(d["train_bin"])
    if meta and meta.get("vocab_size") != mcfg.vocab_size:
        raise ValueError(f"vocab_size модели {mcfg.vocab_size} != vocab_size данных {meta.get('vocab_size')}")
    seq_len = mcfg.max_seq_len
    if looks_like_sft(d["train_bin"], meta) and not labels_path_for(d["train_bin"]).exists():
        raise ValueError(f"{d['train_bin']}: нет labels.bin")
    train_ds = PackedTokens(d["train_bin"], seq_len)
    if looks_like_sft(d["train_bin"], meta) and not train_ds.has_labels:
        raise ValueError(f"{d['train_bin']}: нет labels.bin")
    micro_bs, accum = int(t["micro_batch_size"]), int(t["grad_accum_steps"])
    sampler = ChunkSampler(train_ds, micro_bs, seed=int(cfg.get("seed", 1337)), rank=rank, world_size=world)
    tokens_per_step = micro_bs * seq_len * accum * world

    raw_model = BelarusianLM(mcfg).to(device)
    if args.init_from:
        p = resolve_weight_path(args.init_from)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "model" not in ck:
            raise SystemExit(f"--init-from {p}: в файле нет ключа model")
        if "model_config" in ck:
            assert_init_compatible(mcfg, ck["model_config"], p)
        state = unwrap_state_dict(ck["model"])
        raw_model.load_state_dict(state, strict=True)
    raw_model.gradient_checkpointing = bool(t.get("gradient_checkpointing", False))
    optimizer = build_optimizer(raw_model, t, device)
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    ckpt_last = run_dir / "ckpt_last.pt"
    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        sampler.load_state_dict(ck["sampler"])
        start_step, tokens_seen, best_val = ck["step"], ck["tokens_seen"], ck["best_val_loss"]

    model = raw_model
    if t.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[device.index])

    n_params = raw_model.num_parameters()
    n_body = raw_model.num_parameters(exclude_embeddings=True)
    logger = Logger(run_dir, master)
    if master:
        logger.log({"type": "meta", "step": start_step, "params": n_params, "params_non_embedding": n_body,
                    "tokens_per_step": tokens_per_step, "describe": mcfg.describe()})

    def run_eval(step: int, max_tokens=None, split="val") -> dict:
        bin_path = d["val_bin"] if split == "val" else d["test_bin"]
        res = evaluate_bin(raw_model, bin_path, seq_len, int(t.get("eval_batch_size", micro_bs)), device, amp_ctx,
                           max_tokens=max_tokens)
        return res

    def eval_and_maybe_best(step: int, max_tokens=None) -> dict:
        nonlocal best_val
        ev = run_eval(step, max_tokens)
        is_best = ev["loss"] < best_val
        best_val = min(best_val, ev["loss"])
        logger.log({"type": "eval", "step": step, "tokens_seen": tokens_seen, "val_loss": ev["loss"],
                    "val_ppl": ev["ppl"], "val_bpc": ev.get("bits_per_char")})
        if is_best and step > 0:
            export_model(raw_model, run_dir / "model_best.pt", tokenizer=d.get("tokenizer"), step=step,
                         name=cfg["name"], tokens_seen=tokens_seen)
        return ev

    max_steps = int(t["max_steps"])
    log_every, eval_every = int(t.get("log_interval", 20)), int(t.get("eval_interval", 500))
    save_every = int(t.get("save_interval", 1000))
    eval_tokens = int(t.get("eval_tokens", 2_000_000))
    clip = float(t.get("grad_clip", 1.0))
    model.train()
    t_interval, steps_interval = time.time(), 0
    t_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step, max_steps):
        if master and eval_every > 0 and step % eval_every == 0 and (step > 0 or t.get("eval_at_start", True)):
            eval_and_maybe_best(step, eval_tokens)

        lr = lr_at(step, t)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_sum = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = sampler.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sync = model.no_sync() if (ddp and micro < accum - 1) else nullcontext()
            with sync:
                with amp_ctx:
                    _, loss = model(x, targets=y)
                scaler.scale(loss / accum).backward()
            loss_sum += loss.detach()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step
        steps_interval += 1

        if (step + 1) % log_every == 0 or step == max_steps - 1:
            loss_avg = loss_sum / accum
            if ddp:
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.time() - t_interval
            if master:
                rec = {
                    "type": "train", "step": step + 1, "loss": float(loss_avg),
                    "ppl": math.exp(min(float(loss_avg), 50.0)), "lr": lr, "grad_norm": float(grad_norm),
                    "tokens_per_sec": steps_interval * tokens_per_step / dt, "tokens_seen": tokens_seen,
                    "elapsed_sec": time.time() - t_start,
                }
                if device.type == "cuda":
                    rec["gpu_mem_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    rec["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
                    torch.cuda.reset_peak_memory_stats(device)
                logger.log(rec)
                print(f"step {step + 1}/{max_steps} loss {rec['loss']:.4f} lr {lr:.2e} {rec['tokens_per_sec'] / 1e3:.1f}k tok/s")
            t_interval, steps_interval = time.time(), 0

        if master and save_every > 0 and (step + 1) % save_every == 0:
            save_checkpoint(ckpt_last, raw_model, optimizer, scaler, step + 1, tokens_seen, sampler, best_val, cfg)

    if master:
        save_checkpoint(ckpt_last, raw_model, optimizer, scaler, max_steps, tokens_seen, sampler, best_val, cfg)
        export_model(raw_model, run_dir / "model_final.pt", tokenizer=d.get("tokenizer"), step=max_steps,
                     name=cfg["name"], tokens_seen=tokens_seen)
        if eval_every > 0:
            eval_and_maybe_best(max_steps, eval_tokens)
        final_tokens = t.get("final_eval_tokens")
        val = run_eval(max_steps, final_tokens, "val")
        test = run_eval(max_steps, final_tokens, "test")
        final = {
            "name": cfg["name"], "describe": mcfg.describe(), "params": n_params, "params_non_embedding": n_body,
            "vocab_size": mcfg.vocab_size, "n_layers": mcfg.n_layers, "d_model": mcfg.d_model,
            "tokens_seen": tokens_seen, "train_hours": (time.time() - t_start) / 3600,
            "val_loss": val["loss"], "val_ppl": val["ppl"], "val_bpc": val.get("bits_per_char"),
            "test_loss": test["loss"], "test_ppl": test["ppl"], "test_bpc": test.get("bits_per_char"),
            "test_word_ppl": test.get("word_ppl"),
        }
        write_json(run_dir / "final_metrics.json", final)
        logger.log({"type": "final", "step": max_steps, **{k: v for k, v in final.items() if isinstance(v, (int, float))}})
    logger.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
