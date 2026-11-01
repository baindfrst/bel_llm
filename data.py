"""Сбор корпуса. Команды: hf, alpaca, wiki-old, convert, clean, dedup, split, check, chat."""
from __future__ import annotations

import argparse
import collections
import fnmatch
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from net import parse_qa_text
from util import (
    JsonlWriter,
    WORD_RE,
    batched,
    bounded_imap,
    classify_text,
    looks_broken,
    looks_like_html,
    looks_like_wikitext,
    normalize_text,
    read_jsonl,
    remove_urls_emails,
    strip_html,
    strip_wikitext,
    taraskevica_score,
    words,
    write_json,
)

def iter_parquet(path: str, columns: list[str]):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in available]
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        yield from batch.to_pylist()


def iter_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_records(path: str, columns: list[str]):
    if path.endswith(".parquet"):
        yield from iter_parquet(path, columns)
    elif path.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
        yield from iter_jsonl(path)
    else:
        raise ValueError(f"Неизвестный формат: {path}")


def list_files(repo_id: str, pattern: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if fnmatch.fnmatch(f, pattern))


def download_source(src: dict, raw_dir: Path, cache_dir: Path, delete_shards: bool, overwrite: bool) -> None:
    from huggingface_hub import hf_hub_download

    out_path = raw_dir / f"{src['name']}.jsonl.gz"
    if out_path.exists() and not overwrite:
        return
    files = list_files(src["repo_id"], src["files"])
    if not files:
        raise RuntimeError(f"{src['name']}: нет файлов по шаблону {src['files']} в {src['repo_id']}")

    text_f = src.get("text_field", "text")
    id_f, url_f, title_f = src.get("id_field"), src.get("url_field"), src.get("title_field")
    columns = [c for c in (text_f, id_f, url_f, title_f) if c]
    max_bytes = float(src["max_raw_mb"]) * 1e6 if src.get("max_raw_mb") else float("inf")

    written_bytes, n = 0, 0
    with JsonlWriter(out_path) as w:
        for fi, fname in enumerate(files):
            local = hf_hub_download(src["repo_id"], fname, repo_type="dataset", cache_dir=str(cache_dir))
            for rec in iter_records(local, columns):
                text = rec.get(text_f)
                if not text:
                    continue
                doc_id = rec.get(id_f) if id_f else None
                w.write({
                    "id": f"{src['name']}:{doc_id if doc_id is not None else n}",
                    "source": src["name"],
                    "text": text,
                    "url": rec.get(url_f) if url_f else None,
                    "title": rec.get(title_f) if title_f else None,
                })
                n += 1
                written_bytes += len(text.encode("utf-8"))
                if written_bytes >= max_bytes:
                    break
            if delete_shards:
                real = os.path.realpath(local)
                for p in {local, real}:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if written_bytes >= max_bytes:
                break


def cmd_hf() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--delete-shards", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg.get("raw_dir", "data/raw"))
    cache_dir = Path(cfg.get("hf_cache_dir", "data/hf_cache"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    for src in cfg["sources"]:
        if src.get("kind") != "hf":
            continue
        if args.only:
            if src["name"] not in args.only:
                continue
        elif not src.get("enabled", True):
            continue
        download_source(src, raw_dir, cache_dir, args.delete_shards, args.overwrite)


REPO = "saillab/alpaca-belarusian-cleaned"
SOURCE = "alpaca_be"


def cmd_alpaca() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/alpaca_be.jsonl.gz")
    ap.add_argument("--split", default="train")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(f"data/{args.split}-")]
    if not files:
        raise SystemExit(f"Сплит {args.split} не найден в {REPO}")
    files.sort()

    n = 0
    with JsonlWriter(out) as w:
        for fi, fname in enumerate(files):
            local = hf_hub_download(REPO, fname, repo_type="dataset", cache_dir="data/hf_cache")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1024):
                for rec in batch.to_pylist():
                    ins = (rec.get("instruction") or "").strip()
                    inp = (rec.get("input") or "").strip()
                    resp = (rec.get("output") or "").strip()
                    if not ins or not resp:
                        continue
                    parts = [ins]
                    if inp and inp.lower() != "nan":
                        parts.append(inp)
                    parts.append("Адказ: " + resp)
                    w.write({
                        "id": f"{SOURCE}:{rec.get('id', n)}",
                        "source": SOURCE,
                        "question": "\n".join(parts[:-1]),
                        "answer": resp,
                        "text": "\n".join(parts),
                        "url": None,
                        "title": None,
                    })
                    n += 1


PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "data" / "raw" / "wikipedia_be_old.jsonl.gz"
PARQUET_NAME = "20231101.be-x-old/train-00000-of-00001.parquet"

def cmd_wiki_old() -> int:
    import gzip
    import json
    import os
    import warnings
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    warnings.filterwarnings("ignore")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 1000:
        return 0
    p = hf_hub_download(
        repo_type="dataset",
        repo_id="wikimedia/wikipedia",
        filename=PARQUET_NAME,
        cache_dir=str(PROJECT / "data" / "hf_cache"),
    )
    tbl = pq.read_table(p, columns=["id", "text", "title", "url"])
    n = len(tbl)
    out = OUT.with_suffix(".tmp.gz")
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as f:
        total_chars = 0
        for i in range(n):
            text = tbl["text"][i].as_py()
            if text is None or len(text) < 200:
                continue
            total_chars += len(text)
            f.write(json.dumps({
                "id": f"be-old-{tbl['id'][i].as_py()}",
                "text": text,
                "source": "wikipedia_be_old",
                "lang": "be-x-old",
                "url": tbl["url"][i].as_py() if tbl["url"][i].as_py() else None,
                "title": tbl["title"][i].as_py() if tbl["title"][i].as_py() else None,
            }, ensure_ascii=False) + "\n")
    os.replace(out, OUT)
    return 0


def _parquet_files(manifest_path: Path) -> list[str]:
    m = json.loads(manifest_path.read_text().strip() or "[]")
    return [e["path"] for e in m if e.get("file", "").endswith(".parquet")]


def _jsonl_gz(path: Path, files: list[str], text_get, id_get, source: str):
    import pyarrow.parquet as pq

    out = path
    n = 0
    t0 = time.time()
    with JsonlWriter(out) as w:
        for fi, f in enumerate(files):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=8192):
                for rec in batch.to_pylist():
                    text = text_get(rec)
                    if not text or len(text.strip()) < 1:
                        continue
                    w.write({
                        "id": f"{source}:{id_get(rec, n)}",
                        "source": source,
                        "text": text,
                        "url": rec.get("url") or None,
                        "title": None,
                    })
                    n += 1
            if (fi + 1) % 10 == 0 or fi == len(files) - 1:
                el = time.time() - t0
    return n


def conv_openorca(dest: Path) -> int:
    m = Path("data/raw/newsrc/openorca_be/manifest.json")
    files = _parquet_files(m)
    def text_get(rec):
        q = (rec.get("question") or "").strip()
        a = (rec.get("response") or "").strip()
        sp = (rec.get("system_prompt") or "").strip()
        if not q or not a: return None
        parts = []
        if sp: parts.append(sp)
        parts.append(q)
        parts.append("Адказ: " + a)
        return "\n".join(parts)
    def id_get(rec, i): return rec.get("id", i)
    return _jsonl_gz(dest, files, text_get, id_get, "openorca_be")


def conv_ccmatrix(dest: Path) -> int:
    m = Path("data/raw/newsrc/ccmatrix_enbe/manifest.json")
    files = _parquet_files(m)
    def text_get(rec):
        be = (rec.get("non_english") or "").strip()
        return be if be else None
    def id_get(rec, i): return i
    return _jsonl_gz(dest, files, text_get, id_get, "ccmatrix_enbe")


def conv_c4(dest: Path) -> int:
    import glob, os
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--allenai--c4/snapshots/*/multilingual/c4-be.tfrecord-*.gz"))
    snap = [s for s in snap if not s.endswith(".lock")]
    snap.sort()
    n, t0 = 0, time.time()
    with JsonlWriter(dest) as w:
        for fi, f in enumerate(snap):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for ln, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (d.get("text") or "").strip()
                    if not text:
                        continue
                    w.write({
                        "id": f"c4_be:{fi}-{ln}",
                        "source": "c4_be",
                        "text": text,
                        "url": d.get("url") or None,
                        "title": None,
                    })
                    n += 1
    return n


def cmd_convert() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    jobs = {
        "openorca_be":  (data_raw() / "openorca_be.jsonl.gz", conv_openorca),
        "ccmatrix_enbe":(data_raw() / "ccmatrix_enbe.jsonl.gz", conv_ccmatrix),
        "c4_be":        (data_raw() / "c4_be.jsonl.gz", conv_c4),
    }
    for name, (dest, fn) in jobs.items():
        if only and name not in only:
            continue
        if dest.exists() and not args.force:
            continue
        n = fn(dest)


def data_raw() -> Path:
    return Path("data/raw")


_BOILERPLATE_RE = re.compile(
    r"(cookie|javascript|©|усе правы абаронены|правы абаронены|все права защищены|падпісвайцеся|"
    r"падпішыцеся|чытайце таксама|чытаць далей|читайте также|поделиться|падзяліцеся|\bshare\b|"
    r"telegram|viber|facebook|instagram|twitter|youtube|vk\.com|загрузка\.\.\.|рэклама|реклама|"
    r"пошук па сайце|галоўная старонка|перайсці да)",
    re.I,
)
_TERMINAL_PUNCT = tuple(".!?…:;»\"')")
_REFERENCE_HEADINGS = {
    "зноскі", "літаратура", "спасылкі", "крыніцы", "гл. таксама", "глядзіце таксама",
    "заўвагі", "вонкавыя спасылкі", "бібліяграфія", "прымечанні", "знешнія спасылкі",
}

_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _CFG
    _CFG = cfg


def _drop_reference_tail(lines: list[str]) -> list[str]:
    cut = int(len(lines) * 0.6)
    for i in range(cut, len(lines)):
        if lines[i].strip().lower().rstrip(":") in _REFERENCE_HEADINGS:
            return lines[:i]
    return lines


def _keep_line(line: str, strict: bool, reasons: Counter) -> bool:
    if looks_broken(line):
        reasons["line_broken"] += 1
        return False
    n = len(line)
    letters = sum(1 for ch in line if ch.isalpha())
    if strict:
        if n < 200 and _BOILERPLATE_RE.search(line):
            reasons["line_boilerplate"] += 1
            return False
        if line.count("|") >= 3 or line.count("•") >= 3:
            reasons["line_menu"] += 1
            return False
        if len(line.split()) < 3 and not line.endswith(_TERMINAL_PUNCT):
            reasons["line_short"] += 1
            return False
        if letters < 0.5 * n:
            reasons["line_nonalpha"] += 1
            return False
    elif n > 20 and letters < 0.3 * n:
        reasons["line_nonalpha"] += 1
        return False
    return True


def clean_document(rec: dict, cfg: dict, reasons: Counter) -> dict | None:
    text = rec.get("text") or ""
    if looks_like_html(text):
        text = strip_html(text)
    if looks_like_wikitext(text):
        text = strip_wikitext(text)
    text = remove_urls_emails(text)
    text = normalize_text(text)
    if not text:
        reasons["empty"] += 1
        return None

    lines = text.split("\n")
    if cfg.get("drop_reference_sections"):
        lines = _drop_reference_tail(lines)

    lang_chars = Counter()
    kept = []
    for line in lines:
        lang = classify_text(line)
        lang_chars[lang] += len(line)
        if lang in ("ru", "uk", "other"):
            reasons[f"line_lang_{lang}"] += 1
            continue
        if _keep_line(line, cfg["strict_lines"], reasons):
            kept.append(line)

    denom = lang_chars["be"] + lang_chars["ru"] + lang_chars["uk"] + lang_chars["other"]
    bel_share = lang_chars["be"] / denom if denom else 0.0
    if bel_share < cfg["min_bel_share"]:
        reasons["doc_low_bel_share"] += 1
        return None
    if not kept:
        reasons["doc_empty_after_lines"] += 1
        return None

    text = "\n".join(kept)
    if len(text) < cfg["min_chars"]:
        reasons["doc_short"] += 1
        return None
    toks = text.split()
    wds = WORD_RE.findall(text)
    if len(wds) < cfg["min_words"]:
        reasons["doc_few_words"] += 1
        return None
    mean_wl = sum(len(w) for w in wds) / len(wds)
    if not (cfg["min_mean_word_len"] <= mean_wl <= cfg["max_mean_word_len"]):
        reasons["doc_word_len"] += 1
        return None
    alpha_share = sum(1 for t in toks if any(ch.isalpha() for ch in t)) / len(toks)
    if alpha_share < cfg["min_alpha_word_share"]:
        reasons["doc_alpha_share"] += 1
        return None
    symbols = text.count("#") + text.count("…") + text.count("...")
    if symbols / len(toks) > cfg["max_symbol_word_ratio"]:
        reasons["doc_symbols"] += 1
        return None
    if len(kept) >= 5:
        dup_share = 1.0 - len(set(kept)) / len(kept)
        if dup_share > cfg["max_dup_line_share"]:
            reasons["doc_dup_lines"] += 1
            return None
    if cfg["strict_lines"] and len(kept) >= 3:
        punct = sum(1 for ln in kept if ln.endswith(_TERMINAL_PUNCT)) / len(kept)
        if punct < cfg.get("min_terminal_punct_share", 0.0):
            reasons["doc_no_punct"] += 1
            return None
    tarask = taraskevica_score(text)
    if cfg["drop_taraskevica"] and tarask > cfg["taraskevica_threshold"]:
        reasons["doc_taraskevica"] += 1
        return None

    return {
        "id": rec["id"],
        "source": rec.get("source"),
        "text": text,
        "url": rec.get("url"),
        "title": rec.get("title"),
        "bel_share": round(bel_share, 4),
        "tarask": round(tarask, 3),
    }


def _process_batch(batch: list[dict]):
    reasons: Counter = Counter()
    out = []
    chars_in = 0
    for rec in batch:
        chars_in += len(rec.get("text") or "")
        doc = clean_document(rec, _CFG, reasons)
        if doc is not None:
            out.append(doc)
    return out, reasons, chars_in


def clean_file(in_path: Path, out_path: Path, cfg: dict, workers: int, max_clean_mb: float | None) -> dict:
    t0 = time.time()
    reasons: Counter = Counter()
    docs_in = docs_out = chars_in = chars_out = bytes_out = tarask_docs = 0
    limit = max_clean_mb * 1e6 if max_clean_mb else float("inf")
    with JsonlWriter(out_path) as w, Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for docs, r, ci in bounded_imap(pool, _process_batch, batched(read_jsonl(in_path), 256), 4 * workers):
            reasons.update(r)
            chars_in += ci
            docs_in += 256
            for d in docs:
                if bytes_out >= limit:
                    break
                w.write(d)
                docs_out += 1
                chars_out += len(d["text"])
                bytes_out += len(d["text"].encode("utf-8"))
                tarask_docs += d["tarask"] > cfg["taraskevica_threshold"]
            if bytes_out >= limit:
                pool.terminate()
                break
    stats = {
        "input": str(in_path),
        "output": str(out_path),
        "docs_out": docs_out,
        "chars_in": chars_in,
        "chars_out": chars_out,
        "mb_out": round(bytes_out / 1e6, 2),
        "kept_char_share": round(chars_out / chars_in, 4) if chars_in else 0.0,
        "taraskevica_docs": tarask_docs,
        "reasons": dict(reasons.most_common()),
        "seconds": round(time.time() - t0, 1),
    }
    write_json(out_path.with_name(out_path.name.replace(".jsonl.gz", ".stats.json")), stats)
    return stats


def cmd_clean() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir, clean_dir = Path(cfg["raw_dir"]), Path(cfg["clean_dir"])
    clean_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for src in cfg["sources"]:
        if args.only:
            if src["name"] not in args.only:
                continue
        elif not src.get("enabled", True):
            continue
        prof = dict(cfg["common"])
        prof.update(cfg["profiles"][src.get("profile", "web")])
        if src.get("kind") == "local":
            inputs = [Path(p) for p in sorted(glob.glob(src["files"]))]
        else:
            inputs = [raw_dir / f"{src['name']}.jsonl.gz"]
        for inp in inputs:
            if not inp.exists():
                continue
            out = clean_dir / inp.name
            if out.exists() and not args.overwrite:
                continue
            all_stats[inp.name] = clean_file(inp, out, prof, args.workers, src.get("max_clean_mb"))
    if all_stats:
        write_json(clean_dir / "_summary.stats.json", all_stats)


_NORM_RE = re.compile(r"[\W_]+")


_MERSENNE = np.uint64((1 << 61) - 1)
_U64_MAX = np.iinfo(np.uint64).max


def make_permutations(num_perm: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
    return a, b


def shingle_hashes(text: str, n: int = 5) -> np.ndarray:
    ws = [w.lower() for w in WORD_RE.findall(text)]
    if len(ws) < n:
        grams = {" ".join(ws)}
    else:
        grams = {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}
    return np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))


def minhash_signature(text: str, a: np.ndarray, b: np.ndarray, n: int = 5, chunk: int = 4096) -> np.ndarray:
    h = shingle_hashes(text, n)
    sig = np.full(a.shape[0], _U64_MAX, dtype=np.uint64)
    for s in range(0, len(h), chunk):
        x = h[s:s + chunk, None]
        # a < 2^31, x < 2^32 => a*x + b < 2^64, переполнения нет
        v = (x * a[None, :] + b[None, :]) % _MERSENNE
        np.minimum(sig, v.min(axis=0), out=sig)
    return (sig & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def band_keys(sig_block: np.ndarray) -> np.ndarray:
    """Хеш строк блока сигнатур (N, r) -> (N,) uint64."""
    key = np.full(sig_block.shape[0], np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for j in range(sig_block.shape[1]):
        key = (key ^ sig_block[:, j].astype(np.uint64)) * prime
    return key


class UnionFind:
    """Корень кластера — минимальный индекс (документ с наивысшим приоритетом)."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return int(root)

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            if rx < ry:
                self.parent[ry] = rx
            else:
                self.parent[rx] = ry

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


def lsh_clusters(sig: np.ndarray, bands: int, threshold: float, verify_chunk: int = 200_000) -> UnionFind:
    n, num_perm = sig.shape
    rows = num_perm // bands
    uf = UnionFind(n)
    for b in range(bands):
        block = np.asarray(sig[:, b * rows:(b + 1) * rows])
        keys = band_keys(block)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        dup = np.nonzero(sk[1:] == sk[:-1])[0]
        for s in range(0, len(dup), verify_chunk):
            d = dup[s:s + verify_chunk]
            left, right = order[d], order[d + 1]
            sims = (np.asarray(sig[left]) == np.asarray(sig[right])).mean(axis=1)
            for x, y in zip(left[sims >= threshold], right[sims >= threshold]):
                uf.union(int(x), int(y))
    return uf


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def exact_key(text: str) -> int:
    return _h64(_NORM_RE.sub("", text.lower()))


def line_key(line: str) -> int:
    return _h64(line.strip().lower())


class CountMinSketch:
    def __init__(self, log2_width: int = 24, depth: int = 4, seed: int = 0):
        self.width_mask = np.uint64((1 << log2_width) - 1)
        self.table = np.zeros((depth, 1 << log2_width), dtype=np.uint32)
        rng = np.random.default_rng(seed)
        self.salts = rng.integers(1, np.iinfo(np.int64).max, size=depth, dtype=np.int64).astype(np.uint64)

    def _index(self, h: np.ndarray, d: int) -> np.ndarray:
        x = h ^ self.salts[d]
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        return (x & self.width_mask).astype(np.int64)

    def add(self, h: np.ndarray) -> None:
        for d in range(self.table.shape[0]):
            np.add.at(self.table[d], self._index(h, d), 1)

    def query(self, h: np.ndarray) -> np.ndarray:
        return np.min(np.stack([self.table[d][self._index(h, d)] for d in range(self.table.shape[0])]), axis=0)


_PERM = None


def _init_sig_worker(num_perm: int, seed: int, ngram: int) -> None:
    global _PERM
    _PERM = (*make_permutations(num_perm, seed), ngram)


def _signatures(recs: list[dict]) -> np.ndarray:
    a, b, n = _PERM
    return np.stack([minhash_signature(r["text"], a, b, n) for r in recs])


def ordered_inputs(cfg: dict) -> list[tuple[int, Path]]:
    clean_dir = Path(cfg["clean_dir"])
    items = []
    for src in cfg["sources"]:
        if not src.get("enabled", True):
            continue
        if src.get("kind") == "local":
            paths = [clean_dir / Path(p).name for p in sorted(glob.glob(src["files"]))]
        else:
            paths = [clean_dir / f"{src['name']}.jsonl.gz"]
        for p in paths:
            if p.exists():
                items.append((int(src.get("priority", 100)), p))
    return sorted(items, key=lambda x: x[0])


def cmd_dedup() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--out", default="data/dedup/corpus.jsonl.gz")
    ap.add_argument("--work-dir", default="data/dedup/work")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--boiler-min-docs", type=int, default=30)
    ap.add_argument("--boiler-max-len", type=int, default=500)
    ap.add_argument("--boiler-min-len", type=int, default=15)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    assert args.num_perm % args.bands == 0

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stats: dict = defaultdict(Counter)
    t0 = time.time()
    stage1 = work / "stage1.jsonl.gz"
    seen: set[int] = set()
    with JsonlWriter(stage1) as w:
        for prio, path in ordered_inputs(cfg):
            for rec in read_jsonl(path):
                src = rec.get("source") or path.stem
                stats[src]["docs_in"] += 1
                stats[src]["mb_in"] += len(rec["text"].encode("utf-8")) / 1e6
                k = exact_key(rec["text"])
                if k in seen:
                    stats[src]["exact_dups"] += 1
                    continue
                seen.add(k)
                w.write(rec)
    del seen

    cms = CountMinSketch(seed=args.seed)
    for batch in batched(read_jsonl(stage1), 2000):
        keys = []
        for rec in batch:
            uniq = {line_key(ln) for ln in rec["text"].split("\n")
                    if args.boiler_min_len <= len(ln) <= args.boiler_max_len}
            keys.extend(uniq)
        if keys:
            cms.add(np.array(keys, dtype=np.uint64))

    stage2 = work / "stage2.jsonl.gz"
    sig_path = work / "signatures.u32"
    n_docs = 0
    with JsonlWriter(stage2) as w, open(sig_path, "wb") as sig_f, \
            Pool(args.workers, initializer=_init_sig_worker, initargs=(args.num_perm, args.seed, args.ngram)) as pool:

        def cleaned_batches():
            for batch in batched(read_jsonl(stage1), 1000):
                out = []
                for rec in batch:
                    lines = rec["text"].split("\n")
                    cand = [i for i, ln in enumerate(lines) if args.boiler_min_len <= len(ln) <= args.boiler_max_len]
                    drop = set()
                    if cand:
                        counts = cms.query(np.array([line_key(lines[i]) for i in cand], dtype=np.uint64))
                        drop = {i for i, c in zip(cand, counts) if c >= args.boiler_min_docs}
                    src = rec.get("source")
                    if drop:
                        stats[src]["boiler_lines_removed"] += len(drop)
                        lines = [ln for i, ln in enumerate(lines) if i not in drop]
                        rec["text"] = "\n".join(lines)
                    if len(rec["text"]) < args.min_chars:
                        stats[src]["short_after_boiler"] += 1
                        continue
                    out.append(rec)
                if out:
                    yield out

        for recs, sigs in bounded_imap(pool, _signatures, cleaned_batches(), 4 * args.workers, keep_input=True):
            for rec in recs:
                w.write(rec)
            sig_f.write(np.ascontiguousarray(sigs, dtype=np.uint32).tobytes())
            n_docs += len(recs)

    if n_docs == 0:
        raise SystemExit("После очистки и удаления шаблонных строк не осталось документов — проверьте data/clean")
    sig = np.memmap(sig_path, dtype=np.uint32, mode="r", shape=(n_docs, args.num_perm))
    uf = lsh_clusters(sig, args.bands, args.threshold)
    roots = uf.roots()
    keep = roots == np.arange(n_docs)

    out_path = Path(args.out)
    total_bytes = 0
    with JsonlWriter(out_path) as w:
        for i, rec in enumerate(read_jsonl(stage2)):
            src = rec.get("source")
            if not keep[i]:
                stats[src]["near_dups"] += 1
                continue
            w.write(rec)
            b = len(rec["text"].encode("utf-8"))
            stats[src]["docs_out"] += 1
            stats[src]["mb_out"] += b / 1e6
            total_bytes += b

    summary = {
        "docs_out": int(keep.sum()),
        "mb_out": round(total_bytes / 1e6, 1),
        "seconds": round(time.time() - t0, 1),
        "params": vars(args),
        "per_source": {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                       for k, v in stats.items()},
    }
    write_json(out_path.with_name("dedup.stats.json"), summary)


SPLITS = ("train", "validation", "test")


def split_of(doc_id: str, seed: int, val_frac: float, test_frac: float, salt: int = 0) -> str:
    h = hashlib.blake2b(f"{seed}:{salt}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, "little") / 2 ** 64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "validation"
    return "train"


def train_fraction(source: str, name_to_frac: dict[str, float]) -> float:
    """Доля документов источника, попадающих в train (1.0 — вся, >1 — оверсэмплирование)."""
    return float(name_to_frac.get(source, 1.0))


def train_keep(doc_id: str, source: str, seed: int, frac: float) -> bool:
    if frac >= 1.0:
        return True
    h = hashlib.blake2b(f"{seed + 7919}:{source}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2 ** 64 < frac


def cmd_split() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/dedup/corpus.jsonl.gz")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--test-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--sources", default="configs/sources.yaml")
    args = ap.parse_args()

    name_to_frac: dict[str, float] = {}
    if args.sources and Path(args.sources).exists():
        with open(args.sources, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for src in cfg.get("sources", []):
            frac = src.get("train_sample_frac", 1.0)
            if frac is not None:
                name_to_frac[src["name"]] = float(frac)

    out = Path(args.out_dir)
    writers = {s: JsonlWriter(out / f"{s}.jsonl.gz") for s in SPLITS}
    stats = {s: Counter() for s in SPLITS}
    per_source = defaultdict(lambda: {s: Counter() for s in SPLITS})
    oversample = 0
    for rec in read_jsonl(args.input):
        s = split_of(rec["id"], args.seed, args.val_frac, args.test_frac)
        text = rec["text"]
        b = len(text.encode("utf-8"))
        if s == "train":
            frac = train_fraction(rec.get("source"), name_to_frac)
            if not train_keep(rec["id"], rec.get("source") or "none", args.seed, frac):
                continue
            # Оверсэмплирование >1: дописываем полные копии, каждый со своим солью,
            # чтобы дедупликация/контроль загрязнения (по id) их не спутали.
            for copy in range(max(1, int(frac))):
                rid = rec["id"] if copy == 0 else f"{rec['id']}#os{copy}"
                rec_out = {"id": rid, "source": rec.get("source"), "text": text}
                writers["train"].write(rec_out)
                if copy > 0:
                    oversample += 1
                    stats["train"]["oversampled_docs"] += 1
                    per_source[rec.get("source")]["train"]["oversampled_docs"] += 1
                for c in (stats["train"], per_source[rec.get("source")]["train"]):
                    c["docs"] += 1
                    c["chars"] += len(text)
                    c["bytes"] += b
                stats["train"]["words"] += len(words(text))
        else:
            rec_out = {"id": rec["id"], "source": rec.get("source"), "text": text}
            writers[s].write(rec_out)
            for c in (stats[s], per_source[rec.get("source")][s]):
                c["docs"] += 1
                c["chars"] += len(text)
                c["bytes"] += b
            stats[s]["words"] += len(words(text))
    for w in writers.values():
        w.close()

    summary = {
        "params": vars(args),
        "splits": {s: {**dict(c), "mb": round(c["bytes"] / 1e6, 2)} for s, c in stats.items()},
        "per_source": {src: {s: dict(c) for s, c in d.items()} for src, d in per_source.items()},
    }
    write_json(out / "splits.stats.json", summary)


_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", normalize_text(s).lower()).strip()


def cmd_check() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="eval_prompts.jsonl")
    ap.add_argument("--train", default="data/splits/train.jsonl.gz")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--report", default="reports/contamination.json")
    args = ap.parse_args()

    prompts = [p for p in read_jsonl(args.prompts)]
    keys = {p["id"]: norm(p["prompt"]) for p in prompts}
    strict = {pid for pid, k in keys.items() if len(k.split()) >= args.min_words}
    hits = {pid: 0 for pid in keys}

    train = Path(args.train)
    tmp_clean = train.with_name(train.name + ".filtered.tmp.jsonl.gz")
    writer = JsonlWriter(tmp_clean) if args.remove else None
    contaminated = JsonlWriter(train.with_name("train.contaminated.jsonl.gz")) if args.remove else None
    n_docs = n_removed = 0
    for rec in read_jsonl(train):
        n_docs += 1
        text = _WS.sub(" ", rec["text"].lower())
        bad = False
        for pid, k in keys.items():
            if k in text:
                hits[pid] += 1
                bad = bad or pid in strict
        if writer is not None:
            if bad:
                contaminated.write(rec)
                n_removed += 1
            else:
                writer.write(rec)
    if writer is not None:
        writer.close()
        contaminated.close()
        os.replace(writer.path, train)

    report = {
        "train_docs": n_docs,
        "removed_docs": n_removed,
        "strict_prompt_ids": sorted(strict),
        "strict_hits": {pid: hits[pid] for pid in sorted(strict) if hits[pid]},
        "substring_hits_short_prompts": {pid: hits[pid] for pid in keys if pid not in strict and hits[pid]},
    }
    write_json(args.report, report)


_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_PHRASES = Path(__file__).resolve().parent / "configs" / "phrases.json"
_CALIB_KIND = {"fact": "qa", "refuse": "refuse", "clarify": "unclear", "stop": "refuse"}
MID_EXAMPLES_PER_STEP = 256


def _read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _add(rows, question, answer, source, kind):
    q, a = question.strip(), answer.strip()
    if len(q) < 3 or not a:
        return
    rows.append({
        "id": f"chat:{source}:{kind}:{len(rows)}",
        "source": source,
        "kind": kind,
        "question": q,
        "answer": a,
        "text": f"{q}\nАдказ: {a}",
    })


def load_qa_file(path: Path, source: str) -> list:
    out = []
    for rec in _read_jsonl(path):
        if rec.get("question") and rec.get("answer"):
            pair = (rec["question"], rec["answer"])
        else:
            pair = parse_qa_text(rec.get("text") or "")
        if pair:
            _add(out, pair[0], pair[1], source, "qa")
    return out


def load_calib(path: Path) -> list:
    out = []
    for rec in _read_jsonl(path):
        q = (rec.get("prompt") or rec.get("question") or "").strip()
        a = (rec.get("answer") or rec.get("response") or "").strip()
        if q and a:
            _add(out, q, a, "calib", _CALIB_KIND.get(rec.get("type") or "", "qa"))
    return out


def load_phrases() -> dict:
    return json.loads(_PHRASES.read_text(encoding="utf-8"))


def build_refuse(phr: dict) -> list:
    rows = []
    facts, unclear = phr["refuse"], phr["unclear_answer"]
    for i, q in enumerate(phr["unknown"]):
        _add(rows, q, facts[i % len(facts)], "refuse", "refuse")
        _add(rows, "Скажы, " + q[0].lower() + q[1:], facts[(i + 3) % len(facts)], "refuse", "refuse")
    for i, q in enumerate(phr["unclear"]):
        _add(rows, q, unclear[i % len(unclear)], "refuse", "unclear")
    return rows


def web_to_chat(text: str, rng: random.Random, max_chars: int = 900):
    t = " ".join((text or "").split())
    if len(t) < 220:
        return None
    t = t[:max_chars].rsplit(" ", 1)[0]
    if rng.random() < 0.5:
        parts = _SENT_RE.split(t)
        if len(parts) >= 2 and 40 <= len(parts[0]) <= 220:
            return f"Працягні тэкст:\n{parts[0]}", " ".join(parts[1:]).strip()
    return "Напішы кароткі тэкст па-беларуску.", t


def load_web(paths, cap: int, seed: int) -> list:
    rng = random.Random(seed)
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for rec in _read_jsonl(path):
            pair = web_to_chat(rec.get("text") or "", rng)
            if pair is None:
                continue
            _add(rows, pair[0], pair[1], rec.get("source") or path.stem, "continue")
            if len(rows) >= cap:
                return rows
    return rows


def _first(*cands: str):
    for c in cands:
        p = Path(c)
        if p.exists():
            return p
    return None


def cmd_build() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/splits/chat")
    ap.add_argument("--alpaca", default="data/clean/alpaca_be.jsonl.gz")
    ap.add_argument("--distill", default="data/raw/distill_be.jsonl.gz")
    ap.add_argument("--web", nargs="*", default=[
        "data/clean/fineweb2_be.jsonl.gz",
        "data/clean/madlad400_be_clean.jsonl.gz",
    ])
    ap.add_argument("--web-cap", type=int, default=12000)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for name, path in (
        ("alpaca_be", _first(args.alpaca, "data/raw/alpaca_be.jsonl.gz")),
        ("distill_be", _first(args.distill, "data/clean/distill_be.jsonl.gz")),
    ):
        if path is None:
            continue
        part = load_qa_file(path, name)
        rows.extend(part)

    calib = _first("data/raw/calib_qa.jsonl", "calib_qa.jsonl")
    if calib is not None:
        cal = load_calib(calib)
        rows.extend(cal)

    if not any(r["source"] in {"alpaca_be", "distill_be", "calib"} for r in rows):
        raise SystemExit("нет alpaca, distill и calib")

    refuse = build_refuse(load_phrases())
    rows.extend(refuse)
    web = load_web([Path(p) for p in args.web], args.web_cap, args.seed)
    rows.extend(web)

    seen, uniq = set(), []
    for r in rows:
        key = " ".join(r["question"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val, trn = uniq[:n_val], uniq[n_val:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(out / "train.jsonl.gz") as w:
        for r in trn:
            w.write(r)
    with JsonlWriter(out / "validation.jsonl.gz") as w:
        for r in val:
            w.write(r)
    kinds = collections.Counter(r["kind"] for r in uniq)



def main() -> None:
    cmds = {
        "hf": cmd_hf,
        "alpaca": cmd_alpaca,
        "wiki-old": cmd_wiki_old,
        "convert": cmd_convert,
        "clean": cmd_clean,
        "dedup": cmd_dedup,
        "split": cmd_split,
        "check": cmd_check,
        "chat": cmd_build,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(2)
    fn = cmds[sys.argv[1]]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    fn()


if __name__ == "__main__":
    main()
