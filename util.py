"""Текст, jsonl, токены, конфиг, метрики."""
from __future__ import annotations

import sys

import yaml

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import gzip
import json
import os
from pathlib import Path
from typing import Iterable, Iterator
import copy
import html
import math
import re
import unicodedata
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import torch

def open_text(path: str | Path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", compresslevel=3) if "w" in mode else gzip.open(
            path, mode, encoding="utf-8"
        )
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open_text(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(".tmp-" + self.path.name)
        self._f = open_text(self.tmp, "wt")
        self.count = 0

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> None:
        self._f.close()
        os.replace(self.tmp, self.path)

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._f.close()


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    with JsonlWriter(path) as w:
        for r in records:
            w.write(r)
        return w.count


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bounded_imap(pool, func, iterable: Iterable, max_inflight: int, keep_input: bool = False) -> Iterator:
    from collections import deque

    q: deque = deque()
    for item in iterable:
        q.append((item, pool.apply_async(func, (item,))))
        if len(q) >= max_inflight:
            inp, res = q.popleft()
            yield (inp, res.get()) if keep_input else res.get()
    while q:
        inp, res = q.popleft()
        yield (inp, res.get()) if keep_input else res.get()


def batched(it: Iterable, n: int) -> Iterator[list]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch




def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("defaults", None)
    if base:
        bases = base if isinstance(base, list) else [base]
        merged: dict = {}
        for b in bases:
            merged = deep_merge(merged, load_config(path.parent / b))
        data = deep_merge(merged, data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Переопределение должно иметь вид key.sub=value, получено: {item}")
        key, value = item.split("=", 1)
        set_by_path(data, key.strip(), yaml.safe_load(value))
    return data


def set_by_path(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


_MULTISPACE_RE = re.compile(r" {2,}")
_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{2,6}|#x[0-9a-fA-F]{2,5});")


def _plain(ch: str) -> str | None:
    if ch == "\n" or ch == "\t":
        return " " if ch == "\t" else ch
    if ch in "\v\f":
        return " "
    kind = unicodedata.category(ch)
    if kind == "Zs":
        return " "
    if kind in ("Cf", "Mn"):
        return None
    if ch in "`´’‘ʼ′ʹˈ＇":
        return "'"
    if kind == "Pd" and ch != "-":
        return "—"
    return ch


_LAT2CYR = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і",
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р",
    "C": "С", "T": "Т", "X": "Х", "I": "І", "ŭ": "ў", "Ŭ": "Ў",
})


def _is_cyr(ch: str) -> bool:
    return "а" <= ch <= "я" or "А" <= ch <= "Я" or ch in "ёЁіІўЎїЇєЄґҐ"


def _is_lat(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ch in "ŭŬ"


def _fix_word(word: str) -> str:
    if not any(_is_cyr(ch) for ch in word):
        return word
    if not any(_is_lat(ch) for ch in word):
        return word
    fixed = word.translate(_LAT2CYR)
    if any(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in fixed):
        return word
    return fixed


def fix_homoglyphs(text: str) -> str:
    parts = re.split(r"(\s+)", text)
    for i, part in enumerate(parts):
        if not part or part.isspace():
            continue
        if part in ("i", "I"):
            prev = next((p for p in reversed(parts[:i]) if p.strip()), "")
            nxt = next((p for p in parts[i + 1:] if p.strip()), "")
            if any(_is_cyr(ch) for ch in prev) and any(_is_cyr(ch) for ch in nxt):
                parts[i] = "і" if part == "i" else "І"
                continue
        parts[i] = _fix_word(part)
    return "".join(parts)


def normalize_text(text: str) -> str:
    if "&" in text and _ENTITY_RE.search(text):
        text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for ch in text if (c := _plain(ch)) is not None)
    text = "".join(ch for ch in text if ch in "\n\t" or not (ord(ch) < 32 or 127 <= ord(ch) <= 159))
    text = fix_homoglyphs(text)
    lines = (_MULTISPACE_RE.sub(" ", ln).strip() for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if ln)


WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[«\"„(\-—]?[A-ZА-ЯЁІЎ0-9])")


def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def split_sentences(text: str) -> Iterator[str]:
    for para in text.split("\n"):
        for s in _SENT_SPLIT_RE.split(para):
            s = s.strip()
            if s:
                yield s


RU_WORDS = frozenset(
    "что это как его она оно только если который которая которое очень также сейчас всегда "
    "ещё еще где когда тогда потому может нет был будет чтобы себя своей этого этой этот "
    "чем уже город время человек после через того более всё все про от со во но по до".split()
)
BE_WORDS = frozenset(
    "гэта што як яго яна яно але калі таксама вельмі толькі дзе ёсць быў ужо які якая якое "
    "якія яшчэ пасля праз таму можна трэба нават зусім амаль ад па пра усё ўсё усе ўсе сваю "
    "свае сябе гэтага гэтай гэты цяпер заўсёды тады чалавек горад час паміж каля".split()
)
UK_ONLY = frozenset("їєґ")
RU_ONLY = frozenset("ищъ")
FOREIGN_CYR = RU_ONLY | UK_ONLY

_NON_LETTER_RE = re.compile(r"[\W\d_]+")


@dataclass
class LangCounts:
    letters: int
    cyr: int
    ru: int
    uk: int
    bel: int
    bel_only: int
    i_count: int


def lang_counts(text: str) -> LangCounts:
    low = text.lower()
    letters = len(_NON_LETTER_RE.sub("", low))
    cyr = "".join(ch for ch in low if _is_cyr(ch))
    return LangCounts(
        letters=letters,
        cyr=len(cyr),
        ru=cyr.count("и") + cyr.count("щ") + cyr.count("ъ"),
        uk=cyr.count("ї") + cyr.count("є") + cyr.count("ґ"),
        bel=cyr.count("ў") + cyr.count("і"),
        bel_only=cyr.count("ў") + cyr.count("ы") + cyr.count("э") + cyr.count("ё"),
        i_count=cyr.count("і"),
    )


def classify_text(text: str, min_letters: int = 12, foreign_tol: float = 0.01) -> str:
    c = lang_counts(text)
    if c.letters < min_letters:
        return "unk"
    if c.cyr < 0.5 * c.letters:
        return "other"
    foreign = c.ru + c.uk
    if foreign >= 2 and foreign > foreign_tol * c.cyr:
        if c.uk > 0 or (c.i_count > 0 and c.bel_only == 0):
            return "uk"
        return "ru"
    toks = WORD_RE.findall(text.lower())
    ru_hits = sum(1 for w in toks if w in RU_WORDS)
    be_hits = sum(1 for w in toks if w in BE_WORDS)
    if c.bel == 0 and ru_hits > be_hits:
        return "ru"
    if ru_hits > be_hits + 2:
        return "ru"
    return "be"


def classify_word(word: str) -> str:
    w = word.lower()
    if any(ch in UK_ONLY for ch in w):
        return "uk"
    if any(ch in RU_ONLY for ch in w) or w in RU_WORDS:
        return "ru"
    if "ў" in w or "і" in w or w in BE_WORDS:
        return "be"
    if any(_is_lat(ch) for ch in w):
        return "latin"
    return "neutral"

_TARASK_RE = re.compile(r"(?:с|з|ц|дз)ь(?:в|н|л|ц|с|з)")


def taraskevica_score(text: str) -> float:
    low = text.lower()
    cyr = sum(1 for ch in low if _is_cyr(ch))
    if cyr == 0:
        return 0.0
    return 1000.0 * len(_TARASK_RE.findall(low)) / cyr


def looks_broken(line: str) -> bool:
    if "�" in line:
        return True
    # UTF-8, прочитанный как latin-1 или cp1251: Ð/Ñ или Р/С и следом обломок
    junk = "‚„…†‡€‰‹›‘’“”•–—™°ЂЃЉЊЌЋЏђѓљњћџ"
    bad = 0
    for a, b in zip(line, line[1:]):
        if a in "ÐÑ" and 128 <= ord(b) <= 191:
            bad += 1
        elif a in "РС" and b in junk:
            bad += 1
        if bad >= 3:
            return True
    return False


_DROP_BLOCKS_RE = re.compile(
    r"<(script|style|noscript|iframe|svg|nav|footer|header|form|aside)\b[^>]*>.*?</\1\s*>", re.S | re.I
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_BLOCK_TAG_RE = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article|blockquote|pre)\b[^>]*>", re.I)
_TAG_RE = re.compile(r"<[^>]{1,500}>")
_HTML_HINT_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?>")

_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_WIKI_LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_WIKI_EXT_LINK_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_WIKI_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.M)
_WIKI_FORMAT_RE = re.compile(r"'{2,5}")
_WIKI_TABLE_RE = re.compile(r"^\s*(\{\||\|\}|\|-|!|\|).*$", re.M)

URL_RE = re.compile(r"(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def looks_like_html(text: str) -> bool:
    return len(_HTML_HINT_RE.findall(text[:5000])) >= 3


def strip_html(text: str) -> str:
    text = _COMMENT_RE.sub(" ", text)
    text = _DROP_BLOCKS_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def looks_like_wikitext(text: str) -> bool:
    return text.count("[[") >= 3 or text.count("{{") >= 2


def strip_wikitext(text: str) -> str:
    for _ in range(3):  # вложенные шаблоны
        new = _WIKI_TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new
    text = _WIKI_LINK_RE.sub(r"\1", text)
    text = _WIKI_EXT_LINK_RE.sub(r"\1", text)
    text = _WIKI_HEADING_RE.sub(r"\1", text)
    text = _WIKI_FORMAT_RE.sub("", text)
    text = _WIKI_TABLE_RE.sub("", text)
    return text


def remove_urls_emails(text: str) -> str:
    return EMAIL_RE.sub(" ", URL_RE.sub(" ", text))




TOKEN_DTYPE = np.uint16
LABEL_DTYPE = np.int32
IGNORE_INDEX = -100
SFT_LAYOUT = "sft_records"


def read_token_meta(bin_path: str | Path) -> dict:
    meta_path = Path(bin_path).parent / "meta.json"
    return read_json(meta_path) if meta_path.exists() else {}


def labels_path_for(bin_path: str | Path) -> Path:
    p = Path(bin_path)
    return p.with_name(p.stem + ".labels.bin")


def looks_like_sft(bin_path: str | Path, meta: Optional[dict] = None) -> bool:
    meta = meta if meta is not None else read_token_meta(bin_path)
    if meta.get("chat") or meta.get("has_labels") or meta.get("layout") == SFT_LAYOUT:
        return True
    for part in (Path(bin_path).name, Path(bin_path).parent.name):
        n = part.lower()
        if n in ("chat", "sft") or "_chat" in n or "_sft" in n:
            return True
    return False


class PackedTokens:
    def __init__(self, path: str | Path, seq_len: int, labels_path: Optional[str | Path] = None):
        self.path = str(path)
        self.data = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        self.seq_len = seq_len
        self.meta = read_token_meta(path)
        lp = Path(labels_path) if labels_path is not None else labels_path_for(path)
        self.labels = None
        self.has_labels = False
        if lp.exists():
            lab = np.memmap(str(lp), dtype=LABEL_DTYPE, mode="r")
            if len(lab) != len(self.data):
                raise ValueError(f"{lp}: длина {len(lab)} != токенам {len(self.data)}")
            self.labels = lab
            self.has_labels = True

        divisible = len(self.data) > 0 and len(self.data) % seq_len == 0
        self.record_mode = bool(
            self.has_labels
            and divisible
            and (self.meta.get("layout") == SFT_LAYOUT or self.meta.get("chat"))
        )
        if self.record_mode:
            self.n_chunks = len(self.data) // seq_len
        else:
            self.n_chunks = (len(self.data) - 1) // seq_len
        if self.n_chunks <= 0:
            raise ValueError(f"{path}: слишком мало токенов для seq_len={seq_len}")

    def batch(self, idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.seq_len
        if self.record_mode:
            tok = np.stack([np.asarray(self.data[int(i) * T: (int(i) + 1) * T]) for i in idx]).astype(np.int64)
            lab = np.stack([np.asarray(self.labels[int(i) * T: (int(i) + 1) * T]) for i in idx]).astype(np.int64)
            x = torch.from_numpy(tok.copy())
            y = np.concatenate(
                [lab[:, 1:], np.full((len(idx), 1), IGNORE_INDEX, dtype=np.int64)],
                axis=1,
            )
            return x, torch.from_numpy(y.copy())

        tok = np.stack([np.asarray(self.data[int(i) * T: int(i) * T + T + 1]) for i in idx]).astype(np.int64)
        x = torch.from_numpy(tok[:, :-1].copy())
        if self.labels is None:
            y = torch.from_numpy(tok[:, 1:].copy())
        else:
            lab = np.stack([np.asarray(self.labels[int(i) * T: int(i) * T + T + 1]) for i in idx]).astype(np.int64)
            y = torch.from_numpy(lab[:, 1:].copy())
        return x, y


class ChunkSampler:
    def __init__(self, ds: PackedTokens, batch_size: int, seed: int, rank: int = 0, world_size: int = 1):
        self.ds = ds
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.cursor = 0
        self._order = self._make_order()
        if len(self._order) == 0:
            raise ValueError("ChunkSampler: нет чанков для этого rank")

    def _make_order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        perm = rng.permutation(self.ds.n_chunks)
        return perm[self.rank::self.world_size]

    def _take_from_new_epoch(self, need: int) -> np.ndarray:
        got = []
        while need > 0:
            self.epoch += 1
            self._order = self._make_order()
            if len(self._order) == 0:
                raise ValueError("ChunkSampler: пустая эпоха")
            take = min(need, len(self._order))
            got.append(self._order[:take])
            self.cursor = take
            need -= take
        return np.concatenate(got) if got else np.zeros(0, dtype=np.int64)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + self.batch_size <= len(self._order):
            idx = self._order[self.cursor: self.cursor + self.batch_size]
            self.cursor += self.batch_size
            return self.ds.batch(idx)
        leftover = self._order[self.cursor:]
        rest = self._take_from_new_epoch(self.batch_size - len(leftover))
        idx = np.concatenate([leftover, rest]) if len(leftover) else rest
        return self.ds.batch(idx)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._order = self._make_order()





@torch.no_grad()
def evaluate_bin(
    model: torch.nn.Module,
    bin_path: str | Path,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    amp_ctx=None,
    max_tokens: Optional[int] = None,
    split_name: Optional[str] = None,
) -> dict:
    was_training = model.training
    model.eval()
    ds = PackedTokens(bin_path, seq_len)
    n_chunks = ds.n_chunks
    if max_tokens is not None:
        n_chunks = max(1, min(n_chunks, max_tokens // seq_len))
    amp_ctx = amp_ctx or nullcontext()
    total_nll, total_tok = 0.0, 0
    for i in range(0, n_chunks, batch_size):
        idx = np.arange(i, min(i + batch_size, n_chunks))
        x, y = ds.batch(idx)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx:
            _, loss = model(x, targets=y)
        n_valid = int((y != -100).sum().item())
        if n_valid == 0:
            continue
        total_nll += float(loss) * n_valid
        total_tok += n_valid
    if was_training:
        model.train()

    loss = total_nll / max(total_tok, 1)
    result = {"loss": loss, "ppl": math.exp(min(loss, 50.0)), "tokens": total_tok}

    meta = read_token_meta(bin_path)
    split_name = split_name or Path(bin_path).stem
    stats = meta.get("splits", {}).get(split_name)
    if ds.has_labels:
        result["sft_answer_tokens"] = total_tok
    elif stats and stats.get("tokens"):
        tpc = stats["tokens"] / max(stats["chars"], 1)
        tpw = stats["tokens"] / max(stats["words"], 1)
        result["bits_per_char"] = loss / math.log(2) * tpc
        result["word_ppl"] = math.exp(min(loss * tpw, 50.0))
    return result


def _ngrams(tokens: Sequence[str], n: int):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: List[List[str]], n: int) -> float:
    total, uniq = 0, set()
    for toks in samples:
        grams = _ngrams(toks, n)
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


def repetition_rate(tokens: Sequence[str], n: int) -> float:
    grams = _ngrams(tokens, n)
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def language_metrics(text: str) -> Dict[str, float]:
    c = lang_counts(text)
    toks = words(text)
    labels = Counter(classify_word(w) for w in toks)
    marked = labels["be"] + labels["ru"] + labels["uk"]
    sents = [s for s in split_sentences(text)]
    sent_labels = Counter(classify_text(s, min_letters=10) for s in sents)
    n_sent = sum(v for k, v in sent_labels.items() if k != "unk")
    foreign_letters = sum(text.lower().count(ch) for ch in FOREIGN_CYR)
    return {
        "n_words": len(toks),
        "cyrillic_letter_share": c.cyr / c.letters if c.letters else 0.0,
        "bel_char_ratio": 1.0 - foreign_letters / c.cyr if c.cyr else 0.0,
        "bel_word_ratio": labels["be"] / marked if marked else 1.0,
        "ru_word_share": labels["ru"] / len(toks) if toks else 0.0,
        "uk_word_share": labels["uk"] / len(toks) if toks else 0.0,
        "ru_sentence_share": sent_labels["ru"] / n_sent if n_sent else 0.0,
        "uk_sentence_share": sent_labels["uk"] / n_sent if n_sent else 0.0,
        "has_ru_switch": float(sent_labels["ru"] > 0),
        "has_uk_switch": float(sent_labels["uk"] > 0),
    }


def aggregate_generation_metrics(texts: List[str]) -> Dict[str, float]:
    per = [language_metrics(t) for t in texts]
    tok_lists = [[w.lower() for w in words(t)] for t in texts]
    out: Dict[str, float] = {"n_samples": len(texts)}
    if not per:
        return out
    for key in per[0]:
        out[f"mean_{key}"] = sum(p[key] for p in per) / len(per)
    out["russian_switch_rate"] = out.pop("mean_has_ru_switch")
    out["ukrainian_switch_rate"] = out.pop("mean_has_uk_switch")
    for n in (1, 2, 3):
        out[f"distinct_{n}"] = distinct_n(tok_lists, n)
    for n in (2, 3, 4):
        out[f"rep_{n}gram"] = sum(repetition_rate(t, n) for t in tok_lists) / len(tok_lists)
    return out
