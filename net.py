"""Сеть, кэш, генерация, чат, токенизатор, чекпойнт."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dataclasses import asdict, dataclass, fields
from typing import Optional



@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: Optional[int] = None
    intermediate_size: int = 2560
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    dropout: float = 0.0
    attn_impl: str = "auto"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads

    def describe(self) -> str:
        return f"{self.n_layers}x{self.d_model}, {self.n_heads}/{self.n_kv_heads}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})





class KVCache:

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.seen_tokens = 0
        self._v: List[Optional[torch.Tensor]] = [None] * n_layers
        self._len: List[int] = [0] * n_layers

    @staticmethod
    def _append(buf: Optional[torch.Tensor], used: int, new: torch.Tensor) -> torch.Tensor:
        need = used + new.shape[2]
        if buf is None or buf.shape[2] < need:
            cap = max(need, 2 * (buf.shape[2] if buf is not None else 0), 64)
            grown = new.new_empty(new.shape[0], new.shape[1], cap, new.shape[3])
            if buf is not None and used > 0:
                grown[:, :, :used] = buf[:, :, :used]
            buf = grown
        buf[:, :, used:need] = new
        return buf

    def update(self, layer_idx: int, v: torch.Tensor):
        past = self._len[layer_idx]
        k_start = self.seen_tokens - past
        self._v[layer_idx] = self._append(self._v[layer_idx], past, v)
        self._len[layer_idx] = past + v.shape[2]
        return self._v[layer_idx][:, :, : self._len[layer_idx]], k_start

    def logical_nbytes(self) -> int:
        total = 0
        for i in range(self.n_layers):
            t = self._v[i]
            if t is not None:
                n = self._len[i]
                total += t.shape[0] * t.shape[1] * n * t.shape[3] * t.element_size()
        return total


def analytic_kv_cache_bytes(cfg: ModelConfig, context: int, bytes_per_elem: int = 2, batch: int = 1) -> int:
    per_token = cfg.n_kv_heads * cfg.head_dim * bytes_per_elem
    return batch * cfg.n_layers * context * per_token




try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


def _probe_sdpa_gqa() -> bool:
    try:
        q, kv = torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


SDPA_GQA = _probe_sdpa_gqa()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, initial_len: int):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(initial_len)

    def _build(self, length: int) -> None:
        t = torch.arange(length, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, start: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = start + length
        if end > self.cos_cached.shape[0]:
            self._build(max(end, 2 * self.cos_cached.shape[0]))
        return self.cos_cached[start:end], self.sin_cached[start:end]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * cos + _rotate_half(xf) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.attn_impl = cfg.attn_impl
        self.dropout = cfg.dropout
        d, H, KV, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(d, H * hd, bias=False)
        self.v_proj = nn.Linear(d, KV * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, d, bias=False)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, KV, hd = self.n_heads, self.n_kv_heads, self.head_dim
        past = cache.seen_tokens if cache is not None else 0

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV, hd).transpose(1, 2)
        cos_q, sin_q = rope(past, T)
        q = apply_rope(q, cos_q, sin_q)

        k_start = past
        if cache is not None:
            v, k_start = cache.update(self.layer_idx, v)
        cos_k, sin_k = rope(k_start, v.shape[2])
        k = apply_rope(v, cos_k, sin_k)

        out = self._attend(q, k, v, past, k_start)
        out = out.transpose(1, 2).reshape(B, T, H * hd)
        return self.o_proj(out)

    def _resolve_impl(self, q: torch.Tensor) -> str:
        impl = self.attn_impl
        flash_ok = HAS_FLASH_ATTN and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        if impl == "auto":
            return "flash2" if flash_ok else "sdpa"
        if impl == "flash2" and not flash_ok:
            return "sdpa"
        return impl

    def _expand_kv(self, t: torch.Tensor) -> torch.Tensor:
        rep = self.n_heads // self.n_kv_heads
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    def _attend(self, q, k, v, q_start: int, k_start: int) -> torch.Tensor:
        impl = self._resolve_impl(q)
        dropout_p = self.dropout if self.training else 0.0
        if impl == "flash2":
            out = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=dropout_p, causal=True, window_size=(-1, -1),
            )
            return out.transpose(1, 2)
        if impl == "eager":
            return self._eager(q, k, v, q_start, k_start, dropout_p)
        return self._sdpa(q, k, v, q_start, k_start, dropout_p)

    def _sdpa(self, q, k, v, q_start, k_start, dropout_p):
        T, L = q.shape[2], k.shape[2]
        if SDPA_GQA:
            kw = {"enable_gqa": self.n_heads != self.n_kv_heads}
        else:
            k, v = self._expand_kv(k), self._expand_kv(v)
            kw = {}
        if T == L and q_start == k_start:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True, **kw)
        if T == 1:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, **kw)
        q_pos = torch.arange(q_start, q_start + T, device=q.device)[:, None]
        k_pos = torch.arange(k_start, k_start + L, device=q.device)[None, :]
        mask = k_pos <= q_pos
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, **kw)

    def _eager(self, q, k, v, q_start, k_start, dropout_p):
        T, L = q.shape[2], k.shape[2]
        k, v = self._expand_kv(k), self._expand_kv(v)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        q_pos = torch.arange(q_start, q_start + T, device=q.device)[:, None]
        k_pos = torch.arange(k_start, k_start + L, device=q.device)[None, :]
        scores = scores.masked_fill(~(k_pos <= q_pos), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        if dropout_p > 0:
            probs = F.dropout(probs, p=dropout_p)
        return torch.matmul(probs, v.float()).to(q.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.intermediate_size)
        self.resid_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), rope, cache))
        return x + self.resid_dropout(self.mlp(self.mlp_norm(x)))


class BelarusianLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_seq_len)
        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=resid_std)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        last_only: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        T = input_ids.shape[1]
        x = self.emb_dropout(self.embed(input_ids))
        use_ckpt = self.gradient_checkpointing and self.training and cache is None
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, self.rope, None, use_reentrant=False)
            else:
                x = block(x, self.rope, cache)
        if cache is not None:
            cache.seen_tokens += T
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            flat_t = targets.reshape(-1)
            if not (flat_t != -100).any():
                return logits, logits.sum() * 0.0
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), flat_t, ignore_index=-100
            )
            return logits, loss
        if last_only:
            x = x[:, -1:]
        return self.lm_head(x), None





@dataclass
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.4
    top_k: int = 40
    top_p: float = 0.85
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    use_cache: bool = True


def _apply_repetition_penalty(logits: torch.Tensor, history: List[int], penalty: float) -> None:
    if penalty == 1.0 or not history:
        return
    idx = torch.tensor(sorted(set(history)), device=logits.device, dtype=torch.long)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)


def _ban_repeated_ngrams(logits: torch.Tensor, history: List[int], n: int) -> None:
    if n <= 0 or len(history) < n - 1:
        return
    prefix = tuple(history[len(history) - (n - 1):]) if n > 1 else tuple()
    banned = set()
    for i in range(len(history) - n + 1):
        if tuple(history[i:i + n - 1]) == prefix:
            banned.add(history[i + n - 1])
    if banned:
        logits[list(banned)] = float("-inf")


def sample_next(logits: torch.Tensor, sc: SamplingConfig, generator: Optional[torch.Generator]) -> int:
    if sc.temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / sc.temperature
    if sc.top_k and sc.top_k > 0:
        k = min(sc.top_k, logits.shape[-1])
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if sc.top_p and 0 < sc.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > sc.top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    if generator is not None and probs.device.type != generator.device.type:
        return int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate(
    model: BelarusianLM,
    prompt_ids: List[int],
    sc: SamplingConfig,
    eos_id: Optional[int] = None,
    stop_ids: Optional[List[int]] = None,
    stop_strings: Optional[List[str]] = None,
    decode_fn=None,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    model.eval()
    device = next(model.parameters()).device
    max_len = model.cfg.max_seq_len
    stop = set(stop_ids or [])
    if eos_id is not None:
        stop.add(eos_id)

    history = list(prompt_ids)
    out: List[int] = []
    cache = KVCache(model.cfg.n_layers) if sc.use_cache else None
    ctx = history if sc.use_cache else history[-max_len:]
    logits, _ = model(torch.tensor([ctx], device=device), cache=cache, last_only=True)

    for _ in range(sc.max_new_tokens):
        next_logits = logits[0, -1].float().clone()
        _apply_repetition_penalty(next_logits, history, sc.repetition_penalty)
        _ban_repeated_ngrams(next_logits, history, sc.no_repeat_ngram_size)
        tok = sample_next(next_logits, sc, generator)
        if tok in stop:
            break
        out.append(tok)
        history.append(tok)
        if stop_strings and decode_fn is not None:
            dec = decode_fn(out)
            if any(s in dec for s in stop_strings):
                break
        if sc.use_cache:
            logits, _ = model(torch.tensor([[tok]], device=device), cache=cache, last_only=True)
        else:
            logits, _ = model(torch.tensor([history[-max_len:]], device=device), last_only=True)
    return out


IGNORE_INDEX = -100
QUESTION_PREFIX = "Пытанне: "
ANSWER_PREFIX = "Адказ: "
STOP_STRINGS = (
    "\nПытанне:",
    "\nПытанне ",
    "\n\nПытанне:",
    "\nАдказ:",
)

_ANSWER_SEPS = ("\nАдказ:", "\nАдказ :", "\nAдказ:")


def format_turn(question: str, answer: Optional[str] = None) -> str:
    q = (question or "").strip()
    if answer is None:
        return f"{QUESTION_PREFIX}{q}\n{ANSWER_PREFIX}"
    return f"{QUESTION_PREFIX}{q}\n{ANSWER_PREFIX}{answer.strip()}"


def format_prompt(question: str, history: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    parts = [format_turn(q, a) for q, a in (history or ())]
    parts.append(format_turn(question))
    return "\n".join(parts)


def format_dialog(question: str, answer: str, history: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    parts = [format_turn(q, a) for q, a in (history or ())]
    parts.append(format_turn(question, answer))
    return "\n".join(parts)


def strip_stop_strings(text: str, extra: Iterable[str] = ()) -> str:
    cut = text
    for s in (*STOP_STRINGS, *extra):
        i = cut.find(s)
        if i >= 0:
            cut = cut[:i]
    return cut.rstrip()


def parse_qa_text(text: str) -> Optional[Tuple[str, str]]:
    raw = (text or "").strip()
    if not raw:
        return None
    sep = next((s for s in _ANSWER_SEPS if s in raw), None)
    if sep is None:
        return None
    q, a = raw.split(sep, 1)
    q, a = q.strip(), a.strip()
    if q.startswith("Пытанне:"):
        q = q[len("Пытанне:"):].strip()
    if len(q) < 3 or len(a) < 1:
        return None
    return q, a


def _prefix_len(tok, full_ids: List[int], prefix: str) -> int:
    pref_ids = tok.encode(prefix)
    if full_ids[: len(pref_ids)] == pref_ids:
        return len(pref_ids)
    n = 0
    for i in range(1, len(full_ids) + 1):
        dec = tok.decode(full_ids[:i])
        if prefix.startswith(dec):
            n = i
        if dec.startswith(prefix) and len(dec) >= len(prefix):
            return i
    return n


def encode_supervised(tok, question: str, answer: str, history: Optional[Sequence[Tuple[str, str]]] = None):
    prefix = format_prompt(question, history)
    text = format_dialog(question, answer, history)
    ids = tok.encode(text) + [tok.eos_id]
    n_mask = _prefix_len(tok, ids, prefix)
    n_mask = min(max(n_mask, 0), len(ids) - 1)
    labels = [IGNORE_INDEX] * n_mask + ids[n_mask:]
    return ids, labels


def pad_supervised(
    ids: List[int],
    labels: List[int],
    max_len: int,
    pad_id: int,
) -> Optional[Tuple[List[int], List[int]]]:
    if len(ids) != len(labels):
        raise ValueError(f"ids/labels: {len(ids)} != {len(labels)}")
    if len(ids) > max_len:
        ids, labels = ids[-max_len:], labels[-max_len:]
        if all(x == IGNORE_INDEX for x in labels):
            return None
    if len(ids) < 4:
        return None
    pad_n = max_len - len(ids)
    if pad_n:
        ids = ids + [pad_id] * pad_n
        labels = labels + [IGNORE_INDEX] * pad_n
    return ids, labels



class BelTokenizer:

    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        self.nl_id = self.sp.piece_to_id("<0x0A>")
        if self.nl_id == self.unk_id:
            raise ValueError("В токенизаторе нет byte-fallback токена <0x0A>")
        self._special = {i for i in (self.bos_id, self.eos_id, self.pad_id) if i is not None and i >= 0}

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for i, para in enumerate(text.split("\n")):
            if i > 0:
                ids.append(self.nl_id)
            if para:
                ids.extend(self.sp.encode(para))
        return ids

    def encode_document(self, text: str) -> List[int]:
        return self.encode(text) + [self.eos_id]

    def decode(self, ids: Iterable[int]) -> str:
        parts, cur = [], []
        for t in ids:
            t = int(t)
            if t == self.nl_id:
                parts.append(self.sp.decode(cur))
                cur = []
            elif t in self._special:
                continue
            else:
                cur.append(t)
        parts.append(self.sp.decode(cur))
        return "\n".join(parts)

    def id_to_piece(self, i: int) -> str:
        return self.sp.id_to_piece(int(i))

    def is_byte(self, i: int) -> bool:
        return self.sp.is_byte(int(i))
    
INIT_MUST_MATCH = (
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "intermediate_size",
)


def resolve_weight_path(src: str | Path) -> Path:
    """runs/<name> → model_best.pt / model_final.pt; либо явный путь к .pt."""
    base = Path(src)
    if base.is_file() and base.suffix == ".pt":
        return base
    for name in ("model_best.pt", "model_final.pt"):
        cand = base / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"--init-from: не найден весовой файл ({src})")


def unwrap_state_dict(state: dict) -> dict:
    if not state:
        return state
    if all(str(k).startswith("module.") for k in state):
        return {str(k)[len("module."):]: v for k, v in state.items()}
    return {str(k)[len("module."):] if str(k).startswith("module.") else k: v for k, v in state.items()}


def assert_init_compatible(model_cfg: ModelConfig, ckpt_cfg: dict, path: str | Path) -> None:
    src = ModelConfig.from_dict(ckpt_cfg)
    bad = []
    for key in INIT_MUST_MATCH:
        if getattr(model_cfg, key) != getattr(src, key):
            bad.append(f"{key}: model={getattr(model_cfg, key)!r} checkpoint={getattr(src, key)!r}")
    if bad:
        raise ValueError(f"--init-from {path}: несовместимая архитектура: " + "; ".join(bad))


def load_model(
    path: str | Path, device: str | torch.device = "cpu", attn_impl: Optional[str] = None
) -> Tuple[BelarusianLM, dict]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["model_config"])
    if attn_impl is not None:
        cfg_dict["attn_impl"] = attn_impl
    cfg = ModelConfig.from_dict(cfg_dict)
    model = BelarusianLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def export_model(model: BelarusianLM, path: str | Path, **extra) -> None:
    payload = {"model": model.state_dict(), "model_config": model.cfg.to_dict()}
    payload.update(extra)
    torch.save(payload, str(path))
