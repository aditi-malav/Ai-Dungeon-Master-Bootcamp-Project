import os, json, re
from pathlib import Path
from typing import List, Dict, Any, Optional


_USE_RAG_OK = True
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    _USE_RAG_OK = False

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LTM_FILE = DATA_DIR / "long_term_memory.json"
KEY_FILE = DATA_DIR / "keyword_bank.json"
HINTS_FILE = DATA_DIR / "consistency_hints.json"


def _load_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_records(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Make sure {priority, turn} are ints and note is str.
    This prevents 'TypeError: can only concatenate str (not "int") to str'.
    """
    changed = False
    for r in recs:
        # priority
        if not isinstance(r.get("priority"), int):
            try:
                r["priority"] = int(r.get("priority", 1))
            except Exception:
                r["priority"] = 1
            changed = True
        # turn
        if not isinstance(r.get("turn"), int):
            try:
                r["turn"] = int(r.get("turn", 0))
            except Exception:
                r["turn"] = 0
            changed = True
        # note
        n = r.get("note")
        if n is not None and not isinstance(n, str):
            r["note"] = str(n)
            changed = True
    if changed:
        _save_json(LTM_FILE, recs)
    return recs


class MemoryManager:
    """
    Manages short-term turns and long-term notes.
    Dual-mode: Lightweight (default) or Semantic RAG (MiniLM + FAISS).
    """

    def __init__(self, short_window: int = 5, use_semantic_rag: bool = False):
        self.short_window = short_window
        self.use_semantic_rag = use_semantic_rag and _USE_RAG_OK
        if use_semantic_rag and not _USE_RAG_OK:
            print("RAG requested but sentence-transformers/faiss not available. Falling back to Lightweight.")

        # runtime state
        self.turns: List[Dict[str, str]] = []  # [{user, dm}]
        self.long_memory: List[Dict[str, Any]] = _load_json(LTM_FILE, [])
        self.long_memory = _normalize_records(self.long_memory)
        self.keyword_bank: Dict[str, int] = _load_json(KEY_FILE, {})
        self.hints: List[str] = _load_json(HINTS_FILE, [])

        # semantic index
        self.embedder = None
        self.index = None
        self.corpus: List[str] = []
        if self.use_semantic_rag:
            self._init_rag_index()

    #rag
    def _init_rag_index(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors (use normalized vectors)
        self.corpus = []
        # seed from existing notes
        for rec in self.long_memory:
            self._rag_add_text(rec.get("note", ""))

    def _rag_add_text(self, text: str):
        if not text or not self.embedder or not self.index:
            return
        vec = self.embedder.encode([text], normalize_embeddings=True)
        self.index.add(vec)
        self.corpus.append(text)

    def _rag_search(self, query: str, top_k: int = 5) -> List[str]:
        if not self.embedder or not self.index or self.index.ntotal == 0:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q, top_k)
        hits = []
        for idx in I[0]:
            if 0 <= idx < len(self.corpus):
                hits.append(self.corpus[idx])
        return hits

    #short/long
    def get_short_context(self) -> str:
        tail = self.turns[-self.short_window:]
        lines = []
        for t in tail:
            lines.append(f"Player: {t['user']}")
            lines.append(f"DM: {t['dm']}")
        return "\n".join(lines)

    def add_turn(self, user: str, dm: str):
        self.turns.append({"user": user, "dm": dm})

    def add_consistency_hint(self, hint: str):
        if hint and hint not in self.hints:
            self.hints.append(hint)
            _save_json(HINTS_FILE, self.hints)

    def pop_consistency_hints(self) -> str:
        if not self.hints:
            return ""
        hints = self.hints[:2]
        self.hints = self.hints[2:]
        _save_json(HINTS_FILE, self.hints)
        return "Consistency hints:\n" + "\n".join(f"- {h}" for h in hints)

    # Heuristic extraction 
    KEY_TAGS = {
        "quest": ["quest", "mission", "task", "contract"],
        "item": ["sword", "key", "map", "feather", "ring", "amulet", "potion", "lantern", "gem", "scroll"],
        "ally": ["joins you", "agrees to help", "companion", "party member", "ally"],
        "lore": ["prophecy", "legend", "ancient", "ritual", "temple", "kingdom", "order"],
    }

    def _priority(self, text: str) -> int:
        t = text.lower()
        score = 1
        for _, words in self.KEY_TAGS.items():
            if any(w in t for w in words):
                score += 2
        if "remember:" in t:
            score += 3
        return min(score, 5)

    def pin_note(self, note: str):
        rec = {"note": note, "tags": ["HIGH"], "priority": 5, "turn": -1}
        self.long_memory.append(rec)
        if self.use_semantic_rag:
            self._rag_add_text(note)
        _save_json(LTM_FILE, self.long_memory)

    def extract_key_events(self, dm_reply: str, turn: int):
        """Very small heuristic: take 1–2 strongest lines from the reply."""
        lines = [s.strip() for s in re.split(r"[.\n]", dm_reply) if s.strip()]
        kept = []
        for ln in lines:
            if any(
                w in ln.lower()
                for w in [
                    "find",
                    "give",
                    "take",
                    "enter",
                    "warn",
                    "agree",
                    "attack",
                    "discover",
                    "feather",
                    "lantern",
                    "key",
                    "map",
                ]
            ):
                kept.append(ln)
            if len(kept) >= 2:
                break
        for ln in (kept or lines[:1]):
            rec = {"note": ln, "tags": [], "priority": self._priority(ln), "turn": turn}
            self.long_memory.append(rec)
            if self.use_semantic_rag:
                self._rag_add_text(ln)
        _save_json(LTM_FILE, self.long_memory)

    def recall_relevant(self, k: int, cue: str, npc_focus: Optional[List[str]] = None) -> str:
        """Return bullets of relevant memory depending on mode (robust to mixed types)."""
        header = "Relevant notes:"
        bullets: List[str] = []

        if self.use_semantic_rag:
            top = self._rag_search(cue, top_k=k) or []
            bullets = [f"- {t}" for t in top]
        else:
            
            cue_l = (cue or "").lower()
            focus = [w.lower() for w in (npc_focus or [])]

            def to_int(x, default=0):
                try:
                    return int(x)
                except Exception:
                    return default

            def score(rec: Dict[str, Any]):
                s = to_int(rec.get("priority", 1), 1) 
                n = ((rec.get("note") or "")).lower()  
                if any(w and w in n for w in focus):
                    s += 1
                if any(w and w in n for w in cue_l.split()):
                    s += 1
                return (s, to_int(rec.get("turn", 0), 0))  

            pool = [r for r in self.long_memory[-200:] if (r.get("note") or "").strip()]
            ranked = sorted(pool, key=score, reverse=True)[:k]
            bullets = [f"- {r['note']}" for r in ranked]

        if not bullets:
            return ""
        return header + "\n" + "\n".join(bullets)

  
    def recap(self) -> str:
        tail = self.long_memory[-12:]
        if not tail:
            return "No long-term notes yet."
        lines = [f"- {r.get('note','')}" for r in tail if r.get("note")]
        return "Recent world notes:\n" + "\n".join(lines)

    def compact_long_memory(self, max_notes: int = 18):
        """Deduplicate + cap stored notes to keep prompts small."""
        if not self.long_memory:
            return
        seen = set()
        compact = []
        for rec in reversed(self.long_memory):
            note = (rec.get("note") or "").strip().lower()
            if note and note not in seen:
                seen.add(note)
                compact.append(rec)
            if len(compact) >= max_notes:
                break
        self.long_memory = list(reversed(compact))
        _save_json(LTM_FILE, self.long_memory)


