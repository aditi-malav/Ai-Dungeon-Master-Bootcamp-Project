import os
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
if hasattr(st, "secrets"):
    for k, v in st.secrets.items():
        os.environ[str(k)] = str(v)

from dm_engine import DMEngine
from memory import MemoryManager
from prompts import SYSTEM_PROMPT


st.set_page_config(page_title="AI Dungeon Master", page_icon="🧙‍♂️", layout="centered")
st.title("AI Dungeon Master")

with st.sidebar:
    st.header("⚙️ Settings")

  
    default_use_rag = str(os.getenv("USE_SEMANTIC_RAG", "0")).lower() in ("1", "true", "yes")
    mode = st.radio("Select Mode", ["Lightweight", "Semantic RAG"], index=1 if default_use_rag else 0)
    use_rag = (mode == "Semantic RAG")
    st.session_state["USE_SEMANTIC_RAG"] = use_rag
    st.caption("Lightweight recall + optional Semantic RAG")

   
    default_len = int(os.getenv("MAX_TOKENS", "350"))
    reply_len = st.slider("📝 Reply Length (tokens)", 200, 600, default_len, 20)
    os.environ["MAX_TOKENS"] = str(reply_len)
    st.caption("Shorter replies reduce rate-limit errors and feel snappier.")

    btn_reset = st.button("Reset World")
    btn_recap = st.button("Show Recap")


mode_key = f"mode:{'rag' if use_rag else 'light'}"
if ("engine" not in st.session_state) or (st.session_state.get("engine_mode") != mode_key) or btn_reset:
    st.session_state.engine = DMEngine(use_semantic_rag=use_rag)
    
    st.session_state.memory = MemoryManager(short_window=4, use_semantic_rag=use_rag)
    st.session_state.engine_mode = mode_key
    st.session_state.turn = 0
    st.session_state.chat = []
    st.session_state.opening = (
        "You awaken at the edge of an ancient forest as dusk bleeds into starlight. "
        "A lantern flickers nearby beside a weathered signpost. Footsteps echo faintly from the road. "
        "What do you do?"
    )

engine = st.session_state.engine
memory = st.session_state.memory


st.markdown("### Scene")
st.info(st.session_state.opening)
st.info(f"🧠 Current Mode: {'Semantic RAG' if use_rag else 'Lightweight Memory'}")


if btn_recap:
    st.markdown("#### Recap")
    st.code(memory.recap(), language=None)


for role, text in st.session_state.chat:
    st.chat_message("user" if role == "user" else "assistant").markdown(text)


def build_messages(system_prompt: str, short_ctx: str, long_ctx: str, hints: str, user_text: str):
    parts = []
    if long_ctx: parts.append(long_ctx)
    if hints: parts.append(hints)
    if short_ctx: parts.append("Recent conversation:\n" + short_ctx)
    parts.append("Player says: " + user_text.strip())
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]

def guess_npc_focus(text: str):
    toks = [w.strip(",.;:!?") for w in text.split()]
    caps = []
    for w in toks:
        if len(w) > 0 and w[0].isupper() and w.isalpha():
            lw = w.lower()
            if lw not in caps:
                caps.append(lw)
    return caps[:2]

def trim_bullets(long_ctx: str, max_chars: int = 1200) -> str:
    """Trim long-term bullets to keep prompt small (rate-limit friendly)."""
    if not long_ctx or len(long_ctx) <= max_chars:
        return long_ctx
    lines = long_ctx.splitlines()
    header = lines[0] if lines else ""
    bullets = [l for l in lines[1:] if l.strip().startswith("-")]
    kept, size = [], len(header)
    for b in bullets:
        if size + len(b) + 1 > max_chars:
            break
        kept.append(b)
        size += len(b) + 1
    return "\n".join([header] + kept)


user_text = st.chat_input("Your move... (Tip: type `remember: <note>` to pin a fact)")

if user_text is not None:
    text = user_text.strip() or "Look around quietly."

    
    if text.lower() == "reset":
        st.session_state.engine = DMEngine(use_semantic_rag=use_rag)
        st.session_state.memory = MemoryManager(short_window=4, use_semantic_rag=use_rag)
        st.session_state.turn = 0
        st.session_state.chat = []
        st.success("World memory cleared. A new tale begins...")
        st.stop()

   
    if text.lower().startswith("remember:"):
        note = text.split(":", 1)[1].strip()
        if note:
            memory.pin_note(note)
            st.success(f"Pinned: {note}")
        else:
            st.warning("Usage: remember: <note>")
        st.stop()

    
    st.session_state.turn += 1
    npc_focus = guess_npc_focus(text)

    
    if st.session_state.turn % 4 == 0:
        long_ctx = memory.recall_relevant(k=3, cue=text, npc_focus=npc_focus)
    else:
        long_ctx = ""

    short_ctx = memory.get_short_context()
    hints = memory.pop_consistency_hints()
    long_ctx = trim_bullets(long_ctx, max_chars=1200)

    
    try:
        dm_reply = engine.chat(build_messages(SYSTEM_PROMPT, short_ctx, long_ctx, hints, text))
    except Exception as e:
        dm_reply = f"(Error contacting model: {e})"

    
    st.session_state.chat.append(("user", text))
    st.chat_message("user").markdown(text)
    st.session_state.chat.append(("assistant", dm_reply))
    st.chat_message("assistant").markdown(dm_reply)

    
    if " died" in dm_reply.lower() or " is dead" in dm_reply.lower():
        memory.add_consistency_hint("Reminder: confirm death states; notes may mark this NPC as alive.")

    
    memory.add_turn(text, dm_reply)
    memory.extract_key_events(dm_reply, st.session_state.turn)
    if st.session_state.turn % 4 == 0:
        try:
            memory.compact_long_memory(max_notes=int(os.getenv("MAX_NOTES", "18")))
        except Exception:
            pass


