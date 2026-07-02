# 🧙 AI Dungeon Master

An AI-powered text adventure game where a Large Language Model (LLM) acts as the Dungeon Master, generating immersive RPG-style stories based on player actions.

Unlike traditional chatbots that forget earlier conversations, this project explores **memory architectures for LLMs**, enabling the model to remember important events, characters, locations, quests, and player decisions throughout long-running adventures.

The project compares **two different retrieval strategies** over a shared long-term memory store:

- **Lightweight Retrieval** – keyword matching with priority-based ranking.
- **Semantic Retrieval (RAG)** – embedding-based retrieval using Sentence Transformers and FAISS.

The objective was to understand how different memory retrieval techniques affect story consistency, contextual recall, and long-term interaction quality.

---

# 🎥 Live Demo

## Application

https://ai-dungeon-master-bootcamp-project-sqtqtfhggqmpqsecogwbvh.streamlit.app/

## Demo Video

https://streamable.com/ve9pnd

---

# ✨ Features

## 🎮 Interactive Storytelling

- AI-generated RPG adventures
- Player-driven story progression
- Dynamic world generation
- Streamlit-based web interface

---

## 🧠 Dual Retrieval Modes

### Lightweight Retrieval

A lightweight retrieval mechanism that recalls memories using:

- Keyword matching
- Priority scoring
- NPC name matching
- Recent conversation history

Designed to be computationally inexpensive while maintaining reasonable contextual recall.

---

### Semantic Retrieval (RAG)

A semantic retrieval pipeline built using:

- Sentence Transformers (MiniLM)
- FAISS vector search
- Embedding similarity

Instead of relying on exact keyword matches, this mode retrieves memories based on semantic similarity.

---

## 📖 Long-Term Memory

Important story events are automatically extracted and stored as persistent notes.

Examples include:

- Discovering items
- Receiving quests
- Unlocking locations
- Meeting important NPCs
- Important player decisions

Players can also manually pin memories using:

```text
remember: The silver key opens the northern vault.
```

Pinned memories receive maximum priority during retrieval.

---

## ⚡ Reliability Features

- API retry mechanism for rate limits
- Prompt size management
- Memory deduplication
- Configurable response length
- Automatic consistency hints

---

# 🏗️ High-Level System Architecture

```text
                           Player
                              │
                              ▼
                       Streamlit Frontend
                              │
                              ▼
                        Player Action
                              │
                              ▼
                      Memory Manager
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
 Short-Term Memory                        Long-Term Memory
(Recent Conversation)               (Persistent JSON Storage)
          │                                       │
          │               ┌───────────────────────┴────────────────────────┐
          ▼               ▼                                                ▼
 Lightweight Retrieval                           Semantic Retrieval
 (Keyword + Priority)                   (MiniLM Embeddings + FAISS)
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                    Retrieved Memories
                              │
                              ▼
                     Prompt Construction
                              │
                              ▼
                  Groq / OpenAI Chat API
                              │
                              ▼
                   Dungeon Master Response
                              │
                              ▼
                    Event Extraction
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   Update Short-Term Memory        Update Long-Term Memory
```

---

# 📌 Project Motivation

Large Language Models have limited context windows and do not naturally remember earlier conversations across long interactions.

This project explores different strategies for providing external memory to an LLM, enabling it to:

- remember previous story events,
- recall important characters,
- maintain quest continuity,
- preserve world consistency,
- and generate more coherent long-form adventures.

Rather than fine-tuning the model, memory is implemented externally using retrieval techniques and prompt engineering.\

---

# 🧠 Memory Architecture

The memory subsystem separates **memory storage** from **memory retrieval**.

Every important event generated during gameplay is stored in a persistent long-term memory store. Different retrieval strategies can then operate over the same stored memories.

## Storage Layer

The project maintains two levels of memory.

### Short-Term Memory

Short-term memory stores only the most recent conversation turns during the current session.

It consists of alternating player actions and Dungeon Master responses:

```text
Player: Open the ancient door.

DM: The heavy stone door slowly opens...
```

Only the most recent **N turns** (configurable using `short_window`) are included in future prompts to maintain immediate conversational context while keeping prompts small.

---

### Long-Term Memory

Long-term memory stores important story events that should persist beyond the recent conversation.

Examples include:

- Items discovered
- Important NPCs
- Completed quests
- Important locations
- Significant story events
- Player-pinned memories

These memories are stored persistently in

```text
data/long_term_memory.json
```

Each stored memory contains metadata:

```json
{
    "note": "Player obtained the Silver Key",
    "priority": 5,
    "turn": 18,
    "tags": []
}
```

Unlike short-term memory, long-term memory survives application restarts.

---

# 🔄 Retrieval Strategies

Both retrieval strategies operate over the same long-term memory store.

The difference lies only in **how memories are selected**.

---

## Lightweight Retrieval

The lightweight retriever performs lexical matching without generating embeddings.

Retrieval pipeline:

```text
Player Query
      │
      ▼
Convert to lowercase
      │
      ▼
Keyword Matching
      │
      ▼
NPC Matching
      │
      ▼
Priority Scoring
      │
      ▼
Recent Memories
      │
      ▼
Top-K Relevant Notes
```

Each stored memory receives a ranking score based on:

- Priority
- Keyword overlap
- NPC name match
- Recency

The highest-ranked memories are inserted into the prompt.

### Advantages

- Very fast
- No embedding model required
- No vector search
- Low computational cost

### Limitations

- Depends on exact keywords
- Cannot understand paraphrases
- Limited semantic understanding

---

## Semantic Retrieval (RAG)

Instead of matching keywords, Semantic Retrieval searches for memories based on meaning.

### Index Construction

When Semantic Retrieval is enabled:

```text
Stored Notes
      │
      ▼
Sentence Transformer (MiniLM)
      │
      ▼
Generate Embeddings
      │
      ▼
Store Embeddings
      │
      ▼
FAISS Index
```

The original text remains stored in JSON while the vector representations are stored in a FAISS index.

---

### Retrieval Pipeline

During gameplay:

```text
Player Query
      │
      ▼
Sentence Transformer
      │
      ▼
Query Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Nearest Embeddings
      │
      ▼
Retrieve Original Notes
      │
      ▼
Add Notes to Prompt
```

Instead of requiring exact keyword matches, memories are retrieved based on semantic similarity.

For example,

Stored memory:

```text
Player obtained an enchanted sword.
```

Player later asks:

```text
Draw my magical blade.
```

Although the words are different, semantic retrieval identifies both as referring to the same concept.

---

# 📝 Prompt Construction

Before every LLM call, the project constructs a single prompt containing all relevant context.

```text
                 System Prompt
                       │
                       ▼
           Recent Conversation
                       │
                       ▼
      Retrieved Long-Term Memories
                       │
                       ▼
          Consistency Hints
                       │
                       ▼
             Current Player Input
                       │
                       ▼
              Final Prompt
                       │
                       ▼
                    LLM
```

This allows the language model to generate responses while remaining consistent with previous story events.

---

# 🔄 Complete Memory Lifecycle

```text
Player Action
      │
      ▼
Generate Prompt
      │
      ▼
LLM Response
      │
      ▼
Extract Important Events
      │
      ▼
Assign Priority
      │
      ▼
Store in long_term_memory.json
      │
      ▼
(If Semantic Mode)
      │
      ▼
Generate Embedding
      │
      ▼
Insert into FAISS
```

This process repeats after every interaction, allowing the world state to evolve continuously.

---

# 📌 Memory Retrieval Comparison

| Feature | Lightweight Retrieval | Semantic Retrieval |
|----------|----------------------|--------------------|
| Keyword Matching | ✅ | ❌ |
| Embedding Search | ❌ | ✅ |
| FAISS | ❌ | ✅ |
| Exact Match | ✅ | ⚠️ Not Required |
| Semantic Understanding | ❌ | ✅ |
| Computational Cost | Low | Moderate |
| Best Use Case | Small memory, exact recall | Long conversations, semantic recall |

---

# 🛠️ Technology Stack

## Backend

- Python 3.10+
- Requests
- JSON
- dotenv

## Frontend

- Streamlit

## Large Language Models

- Groq API
- OpenAI API

The project is provider-independent and can switch between Groq and OpenAI through environment variables.

---

## Memory & Retrieval

### Lightweight Retrieval

- Keyword Matching
- Priority-based Ranking
- Recency-aware Retrieval
- NPC-aware Retrieval

### Semantic Retrieval

- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Cosine Similarity Search

---

## AI Concepts Used

- Prompt Engineering
- Retrieval-Augmented Generation (RAG)
- Conversational Memory
- Semantic Search
- Vector Embeddings
- Similarity Search
- Context Management

---

# 📂 Project Structure

```text
AI-Dungeon-Master/
│
├── app.py                 # Streamlit UI
├── dm_engine.py           # LLM API communication
├── memory.py              # Memory manager & retrieval
├── prompts.py             # System prompt
├── requirements.txt
├── README.md
│
├── data/
│   ├── long_term_memory.json
│   ├── keyword_bank.json
│   └── consistency_hints.json
│
└── .env
```

---

# 📁 File Overview

## app.py

Responsible for:

- Streamlit user interface
- Player interaction
- Prompt construction
- Switching retrieval modes
- Managing game session

---

## dm_engine.py

Acts as the communication layer between the application and the language model.

Responsibilities:

- Provider selection
- API authentication
- HTTP request handling
- Retry mechanism
- Rate-limit handling
- Returning generated responses

---

## memory.py

Core component of the project.

Responsible for:

- Short-term memory
- Long-term memory
- Memory persistence
- Event extraction
- Lightweight retrieval
- Semantic retrieval (RAG)
- Memory cleanup
- Consistency hints

---

## prompts.py

Contains the Dungeon Master's system prompt that defines:

- Story style
- Response constraints
- World consistency
- NPC behaviour
- Memory usage

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/aditi-malav/Ai-Dungeon-Master-Bootcamp-Project.git

cd Ai-Dungeon-Master-Bootcamp-Project
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file.

```env
PROVIDER=groq

MODEL_NAME=llama-3.1-8b-instant

GROQ_API_KEY=your_api_key

OPENAI_API_KEY=your_api_key

USE_SEMANTIC_RAG=1

MAX_TOKENS=350

MAX_NOTES=18
```

---

# ▶️ Running the Project

```bash
streamlit run app.py
```

The application opens in your default browser.

---

# 🎮 Gameplay Example

Player

```text
Enter the abandoned castle.
```

↓

Dungeon Master

```text
The castle doors slowly creak open. Dust fills the air as broken chandeliers sway above you...
```

↓

Player

```text
remember: The hidden staircase is behind the throne.
```

↓

The note is permanently stored and can be retrieved later during gameplay.

---

# ⚙️ Configuration

The following settings can be adjusted directly from the Streamlit sidebar.

- Retrieval Mode
  - Lightweight
  - Semantic RAG

- Maximum Response Tokens

- Reset World

- Show Story Recap

These settings allow users to compare retrieval strategies while controlling response length.

---

# 🎯 Design Decisions

## Why Two Retrieval Modes?

The primary objective of this project was not only to build an AI Dungeon Master but also to explore different retrieval strategies for providing memory to Large Language Models.

Rather than coupling memory storage with a single retrieval technique, the project separates **memory storage** from **memory retrieval**.

Both retrieval modes operate over the same persistent long-term memory but differ in how relevant memories are selected.

### Lightweight Retrieval

Lightweight Retrieval uses:

- Keyword matching
- Priority-based ranking
- NPC name matching
- Recency

This approach is computationally inexpensive and works well when player queries contain similar keywords to stored memories.

### Semantic Retrieval (RAG)

Semantic Retrieval converts memories into dense vector embeddings using Sentence Transformers and retrieves them using FAISS similarity search.

Unlike keyword matching, semantic retrieval can recall memories even when different words are used.

Example:

Stored Memory

```text
Player obtained an enchanted sword.
```

Later Query

```text
Draw my magical blade.
```

Although none of the keywords match exactly, semantic similarity retrieves the correct memory.

---

# Why Sentence Transformers?

The project requires converting textual memories into dense vector embeddings.

Sentence Transformers provide:

- Sentence-level embeddings
- Fast inference
- Lightweight models
- Strong semantic representations

The MiniLM model (`all-MiniLM-L6-v2`) was selected because it offers an excellent balance between retrieval quality and computational efficiency.

---

# Why FAISS?

FAISS is an efficient vector similarity search library developed by Meta.

Instead of comparing a query against every stored embedding, FAISS indexes vectors and performs efficient nearest-neighbor search.

Benefits include:

- Fast similarity search
- Scalable retrieval
- Easy integration
- Open-source implementation

---

# Why JSON Storage?

Long-term memories are stored in JSON because:

- Simple persistence
- Human-readable format
- Easy debugging
- Suitable for prototype-scale applications

For larger systems, a database-backed memory layer would be preferable.

---

# Why Prompt Engineering Instead of Fine-Tuning?

The objective of the project was to improve contextual consistency without modifying the underlying language model.

Memory is injected into prompts dynamically, allowing the same model to generate context-aware responses without additional training.

---

# ⚠️ Current Limitations

The current implementation was designed as a prototype for exploring conversational memory architectures.

Some limitations remain:

- Event extraction relies on keyword-based heuristics.
- Memories are stored as plain text rather than structured entities.
- FAISS index is reconstructed whenever Semantic Retrieval is initialized.
- Retrieval modes operate independently rather than as a hybrid retriever.
- Long-term memory currently uses a local JSON store rather than a production database.

Despite these limitations, the architecture demonstrates the complete retrieval pipeline used in modern RAG-based conversational systems.

---

# 🚀 Future Improvements

The following improvements could significantly enhance the system.

## Smarter Event Extraction

Replace heuristic keyword extraction with an LLM that generates structured memories.

Example:

```json
{
    "type": "item",
    "entity": "Silver Key",
    "importance": 5,
    "location": "Castle"
}
```

---

## Structured Memory

Instead of storing plain text notes, organize memories into structured entities.

Example:

- Characters
- Quests
- Items
- Locations
- Story Events

This enables more accurate retrieval and easier querying.

---

## Hybrid Retrieval

Combine the strengths of both retrieval strategies.

```text
Player Query
      │
      ▼
Keyword Retrieval
      │
      ├───────────────┐
      ▼               │
Semantic Retrieval    │
      │               │
      └──────┬────────┘
             ▼
      Merge Results
             ▼
      Remove Duplicates
             ▼
         Top-K Notes
             ▼
            LLM
```

Hybrid retrieval is widely used in modern Retrieval-Augmented Generation systems.

---

## Persistent Vector Database

Replace the in-memory FAISS index with a persistent vector database such as:

- Qdrant
- ChromaDB
- Pinecone

This avoids rebuilding embeddings whenever the application restarts and enables scalable deployments.

---

## Session-Based Memory

Currently, all long-term memories are stored locally.

A production implementation would isolate memories per:

- User
- Story
- Session

allowing multiple independent adventures.

---

## LLM-Based Memory Ranking

Instead of heuristic priority scoring, an LLM could estimate the long-term importance of each memory before storing it.

---

## Character Knowledge Graph

Represent NPCs, quests, and relationships using a graph structure for more consistent world-building.

---

## 📚 Learning Outcomes

Through this project I explored:

- Designing conversational memory systems
- Retrieval-Augmented Generation (RAG)
- Semantic search using Sentence Transformers
- Vector similarity search with FAISS
- Prompt engineering
- Long-term context management
- LLM API integration
- Streamlit application development
- Trade-offs between keyword-based and semantic retrieval
- Building end-to-end AI applications





