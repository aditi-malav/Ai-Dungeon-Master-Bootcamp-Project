# AI Dungeon Master

> **An AI-powered RPG storytelling engine that explores long-term memory architectures for Large Language Models (LLMs).**

AI Dungeon Master is an interactive text-based adventure game where an LLM acts as the Dungeon Master, generating immersive stories based on player actions. The project focuses on overcoming the limited context window of LLMs by introducing a persistent memory system capable of remembering important events, characters, quests, locations, and player decisions throughout long-running adventures.

The project implements and compares two interchangeable retrieval strategies operating on the same memory store:

- **Lightweight Retrieval** — Keyword Matching + Priority Ranking
- **Semantic Retrieval (RAG)** — Sentence Transformers + FAISS

This modular design makes it easy to experiment with different memory architectures while improving story consistency and contextual recall.

---

## Live Demo

### Application

https://ai-dungeon-master-bootcamp-project-sqtqtfhggqmpqsecogwbvh.streamlit.app/

### Demo Video

https://streamable.com/ve9pnd

---

# Project Highlights

- Dual memory architectures for long-term conversational memory
- Switchable keyword-based and semantic retrieval pipelines
- Persistent long-term memory across sessions
- Automatic extraction and prioritization of important story events
- Manual memory pinning using `remember:`
- Modular architecture supporting Groq and OpenAI APIs
- Interactive Streamlit-based web application
- Designed to study practical memory systems for LLM-powered applications

---

# Project Overview

Large Language Models generate engaging narratives but struggle to maintain consistency over long conversations because they have a limited context window.

This project addresses that limitation by separating **memory storage** from **memory retrieval**. Important events are stored outside the model, and only the most relevant memories are retrieved and injected into the prompt before each response.

The architecture allows multiple retrieval strategies to operate on the same persistent memory, making it easy to compare their effectiveness without changing the underlying language model.

---

# Objectives

- Build an AI-powered Dungeon Master for interactive storytelling.
- Explore different approaches to long-term memory for LLMs.
- Compare keyword-based retrieval with semantic retrieval (RAG).
- Improve story continuity during extended conversations.
- Demonstrate practical applications of Retrieval-Augmented Generation.

---

# System Architecture

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
 (Recent Context)                   (Persistent JSON Store)
          │                                       │
          ├──────────────────────┬────────────────┤
          ▼                      ▼
 Lightweight Retrieval     Semantic Retrieval
 (Keyword + Priority)      (MiniLM + FAISS)
          │                      │
          └──────────────┬───────┘
                         ▼
                 Retrieved Memories
                         │
                         ▼
                  Prompt Construction
                         │
                         ▼
                  Groq / OpenAI API
                         │
                         ▼
               Dungeon Master Response
                         │
                         ▼
              Automatic Event Extraction
                         │
                         ▼
                Update Memory Store
```

---

# Workflow

```text
Player Input
      │
      ▼
Retrieve Relevant Memories
      │
      ▼
Build Context-Aware Prompt
      │
      ▼
Generate Response using LLM
      │
      ▼
Extract Important Story Events
      │
      ▼
Store in Long-Term Memory
      │
      ▼
Available for Future Retrieval
```

Every interaction updates the memory store, allowing the world to evolve while maintaining consistency across long-running adventures.

---

# Core Features

## Interactive Storytelling

- AI-generated RPG adventures
- Dynamic world generation
- Player-driven narrative progression
- Persistent world state

---

## Dual Memory Retrieval

### Lightweight Retrieval

A fast lexical retrieval strategy based on:

- Keyword matching
- Priority ranking
- NPC matching
- Recency scoring

Suitable for smaller memory stores with minimal computational overhead.

### Semantic Retrieval (RAG)

Embedding-based retrieval powered by:

- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS vector similarity search

Retrieves memories based on semantic similarity rather than exact keyword matches, improving contextual recall in longer adventures.

---

## Persistent Long-Term Memory

Important story events are automatically extracted and stored, including:

- Characters
- Quests
- Locations
- Items
- Major player decisions

Players can also manually pin memories using:

```text
remember: The silver key opens the northern vault.
```

Pinned memories receive the highest retrieval priority.

---

## Reliability Features

- API retry mechanism
- Configurable response length
- Prompt size management
- Memory deduplication
- Automatic consistency hints

---
---

# Memory Retrieval Comparison

The project supports two interchangeable retrieval strategies operating on the same persistent memory store.

| Feature | Lightweight Retrieval | Semantic Retrieval (RAG) |
|---------|-----------------------|--------------------------|
| Retrieval Method | Keyword Matching + Priority Ranking | Embedding Similarity Search |
| Semantic Understanding | No | Yes |
| Vector Search | No | FAISS |
| Computational Cost | Low | Moderate |
| Retrieval Speed | Very Fast | Fast |
| Best Use Case | Small Memory Stores | Long-Term Story Consistency |

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3 |
| Frontend | Streamlit |
| LLM Providers | Groq API, OpenAI API |
| Memory Storage | JSON |
| Semantic Search | Sentence Transformers (MiniLM) |
| Vector Search | FAISS |
| Environment | python-dotenv |
| Networking | Requests |

---

# Project Structure

```text
AI-Dungeon-Master/
│
├── app.py                 # Streamlit interface
├── dm_engine.py           # LLM communication layer
├── memory.py              # Memory management & retrieval
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

# Installation

Clone the repository

```bash
git clone https://github.com/aditi-malav/Ai-Dungeon-Master-Bootcamp-Project.git

cd Ai-Dungeon-Master-Bootcamp-Project
```

Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a `.env` file in the project root.

```env
PROVIDER=groq

MODEL_NAME=llama-3.1-8b-instant

GROQ_API_KEY=your_api_key

OPENAI_API_KEY=your_api_key

USE_SEMANTIC_RAG=1

MAX_TOKENS=350

MAX_NOTES=18
```

| Variable | Description |
|----------|-------------|
| `PROVIDER` | LLM provider (`groq` or `openai`) |
| `MODEL_NAME` | Model used for response generation |
| `USE_SEMANTIC_RAG` | Enables Semantic Retrieval |
| `MAX_TOKENS` | Maximum response length |
| `MAX_NOTES` | Maximum memories retrieved |

---

# Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser.

---

# Usage

### Select a Retrieval Mode

- Lightweight Retrieval
- Semantic Retrieval (RAG)

### Start an Adventure

Describe your actions naturally.

```text
Enter the abandoned castle.
```

```text
Search the library for clues.
```

```text
Talk to the old wizard.
```

### Save Important Memories

Store information that should persist throughout the adventure.

```text
remember: The silver key opens the northern vault.
```

Pinned memories are given the highest retrieval priority.

---

# Demo

## Live Application

https://ai-dungeon-master-bootcamp-project-sqtqtfhggqmpqsecogwbvh.streamlit.app/

## Demo Video

https://streamable.com/ve9pnd

---
---

# Design Decisions

### Separate Memory Storage from Retrieval

The system decouples memory storage from retrieval, allowing multiple retrieval strategies to operate on the same persistent memory without changing the underlying architecture.

### Dual Retrieval Modes

Both keyword-based and semantic retrieval are implemented to compare their trade-offs in speed, computational cost, and contextual recall.

### External Memory over Fine-Tuning

Instead of modifying the LLM, relevant memories are dynamically retrieved and injected into the prompt. This makes the system model-agnostic and easy to extend.

### Modular Architecture

The application is divided into independent components for the UI, memory management, prompt construction, and LLM communication, improving maintainability and experimentation.

---

# Current Limitations

- Event extraction relies on lightweight heuristics.
- Long-term memories are stored as plain text instead of structured entities.
- The FAISS index is rebuilt when Semantic Retrieval is initialized.
- Memories are stored locally using JSON, making the current implementation suitable for a single user.
- Retrieval strategies operate independently rather than as a hybrid system.

---

# Future Improvements

## Hybrid Retrieval

Combine keyword-based and semantic retrieval to improve recall while preserving precision.

```text
          Player Query
               │
        ┌──────┴──────┐
        ▼             ▼
 Keyword Search   Semantic Search
        │             │
        └──────┬──────┘
               ▼
       Merge & Re-rank Results
               ▼
      Context-Aware Prompt
```

---

## Structured Memory

Store memories as structured entities instead of plain text.

Example:

```json
{
  "type": "quest",
  "name": "Retrieve the Silver Key",
  "location": "Ancient Castle",
  "importance": 5
}
```

---

## Persistent Vector Database

Replace the in-memory FAISS index with a production-ready vector database such as:

- ChromaDB
- Qdrant
- Pinecone

---

## Smarter Memory Extraction

Use an LLM to automatically identify and categorize important story events instead of relying on heuristic rules.

---

## Session-Based Memory

Support multiple users and multiple adventures by maintaining isolated memory stores for each session.

---

## Character Knowledge Graph

Represent characters, quests, locations, and relationships as a knowledge graph to improve world consistency and contextual reasoning.

---

# Learning Outcomes

This project provided hands-on experience with:

- Designing memory architectures for LLM applications
- Retrieval-Augmented Generation (RAG)
- Semantic search using Sentence Transformers
- Vector similarity search with FAISS
- Prompt engineering for context-aware generation
- Managing short-term and long-term conversational memory
- Integrating Groq and OpenAI APIs
- Building and deploying AI applications with Streamlit
- Evaluating trade-offs between lexical and semantic retrieval

---

# Potential Applications

Although developed as an AI Dungeon Master, the same architecture can be adapted for:

- AI personal assistants with persistent memory
- Customer support chatbots
- Educational tutors
- Interactive role-playing agents
- Multi-session conversational AI
- Knowledge management systems

---







