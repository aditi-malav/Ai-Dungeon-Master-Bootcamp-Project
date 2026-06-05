# AI Dungeon Master

Stateful interactive storytelling platform built using Large Language Models, semantic retrieval, and memory-aware prompt construction.

## Overview

AI Dungeon Master is a conversational storytelling system where a Large Language Model acts as a Dungeon Master and generates RPG-style adventures based on player actions.

The project explores techniques for maintaining narrative continuity across long conversations through a combination of short-term conversational memory and optional semantic retrieval.

Core capabilities include:

- Stateful narrative generation
- Dual-mode memory management
- Semantic memory retrieval using FAISS
- Retrieval-aware prompt construction
- Persistent note pinning
- Automatic key-event extraction
- Rate-limit-aware inference handling

---

## Features

### Dual Memory Modes

#### Lightweight Memory

- Sliding-window conversational memory
- Fast context retrieval
- Low token usage

#### Semantic RAG Memory

- Long-term memory storage
- Semantic retrieval using embeddings
- Context-aware recall of relevant events

### Narrative Continuity

- Persistent world-state tracking
- Event-based memory extraction
- Memory pinning through player notes
- Consistency reminder system

### LLM Integration

Supports either:

- Groq API
- OpenAI API

through a configurable inference backend.

### Prompt Construction Pipeline

Player input is combined with:

- Recent conversation history
- Retrieved long-term memories
- Consistency hints
- Pinned notes

before being sent to the language model.

### Reliability Features

- Automatic retry handling for API rate limits
- Context trimming to control prompt size
- Memory compaction to prevent unbounded growth

---

## System Architecture

```text
Player Input
      ↓
Short-Term Memory
      ↓
(Optional) Semantic Retrieval
      ↓
Prompt Construction
      ↓
LLM Inference
      ↓
Story Generation
      ↓
Memory Update
      ↓
Event Extraction
```

---

## Technology Stack

### Languages & Frameworks

- Python
- Streamlit

### AI & Retrieval

- Groq API
- OpenAI API
- Sentence Transformers
- FAISS

### Concepts

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Conversational Memory
- Prompt Engineering
- Stateful Session Management

---

## Project Structure

```text
AI-Dungeon-Master/
│
├── app.py
├── dm_engine.py
├── memory.py
├── prompts.py
├── requirements.txt
├── .env
└── README.md
```

---

## Memory System

### Persistent Notes

Players can store important information:

```text
remember: The silver key opens the northern vault.
```

Pinned notes are retained across future interactions.

### Event Extraction

The system automatically extracts notable events from generated responses and stores them as long-term memory entries.

### Semantic Retrieval

When Semantic RAG mode is enabled, stored memories are embedded using Sentence Transformers and indexed with FAISS for similarity-based retrieval.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/aditi-malav/Ai-Dungeon-Master-Bootcamp-Project.git
cd Ai-Dungeon-Master-Bootcamp-Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

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

## Run the Application

```bash
streamlit run app.py
```

---

## Live Demo

**Application**  
https://ai-dungeon-master-bootcamp-project-sqtqtfhggqmpqsecogwbvh.streamlit.app/

**Demo Video**  
https://streamable.com/ve9pnd

---

## Future Improvements

- FastAPI backend
- React frontend
- Persistent vector database
- Multiplayer support
- Voice interactions
- Richer memory extraction strategies

---

## Learning Outcomes

This project explores:

- Stateful conversational systems
- Retrieval-Augmented Generation
- Semantic search
- Prompt construction
- Memory management for LLM applications
- Long-context interaction design


