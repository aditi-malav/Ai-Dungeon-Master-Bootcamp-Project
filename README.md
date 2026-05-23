# AI Dungeon Master

Stateful AI-powered interactive storytelling engine built using Large Language Models with dynamic prompt orchestration, dual-mode memory systems, and semantic retrieval pipelines.

## Overview

AI Dungeon Master is a stateful conversational AI system where Large Language Models act as an intelligent Dungeon Master capable of maintaining narrative continuity, contextual memory, and immersive RPG-style interactions.

The project combines:
- Dual-mode AI memory systems
- Semantic RAG-based retrieval
- Stateful session orchestration
- Dynamic prompt engineering
- Context-aware memory injection
- Multi-provider LLM inference

The architecture focuses on solving key challenges in conversational AI systems such as:
- Context loss
- Hallucinations
- Narrative inconsistency
- Token optimization
- Stateful long-term interactions

---

## Live Application

App Link  
https://ai-dungeon-master-bootcamp-project-sqtqtfhggqmpqsecogwbvh.streamlit.app/

Demo Video  
https://streamable.com/ve9pnd

---

## Key Features

### Dual-mode AI Memory System
- Lightweight short-term conversational memory
- Semantic RAG-based long-term retrieval system

### Stateful Narrative Consistency
- Persistent world-state tracking
- Context-aware memory injection
- NPC and event continuity handling

### Multi-provider LLM Support
- Groq API integration
- OpenAI API integration

### Dynamic Prompt Orchestration
- Structured context assembly
- Retrieval-aware prompt construction
- Context compression for token efficiency

### Creative Memory Features
- Persistent note pinning using:
```text
remember: <note>
```

- Automatic key-event extraction
- Consistency-aware recall hints
- Context-focused retrieval scheduling

### Reliability Engineering
- Retry handling for API rate limits
- Fault-tolerant inference pipeline
- Token-efficient context trimming

---

## System Architecture

```text
Player Input
      ↓
Short-Term Memory Retrieval
      ↓
Semantic RAG Recall
      ↓
Prompt Construction Pipeline
      ↓
LLM Inference
      ↓
Narrative Generation
      ↓
Consistency Validation
      ↓
Memory Update & Event Extraction
```

---

## Tech Stack

### Languages & Frameworks
- Python
- Streamlit

### AI & NLP
- OpenAI API
- Groq API
- Prompt Engineering
- Semantic Retrieval
- Retrieval-Augmented Generation (RAG)

### Backend Concepts
- Stateful Session Management
- Dynamic Prompt Pipelines
- Context Compression
- Retrieval Systems
- Fault-Tolerant API Handling
- Context-Aware Memory Recall

---

## Project Structure

```bash
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

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
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

## Run Application

```bash
streamlit run app.py
```

---

## Example Gameplay

```text
Player:
"I cautiously enter the abandoned castle."

AI Dungeon Master:
"The rusted iron gates creak shut behind you as torchlight flickers across damp stone walls..."
```

---

## Memory System

### Lightweight Memory Mode
- Fast conversational recall
- Token-efficient short-term context

### Semantic RAG Mode
- Context-aware retrieval
- Long-term narrative consistency
- Semantic memory search

---

## Persistent Memory Notes

Players can permanently store important information:

```text
remember: The silver key opens the northern vault.
```

Pinned notes are automatically injected into future prompts to preserve continuity.

---

## Reliability & Optimization Features

- Automatic retry handling for API rate limits
- Dynamic retrieval scheduling
- Token-efficient context management
- Multi-provider inference abstraction
- Consistency-aware memory hints

---

## Future Improvements

- React frontend
- FastAPI backend
- Vector database integration
- Multiplayer support
- Voice-enabled gameplay
- LangGraph workflows


---

## Learning Outcomes

This project explores:
- AI Agent Architectures
- Retrieval-Augmented Generation
- Prompt Orchestration
- Stateful Conversational AI
- Semantic Retrieval Systems
- Context Management
- AI Workflow Engineering


