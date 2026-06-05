# AI Dungeon Master

A text-based adventure game where a Large Language Model acts as the Dungeon Master and generates story responses based on player actions.

The project explores different approaches to giving an LLM memory so it can remember important events and maintain better story continuity during longer conversations.

## Live Demo

### Application

https://ai-dungeon-master-bootcamp-project-sqtqtfhggqmpqsecogwbvh.streamlit.app/

### Demo Video

https://streamable.com/ve9pnd

---

## Features

### Interactive Storytelling

- Player-driven text adventure
- AI-generated story responses
- Streamlit-based web interface

### Dual Memory Modes

#### Lightweight Memory

- Stores recent conversation history
- Retrieves important notes using keyword matching and priority scores
- Fast and lightweight

#### Semantic Memory

- Stores important story events as long-term notes
- Uses Sentence Transformers to generate embeddings
- Uses FAISS for similarity search
- Retrieves memories related to the current player action

### Long-Term Notes

Important story events are stored and reused later in the adventure.

Examples include:

- Discovering items
- Receiving quests
- Finding locations
- Meeting important characters

Players can also manually save information:

```text
remember: The silver key opens the northern vault.
```

### Event Extraction

After each AI response, the system extracts notable events using simple keyword-based rules and stores them as long-term memory.

### Reliability Features

- Retry handling for API rate limits
- Prompt size control
- Memory cleanup and deduplication

---

## How It Works

1. The player enters an action.
2. Recent conversation history is collected.
3. Relevant long-term memories are retrieved.
4. A prompt is built using:
   - System instructions
   - Recent conversation
   - Retrieved memories
   - Consistency hints
5. The prompt is sent to the language model.
6. The generated response is shown to the player.
7. Important events are extracted and stored for future use.

### System Flow

```text
Player Input
      ↓
Memory Retrieval
      ↓
Prompt Construction
      ↓
LLM Response
      ↓
Event Extraction
      ↓
Memory Update
```

---

## Technology Stack

### Backend

- Python

### User Interface

- Streamlit

### Language Models

- Groq API
- OpenAI API

### Semantic Retrieval

- Sentence Transformers
- FAISS

### Concepts Used

- Conversational Memory
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering

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

### Short-Term Memory

Recent conversation turns are stored and included in future prompts.

### Long-Term Memory

Important events are saved as notes and can be retrieved later.

### Pinned Notes

Players can manually save information:

```text
remember: The village elder owes me a favor.
```

### Semantic Retrieval

When Semantic Memory mode is enabled:

1. Notes are converted into embeddings.
2. Embeddings are stored in a FAISS index.
3. Relevant notes are retrieved using similarity search.
4. Retrieved notes are added to future prompts.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/aditi-malav/Ai-Dungeon-Master-Bootcamp-Project.git
cd Ai-Dungeon-Master-Bootcamp-Project
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create a .env File

```env
PROVIDER=groq
MODEL_NAME=llama-3.1-8b-instant

GROQ_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key

USE_SEMANTIC_RAG=1
MAX_TOKENS=350
MAX_NOTES=18
```

### Run the Application

```bash
streamlit run app.py
```

---

## Future Improvements

- Better event extraction using LLMs
- More structured character tracking
- Persistent vector database
- FastAPI backend
- React frontend
- Multiplayer support
- Voice interactions

---

## Learning Outcomes

Through this project, I explored:

- Building applications with LLM APIs
- Memory management for conversational systems
- Semantic search using embeddings
- Retrieval-Augmented Generation (RAG)
- Prompt construction and context management
- Streamlit application development
- Long-context interaction design


