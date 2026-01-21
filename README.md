# Chatbot

- Chatbot with document retrieval and web search function.

## Features

- **Chitchat**: Chat.
- **Document Retrieval**: LLM perform document searches based on user queries. If central subject or key entity is found (e.g., brand names, product names), will auto search for other nicknames of these entities, and to be used as filters to increase retrieval precision.
- **Web Search**: Perform web search using Tavily based on user queries.

*** Only these three intents will be recognized.

## Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API keys (Default Model Spec: GPT-4o)
- Tavily API keys

## Setup

### Local Setup

1. Clone the repository:
```bash
git clone 
cd Chatbot_001
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys. For other variables amendment, refers to config/settings.py
```

5. Run the application:
```bash
streamlit run main.py
```

### Docker Setup

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
`````

2. Build and run with Docker Compose:
```bash
docker-compose run --rm chatbot-app

or

docker-compose up --build
```

## Main Application Usage

1. Amend env file accordingly, especially the INPUT_FILENAME which refers to the filename of raw_data.
2. Faiss setup: place raw_data in **input** directory. Then, bash run "python vector_index_init.py" to setup faiss index.
3. Bash run "streamlit run main.py"


## Module Workflow

```
**Flow**
┌─────────────┐
│   Query     │──► User Query
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Intention  │──► Detect intention: Chitchat, Doc Retrieval or Web Search
└──────┬──────┘    
       │
       ▼
┌─────────────┐
│ Route & Exec│──► Route to respective module, and execute the module 
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Output   │──► LLM output based on the enhanced context
└─────────────┘
```


# File System Structure
```
project/
├── artifact/       # Generated outputs and intermediate artifacts (dataframes, plots, reports)
│   ├── faiss/      # Saved faiss index and data in pickle format
├── input/          # Input data files
│   ├── raw_data/   # **Raw Data**
├── logs/           # Logs
├── misc/           # Miscellaneous files
├── src/            # Source code
│   ├── agents/     # Primary Agents and Workflow Source Code
│   ├── assets/     # Static resources (prompts, templates)
│   ├── configs/    # Configuration files and environmental variables
│   ├── data/       # Faiss setup and load related files
│   └── tools/      # Utility functions
└── tests/          # Test suites ongoing (boilerplate code)
```
