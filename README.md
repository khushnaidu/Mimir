# Mimir: Political Analysis Tool

Mimir is an advanced political analysis tool that combines retrieval-augmented generation (RAG) with lightweight open-source models to provide nuanced political insights. The system retrieves relevant political discourse from Reddit and recent news articles, then generates balanced analysis across the political spectrum.

## Features

- Semantic search of political discussions using vector embeddings
- News API integration for current events context
- Multiple LLM support (OpenAI models and open-source TinyLlama)
- Parameter-efficient fine-tuning with LoRA
- Chrome extension for analyzing political text on any webpage
- Political analysis capabilities including:
  - Policy difference analysis
  - Bipartisan opportunity identification
  - Political speech/statement analysis
  - Policy keyword extraction

## Core NLP Tasks

Mimir performs three specialized LLM tasks that form the backbone of its analysis pipeline:

1. **Query Reformulation**: Transforms user input into optimized semantic search queries for Reddit retrieval
2. **News Query Extraction**: Extracts focused keywords to find relevant articles via News API
3. **Context Summarization**: Synthesizes Reddit posts and news articles into balanced political analysis

For detailed information on these tasks, see [Core NLP Tasks Documentation](docs/core_nlp_tasks.md).

## Architecture Overview

```
                           ┌─────────────────┐
                           │ Chrome Extension│
                           └────────┬────────┘
                                    │
                                    ▼
┌──────────────┐            ┌───────────────┐           ┌───────────────┐
│ Vector Store  │◄──────────┤  Flask API    ├──────────►│  News API     │
└──────┬───────┘            └───────┬───────┘           └───────────────┘
       │                            │
       │                            ▼
       │                    ┌───────────────┐
       └──────────────────► │  LLM Pipeline │
                            └───────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Flask
- Chrome browser (for extension)
- NewsAPI key
- OpenAI API key (optional, for commercial models)
- Hugging Face token (optional, for downloading models)

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/khushnaidu/Mimir.git
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEWSAPI_KEY=your_newsapi_key
   HF_TOKEN=your_huggingface_token
   ```

5. Run the Flask application:
   ```
   FLASK_APP=app.main flask run
   ```
   
   The server will start at `http://127.0.0.1:5000/`

### Data & Models

All processed data and model weights are included in the repo under `app/data/`.  
- **Reddit Posts with Embeddings**: `app/data/reddit_posts_with_embeddings.json` (+ metadata JSON)


### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in the top-right corner)
3. Click "Load unpacked" and select the `chrome-extension` directory from the repository
4. The Mimir extension icon should appear in your browser toolbar

## Detailed Component Explanations

### 1. Data Acquisition

Mimir utilizes data from two primary sources:

- **Reddit political discussions**: Pre-processed political discussions from various subreddits
- **News API**: Real-time news articles related to political topics

The Reddit data is preprocessed and embedded for efficient retrieval, while news data is fetched in real-time based on extracted keywords.

### 2. Vector Store (`app/utils/vector_store.py`)

The vector store manages embeddings for political discussions, enabling semantic search:

- Uses Sentence Transformers (all-MiniLM-L6-v2) for embedding generation
- Stores embeddings as NumPy arrays and metadata as JSON
- Provides methods for similarity search and document addition
- Supports rich metadata with post content, author, and comments

### 3. LLM Pipeline (`app/utils/llm_pipeline.py`)

The LLM pipeline manages interactions with various language models:

- Supports both commercial (OpenAI) and open-source models (TinyLlama)
- Implements parameter-efficient fine-tuning with LoRA
- Provides specialized prompts for political analysis
- Features task-specific optimizations for:
  - Query reformatting (improving semantic search)
  - News query extraction (generating keywords for news search)
  - Context summarization (synthesizing multiple perspectives)

### 4. Political Analysis (`app/utils/political_analysis.py`)

Specialized module for political text analysis:

- Policy differences analysis between political parties
- Identification of bipartisan opportunities
- Political speech/statement analysis
- Policy keyword extraction
- Balanced presentation of multiple political perspectives

### 5. RAG Pipeline (`app/utils/rag_pipeline.py`)

Combines retrieval and generation for enhanced political analysis:

- Reformats user queries for optimal semantic search
- Retrieves relevant political discussions from vector store
- Extracts optimal keywords for news search
- Combines retrieved context with LLM capabilities for comprehensive analysis

### 6. News API Integration (`app/utils/news_api.py`)

Provides real-time news context:

- Asynchronous client for NewsAPI
- Searches multiple queries in parallel
- Filters and ranks results by relevance
- Integrates news content with political discussions for comprehensive analysis

### 7. Flask API (`app/main.py`)

Central backend that coordinates all components:

- Exposes endpoints for text analysis
- Manages model selection and configuration
- Processes user queries through the RAG pipeline
- Collects user feedback for model evaluation
- Generates performance reports and comparisons

### 8. Chrome Extension

Browser extension for analyzing political content on any webpage:

- `popup.html` & `popup.js`: User interface for text selection and analysis
- `content_script.js`: Handles webpage interaction
- `background.js`: Manages extension lifecycle
- Communicates with Flask backend for analysis processing

## Open-Source Model Details

### TinyLlama with LoRA Optimization

Mimir implements Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) for optimizing TinyLlama for political analysis:

- **Base model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Adaptation technique**: LoRA (Low-Rank Adaptation)
- **Configuration**:
  - Target modules: Query and Value projections (`q_proj`, `v_proj`)
  - Rank (r): 8
  - Alpha: 16
  - Task type: Causal Language Modeling

This approach:
- Freezes the pre-trained model weights
- Adds small, trainable rank decomposition matrices
- Reduces trainable parameters from millions to thousands
- Enables efficient adaptation without catastrophic forgetting

## Performance and Evaluation

Mimir includes an evaluation framework that compares model performance across several dimensions:

- **Latency**: Response time for various analysis tasks
- **Token efficiency**: Number of tokens used per request
- **Output quality**: Assessed through user feedback (currently not working on UI end, but can submit feedback through curl command)
- **Model comparison**: Relative performance of different models on identical tasks

The evaluation data is collected during normal operation and can be analyzed through the `/evaluation/reports` endpoint.

## Usage Examples

### API Usage

```python
import requests

# Analyze political text
response = requests.post('http://localhost:5000/analyze', json={
    'text': 'What are the key policy differences between Democrats and Republicans on healthcare?',
    'model': 'tinyllama-1.1b'
})

result = response.json()
print(result['summary'])
```

### Chrome Extension

1. Navigate to a webpage with political content
2. Select text by highlighting it
3. Right-click and select "Analyze with Mimir" or click the Mimir extension icon
4. View the balanced analysis in the popup window

## Documentation

- [Installation Guide](docs/installation.md)
- [API Documentation](docs/api.md)
- [Chrome Extension Guide](docs/chrome_extension.md)
- [Model Optimization](docs/model_optimization.md)
- [Core NLP Tasks](docs/core_nlp_tasks.md)

## Acknowledgments

- TinyLlama team for the base model
- Hugging Face for model hosting and transformers library
- Reddit for the political discussion data
- NewsAPI for real-time news access

## Contributors

- Khush Naidu     (khush.naidu@gmail.com)  
- Sai Prajwal Kongalla  (saiprajwal.kongalla@gmail.com)  
- Varsha Chamakura    (varsha.chamakura@gmail.com)  

## Major Contributions

- **Khush Naidu**  
  - Flask API setup & endpoint definitions (`app/main.py`)  
  - Data acquisition pipeline (`app/utils/data_acquisition.py`)  
  - RAG pipeline (embedding generation, similarity search) (`app/utils/rag_pipeline.py`)  
  - LLM pipeline tasks: query reformulation, keyword extraction, summarization (`app/utils/llm_pipeline.py`)  
  - LoRA fine-tuning and TinyLlama integration (`app/utils/llm_pipeline.py`)  
  - Task-specific parameter optimization  
  - Pipeline and model testing & evaluation scripts (`app/utils/evaluate.py`)  (
  - Documentation (this README + docs/)  

- **Sai Prajwal Kongalla**  
  - Chrome extension development (UI, popup.html/js, content_script.js, background.js) (`chrome-extension/`)  
  - NewsAPI integration module (`app/utils/news_api.py`)  
  - Vector store implementation (`app/utils/vector_store.py`)  
  - Secure communication between extension and Flask backend  

- **Varsha Chamakura**  
  - Project documentation and report write-up (`docs/`)  
  - Presentation slides and demo video 
  - Frontend UI enhancements for Chrome extension  
  - Testing (end-to-end and unit tests)  

---

