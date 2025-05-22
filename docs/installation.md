# Mimir Installation Guide

This document provides detailed installation instructions for setting up the Mimir political analysis tool.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.8+ installed
- pip package manager
- Git
- Chrome browser (for extension)
- API keys:
  - NewsAPI key (required): [Get one here](https://newsapi.org/register)
  - OpenAI API key (optional for commercial models): [Get one here](https://platform.openai.com/signup)
  - Hugging Face token (optional for downloading models): [Get one here](https://huggingface.co/join)

## Step 1: Clone the Repository

```bash
git clone https://github.com/khushnaidu/Mimir.git
cd Mimir
```

## Step 2: Set Up Python Environment

### Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
# venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter any errors, you may need to install some dependencies individually:

```bash
pip install flask flask-cors openai python-dotenv sentence-transformers
pip install torch transformers peft
```

## Step 3: Configure Environment Variables

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key
HF_TOKEN=your_huggingface_token
```

This file should be kept secure and never committed to version control.

## Step 4: Prepare Data Directory

Ensure the data directory exists for storing embeddings and metadata:

```bash
mkdir -p Mimir/app/data
```

## Step 5: Run the Flask Application

Start the Flask application using:

```bash
cd Mimir
FLASK_APP=app.main flask run
```

For development with automatic reloading:

```bash
FLASK_APP=app.main FLASK_ENV=development flask run
```

The server will start at `http://127.0.0.1:5000/`

## Step 6: Install Chrome Extension

1. Open Chrome browser
2. Navigate to `chrome://extensions/`
3. Enable "Developer mode" using the toggle in the top-right corner
4. Click "Load unpacked"
5. Navigate to and select the `Mimir/chrome-extension` directory
6. The Mimir extension icon should appear in your browser toolbar

## Troubleshooting

### Model Loading Issues

If you encounter issues loading the TinyLlama model:

```bash
# Make sure the models directory exists
mkdir -p Mimir/app/models/tinyllama-1.1b

# The app will automatically download the model when first needed,
# or you can manually download from Hugging Face
```

### API Connection Errors

If you're getting API connection errors:

1. Verify your API keys in the `.env` file
2. Check your internet connection
3. Ensure the API services (OpenAI, NewsAPI) are operational

### Extension Not Working

If the Chrome extension isn't working properly:

1. Check the developer console for errors
2. Ensure the Flask backend is running
3. Verify the extension is pointed to the correct API endpoint (should be `http://localhost:5000/analyze` by default)


## Next Steps

Once installation is complete, you can:

1. Test the API using the curl command or Postman
2. Use the Chrome extension to analyze political text on websites
3. Check out the `/models` endpoint to see available language models
4. Explore the `/analyze` endpoint for text analysis

For more information, see the [main README.md](../README.md) and [API documentation](api.md). 
