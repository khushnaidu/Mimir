# Mimir Chrome Extension

This document provides detailed information about the Mimir Chrome extension, which enables users to analyze political content directly from any webpage.

## Overview

The Mimir Chrome extension allows users to:

1. Select any political text on a webpage
2. Analyze it using the Mimir political analysis backend
3. View balanced political perspectives and relevant context
4. Explore related news articles and Reddit discussions

## Installation

### Loading the Unpacked Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" using the toggle in the top-right corner
3. Click "Load unpacked"
4. Navigate to and select the `Mimir/chrome-extension` directory
5. The Mimir extension icon should appear in your browser toolbar

### Alternative: Manual Installation

If you received the extension as a .crx file:

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" using the toggle in the top-right corner
3. Drag and drop the .crx file onto the extensions page
4. Click "Add extension" when prompted

## Extension Components

The Mimir Chrome extension consists of the following files:

### 1. `manifest.json`

This file defines the extension's metadata, permissions, and components:

```json
{
  "manifest_version": 3,
  "name": "Mimir Political Analysis",
  "version": "1.0",
  "description": "Analyze political content with balanced perspectives",
  "permissions": ["activeTab", "contextMenus", "storage"],
  "action": {
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon16.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content_script.js"]
    }
  ],
  "background": {
    "service_worker": "background.js"
  }
}
```

### 2. `popup.html` and `popup.js`

These files define the extension's user interface:

- Text input area for manual entry
- Model selection dropdown
- Analysis results display
- News and Reddit tab navigation
- Loading indicators and error messages

### 3. `content_script.js`

This script runs in the context of web pages to:

- Capture selected text
- Communicate with the popup and background script
- Handle text highlighting and context menu interactions

### 4. `background.js`

The background script:

- Creates and manages context menu items
- Handles communication between content script and popup
- Manages API requests to the Mimir backend
- Stores user preferences and session data

## Using the Extension

### Method 1: Context Menu

1. Select text on any webpage
2. Right-click the selected text
3. Choose "Analyze with Mimir" from the context menu
4. A popup will appear with the analysis results

### Method 2: Extension Popup

1. Click the Mimir icon in the Chrome toolbar
2. Either:
   - Paste text into the input field, or
   - Click "Analyze current selection" if text is selected on the page
3. Choose a model from the dropdown (optional)
4. Click "Analyze"
5. View the results in the popup

## Extension Features

### Model Selection

The extension allows selection between different LLM options:

- GPT-3.5-Turbo: Fast commercial model (default)
- GPT-4: Advanced commercial model with enhanced reasoning
- TinyLlama: Open-source model optimized for political analysis

### Results Display

Analysis results are displayed in several sections:

1. **Summary**: A balanced overview of the political topic
2. **News Context**: Relevant news articles related to the topic
3. **Reddit Discussions**: Related political discussions from Reddit
4. **Performance Metrics**: Response time and model information

### User Feedback

Users can provide feedback on analysis quality:

- 1-5 star rating system
- Optional comments field
- Feedback is sent to the backend for model evaluation

## Configuration

### Backend URL

By default, the extension connects to `http://localhost:5000`. To change this:

1. Open `popup.js`
2. Find the `API_BASE_URL` constant
3. Change it to your desired backend URL
4. Reload the extension

### Styling Customization

To customize the extension's appearance:

1. Edit `popup.css` to modify colors, fonts, and layout
2. Adjust the responsive design parameters for different screen sizes

## Troubleshooting

### Extension Not Working

1. Ensure the Mimir backend is running
2. Check that the backend URL is correctly configured
3. Verify that your browser has internet access
4. Look for errors in the browser console (Right-click > Inspect > Console)

### Slow Analysis

1. Try using a faster model (e.g., GPT-3.5-Turbo instead of TinyLlama)
2. Reduce the length of text being analyzed
3. Check if your backend server has sufficient resources

### Missing Icons or UI Elements

1. Verify the extension was loaded correctly
2. Try reloading the extension or restarting Chrome
3. Check if all files in the extension directory are present

## Development and Customization

### Adding New Features

To add new features to the extension:

1. Update `content_script.js` for page interaction features
2. Modify `popup.js` and `popup.html` for UI changes
3. Extend `background.js` for new background functionality

### Debugging

For development and debugging:

1. Access the background script console:
   - Go to `chrome://extensions/`
   - Find the Mimir extension
   - Click "background page" under "Inspect views"

2. Debug the popup:
   - Right-click the extension icon
   - Select "Inspect Popup"

3. Monitor content script:
   - Right-click on any webpage
   - Select "Inspect"
   - Navigate to the Console tab 