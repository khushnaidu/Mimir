// Global variables
let availableModels = ["gpt-3.5-turbo", "gpt-4", "tinyllama-1.1b"];
let selectedModel = "gpt-3.5-turbo";      // Default model
let lastQueryId = null;                   // To store the query ID for feedback

document.addEventListener('DOMContentLoaded', function () {
    const analyzeButton = document.getElementById('analyzeButton');
    const textInput = document.getElementById('textInput');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultContainer = document.getElementById('resultContainer');
    const summaryContent = document.getElementById('summaryContent');
    const newsContent = document.getElementById('newsContent');
    const redditContent = document.getElementById('redditContent');

    // Handle tab switching
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const tabName = tab.getAttribute('data-tab');
            document.getElementById(`${tabName}Tab`).classList.add('active');
        });
    });

    // Handle analyze button click
    analyzeButton.addEventListener('click', () => {
        const text = textInput.value.trim();

        if (text === '') {
            alert('Please enter some text to analyze.');
            return;
        }

        // Show loading, hide results
        loadingIndicator.style.display = 'block';
        resultContainer.style.display = 'none';

        // Send data to the background script for API call
        chrome.runtime.sendMessage(
            {
                action: 'fetchAnalysis',
                text: text,
                model: selectedModel  // Include the selected model
            },
            (response) => {
                // Hide loading
                loadingIndicator.style.display = 'none';

                if (response && response.data) {
                    // Show results container
                    resultContainer.style.display = 'block';

                    // Display summary
                    summaryContent.innerHTML = `<p>${response.data.summary || 'No summary available.'}</p>`;

                    // Store query ID for feedback
                    lastQueryId = response.data.query_id;

                    // Display feedback container if query ID exists
                    if (lastQueryId) {
                        document.getElementById('feedbackContainer').style.display = 'block';
                    }

                    // Display metrics if available
                    if (response.data.performance_metrics) {
                        const metrics = response.data.performance_metrics;
                        const metricsContent = document.getElementById('metricsContent');
                        metricsContent.innerHTML = `
                            <div class="metrics-container">
                              <p><strong>Model used:</strong> ${response.data.model_used || selectedModel}</p>
                              <p><strong>Total processing time:</strong> ${metrics.total_process_time.toFixed(2)}s</p>
                              <p><strong>Query reformulation:</strong> ${metrics.reformatting_time.toFixed(2)}s</p>
                              <p><strong>News query extraction:</strong> ${metrics.news_query_time.toFixed(2)}s</p>
                              <p><strong>Context summarization:</strong> ${metrics.summarization_time.toFixed(2)}s</p>
                            </div>
                        `;
                    }

                    // Display news articles
                    if (response.data.raw_context && response.data.raw_context.news_articles && response.data.raw_context.news_articles.length > 0) {
                        let newsHtml = '';
                        response.data.raw_context.news_articles.forEach(article => {
                            newsHtml += `
                <div class="news-article">
                  <div class="article-title">${article.title}</div>
                  <div class="article-source">${article.source} - ${new Date(article.published_at).toLocaleDateString()}</div>
                  <div class="article-description">${article.description}</div>
                  ${article.url ? `<a href="${article.url}" target="_blank">Read more</a>` : ''}
                </div>
              `;
                        });
                        newsContent.innerHTML = newsHtml;
                    } else {
                        newsContent.innerHTML = '<p>No news articles found.</p>';
                    }

                    // Display Reddit posts
                    if (response.data.raw_context && response.data.raw_context.reddit_posts && response.data.raw_context.reddit_posts.length > 0) {
                        let redditHtml = '';
                        response.data.raw_context.reddit_posts.forEach(post => {
                            redditHtml += `
                <div class="news-article">
                  <div class="article-title">${post.title}</div>
                  <div class="article-source">r/${post.subreddit} - ${new Date(post.created_utc * 1000).toLocaleDateString()}</div>
                  <div class="article-description">${post.selftext || post.body || ''}</div>
                  ${post.url ? `<a href="${post.url}" target="_blank">View on Reddit</a>` : ''}
                </div>
              `;
                        });
                        redditContent.innerHTML = redditHtml;
                    } else {
                        redditContent.innerHTML = '<p>No Reddit posts found.</p>';
                    }
                } else {
                    // Show error
                    resultContainer.style.display = 'block';
                    summaryContent.innerHTML = '<p class="windsurf-error">Error fetching analysis.</p>';
                    newsContent.innerHTML = '';
                    redditContent.innerHTML = '';
                }
            }
        );
    });

    // Set up the model selector
    const modelSelector = document.getElementById('modelSelector');
    modelSelector.addEventListener('change', function () {
        selectedModel = this.value;
    });

    // Try to fetch available models from the server
    fetchAvailableModels();

    // Set up text input keypress listener for Enter key
    textInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeText();
        }
    });

    // Set up feedback buttons
    const feedbackButtons = document.querySelectorAll('.feedback-btn');
    feedbackButtons.forEach(button => {
        button.addEventListener('click', function () {
            const rating = parseInt(this.getAttribute('data-rating'));
            submitFeedback(rating);
        });
    });
});

// Fetch available models from the server
function fetchAvailableModels() {
    fetch('http://localhost:5000/models')
        .then(response => response.json())
        .then(data => {
            if (data.models && Array.isArray(data.models)) {
                availableModels = data.models;
                selectedModel = data.default_model || "gpt-3.5-turbo";
                updateModelSelector();
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            // Use default models if fetch fails
            updateModelSelector();
        });
}

// Update the model selector with available models
function updateModelSelector() {
    const modelSelector = document.getElementById('modelSelector');
    modelSelector.innerHTML = '';

    availableModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;

        // Add "(Open Source)" label for non-OpenAI models
        if (!model.startsWith('gpt-')) {
            option.textContent += ' (Open Source)';
        }

        if (model === selectedModel) {
            option.selected = true;
        }
        modelSelector.appendChild(option);
    });
}

// Analyze text with the selected model
function analyzeText() {
    const textInput = document.getElementById('textInput').value.trim();
    if (!textInput) {
        alert('Please enter some text to analyze.');
        return;
    }

    // Show loading indicator and hide results
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('feedbackContainer').style.display = 'none';

    // API request to analyze text
    fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: textInput,
            model: selectedModel,
            collect_feedback: true
        })
    })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            document.getElementById('loadingIndicator').style.display = 'none';

            // Check for errors
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Store query ID for feedback
            lastQueryId = data.query_id;

            // Show result container
            document.getElementById('resultContainer').style.display = 'block';

            // Populate summary tab
            document.getElementById('summaryContent').textContent = data.summary;

            // Populate news tab
            const newsContent = document.getElementById('newsContent');
            newsContent.innerHTML = '';

            const newsArticles = data.raw_context.news_articles || [];
            if (newsArticles.length > 0) {
                newsArticles.forEach(article => {
                    const articleEl = document.createElement('div');
                    articleEl.className = 'news-article';

                    const titleEl = document.createElement('div');
                    titleEl.className = 'article-title';
                    titleEl.textContent = article.title || 'Untitled';
                    articleEl.appendChild(titleEl);

                    const sourceEl = document.createElement('div');
                    sourceEl.className = 'article-source';
                    sourceEl.textContent = `${article.source?.name || 'Source'} - ${formatDate(article.publishedAt)}`;
                    articleEl.appendChild(sourceEl);

                    const descEl = document.createElement('div');
                    descEl.className = 'article-description';
                    descEl.textContent = article.description || article.content || 'No description available';
                    articleEl.appendChild(descEl);

                    if (article.url) {
                        const linkEl = document.createElement('a');
                        linkEl.href = article.url;
                        linkEl.target = '_blank';
                        linkEl.textContent = 'Read more';
                        articleEl.appendChild(linkEl);
                    }

                    newsContent.appendChild(articleEl);
                });
            } else {
                newsContent.textContent = 'No news articles found.';
            }

            // Populate Reddit tab
            const redditContent = document.getElementById('redditContent');
            redditContent.innerHTML = '';

            const redditPosts = data.raw_context.reddit_posts || [];
            if (redditPosts.length > 0) {
                redditPosts.forEach(post => {
                    const postEl = document.createElement('div');
                    postEl.className = 'news-article';  // Reuse the same styling

                    const titleEl = document.createElement('div');
                    titleEl.className = 'article-title';
                    titleEl.textContent = post.title || 'Untitled';
                    postEl.appendChild(titleEl);

                    const sourceEl = document.createElement('div');
                    sourceEl.className = 'article-source';
                    sourceEl.textContent = `r/${post.subreddit} - ${formatDate(post.created_utc * 1000)}`;
                    postEl.appendChild(sourceEl);

                    const contentEl = document.createElement('div');
                    contentEl.className = 'article-description';
                    contentEl.textContent = post.selftext || post.text || 'No content available';
                    postEl.appendChild(contentEl);

                    if (post.url) {
                        const linkEl = document.createElement('a');
                        linkEl.href = post.url;
                        linkEl.target = '_blank';
                        linkEl.textContent = 'View on Reddit';
                        postEl.appendChild(linkEl);
                    }

                    redditContent.appendChild(postEl);
                });
            } else {
                redditContent.textContent = 'No Reddit posts found.';
            }

            // Populate metrics tab if available
            const metricsContent = document.getElementById('metricsContent');
            metricsContent.innerHTML = '';

            if (data.performance_metrics) {
                const metrics = data.performance_metrics;
                const modelName = data.model_used || selectedModel;

                const metricsHtml = `
                <h3>Model: ${modelName}</h3>
                <p><strong>Total Processing Time:</strong> ${Math.round(metrics.total_process_time * 1000)}ms</p>
                <p><strong>Query Reformatting:</strong> ${Math.round(metrics.reformatting_time * 1000)}ms</p>
                <p><strong>News Query Extraction:</strong> ${Math.round(metrics.news_query_time * 1000)}ms</p>
                <p><strong>Context Summarization:</strong> ${Math.round(metrics.summarization_time * 1000)}ms</p>
            `;

                metricsContent.innerHTML = metricsHtml;
            } else {
                metricsContent.textContent = 'Performance metrics not available.';
            }

            // Show feedback container if we have a query ID
            if (lastQueryId) {
                document.getElementById('feedbackContainer').style.display = 'block';
            }

            // Select summary tab by default
            document.querySelector('.tab[data-tab="summary"]').click();
        })
        .catch(error => {
            document.getElementById('loadingIndicator').style.display = 'none';
            alert('Error connecting to the server. Please try again later.');
            console.error('Error:', error);
        });
}

// Submit user feedback to the server
function submitFeedback(rating) {
    if (!lastQueryId) {
        alert('Cannot submit feedback without a valid query ID.');
        return;
    }

    const comments = document.getElementById('feedbackComments').value || '';

    fetch('http://localhost:5000/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query_id: lastQueryId,
            rating: rating,
            comments: comments
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('feedbackContainer').innerHTML = '<p>Thank you for your feedback!</p>';
            } else {
                alert('Error submitting feedback: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
            alert('Error submitting feedback. Please try again.');
        });
}

// Helper function to format dates
function formatDate(dateStr) {
    if (!dateStr) return 'Unknown date';
    try {
        const date = new Date(dateStr);
        return date.toLocaleDateString();
    } catch (e) {
        return 'Unknown date';
    }
} 