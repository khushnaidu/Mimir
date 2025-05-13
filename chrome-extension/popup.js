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
            { action: 'fetchAnalysis', text: text },
            (response) => {
                // Hide loading
                loadingIndicator.style.display = 'none';

                if (response && response.data) {
                    // Show results container
                    resultContainer.style.display = 'block';

                    // Display summary
                    summaryContent.innerHTML = `<p>${response.data.summary || 'No summary available.'}</p>`;

                    // Display news articles
                    if (response.data.news_articles && response.data.news_articles.length > 0) {
                        let newsHtml = '';
                        response.data.news_articles.forEach(article => {
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
                    if (response.data.reddit_posts && response.data.reddit_posts.length > 0) {
                        let redditHtml = '';
                        response.data.reddit_posts.forEach(post => {
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
}); 