let popupDiv = null;

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "analyzeSelection") {
    const selection = window.getSelection();
    if (!selection || selection.toString().trim() === "") return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const selectedText = selection.toString();

    // Show loading popup
    showPopup(rect, "<div class='windsurf-loading'>Analyzing...</div>");

    // Send selected text to background for API call
    chrome.runtime.sendMessage(
      { action: "fetchAnalysis", text: selectedText },
      (response) => {
        if (response && response.data) {
          const summary = response.data.summary;
          const newsArticles = response.data.raw_context.news_articles || [];
          const redditPosts = response.data.raw_context.reddit_posts || [];

          // Construct a tabbed interface
          let tabsHtml = `
            <div class="tabs">
              <div class="tab active" data-tab="summary">Summary</div>
              <div class="tab" data-tab="news">News (${newsArticles.length})</div>
              <div class="tab" data-tab="reddit">Reddit (${redditPosts.length})</div>
            </div>
          `;

          let summaryContent = `<div id="summaryTab" class="tab-content active">
            <div class="windsurf-summary"><b>Summary:</b><br>${summary}</div>
          </div>`;

          let newsContent = `<div id="newsTab" class="tab-content">`;
          if (newsArticles.length > 0) {
            newsArticles.forEach(article => {
              newsContent += `
                <div class="news-article">
                  <div class="article-title">${article.title || 'Untitled'}</div>
                  <div class="article-source">${article.source?.name || 'Source'} - ${formatDate(article.publishedAt)}</div>
                  <div class="article-description">${article.description || article.content || 'No description available'}</div>
                  ${article.url ? `<a href="${article.url}" target="_blank">Read more</a>` : ''}
                </div>
              `;
            });
          } else {
            newsContent += '<p>No news articles found.</p>';
          }
          newsContent += '</div>';

          let redditContent = `<div id="redditTab" class="tab-content">`;
          if (redditPosts.length > 0) {
            redditPosts.forEach(post => {
              redditContent += `
                <div class="news-article">
                  <div class="article-title">${post.title || 'Untitled'}</div>
                  <div class="article-source">r/${post.subreddit} - ${formatDate(post.created_utc * 1000)}</div>
                  <div class="article-description">${post.selftext || post.text || 'No content available'}</div>
                  ${post.url ? `<a href="${post.url}" target="_blank">View on Reddit</a>` : ''}
                </div>
              `;
            });
          } else {
            redditContent += '<p>No Reddit posts found.</p>';
          }
          redditContent += '</div>';

          showPopup(rect, tabsHtml + summaryContent + newsContent + redditContent);

          // Set up tab switching
          setTimeout(() => {
            if (popupDiv) {
              const tabs = popupDiv.querySelectorAll('.tab');
              tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                  // Remove active class from all tabs and contents
                  tabs.forEach(t => t.classList.remove('active'));
                  const tabContents = popupDiv.querySelectorAll('.tab-content');
                  tabContents.forEach(content => content.classList.remove('active'));

                  // Add active class to clicked tab and corresponding content
                  tab.classList.add('active');
                  const tabName = tab.getAttribute('data-tab');
                  popupDiv.querySelector(`#${tabName}Tab`).classList.add('active');
                });
              });
            }
          }, 0);
        } else {
          showPopup(rect, "<div class='windsurf-error'>Error fetching analysis.</div>");
        }
      }
    );
  }
});

function formatDate(dateStr) {
  if (!dateStr) return 'Unknown date';
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString();
  } catch (e) {
    return 'Unknown date';
  }
}

function showPopup(rect, html) {
  // Always remove any existing popup
  if (popupDiv) popupDiv.remove();

  // Create new popup
  popupDiv = document.createElement("div");
  popupDiv.className = "windsurf-popup";
  popupDiv.style.position = "fixed";
  popupDiv.style.zIndex = 2147483647; // Maximum z-index to ensure visibility
  popupDiv.style.background = "#fff";
  popupDiv.style.border = "1px solid #ccc";
  popupDiv.style.borderRadius = "8px";
  popupDiv.style.padding = "12px";
  popupDiv.style.boxShadow = "0 2px 12px rgba(0,0,0,0.15)";
  popupDiv.style.minWidth = "320px";
  popupDiv.style.maxWidth = "400px";
  popupDiv.style.maxHeight = "80vh"; // Limit height to 80% of viewport
  popupDiv.style.overflowY = "auto"; // Add scrolling for content

  // Add content
  popupDiv.innerHTML = html + `<div style='text-align:right;margin-top:8px;'><button class='windsurf-close'>Close</button></div>`;

  // Add to document
  document.body.appendChild(popupDiv);

  // Fixed position at center-right of screen instead of relative to selection
  const popupRect = popupDiv.getBoundingClientRect();
  const top = Math.max(20, (window.innerHeight - popupRect.height) / 2);
  const right = 20; // 20px from right edge

  popupDiv.style.top = `${top}px`;
  popupDiv.style.right = `${right}px`;
  popupDiv.style.left = "auto"; // Clear any left value

  // Setup close button
  popupDiv.querySelector('.windsurf-close').onclick = () => popupDiv.remove();

  // Make it draggable for better user experience
  let isDragging = false;
  let startX, startY, startLeft, startTop;

  // Add a drag handle
  const dragHandle = document.createElement("div");
  dragHandle.style.cursor = "move";
  dragHandle.style.textAlign = "center";
  dragHandle.style.padding = "4px 0";
  dragHandle.style.marginBottom = "8px";
  dragHandle.style.borderBottom = "1px solid #eee";
  dragHandle.innerHTML = "<span style='color:#999;'>• • •</span>";

  popupDiv.insertBefore(dragHandle, popupDiv.firstChild);

  // Setup drag functionality
  dragHandle.addEventListener("mousedown", startDrag);

  function startDrag(e) {
    e.preventDefault();
    isDragging = true;

    // Get starting positions
    startX = e.clientX;
    startY = e.clientY;
    startLeft = parseInt(popupDiv.style.right, 10) || 0;
    startTop = parseInt(popupDiv.style.top, 10) || 0;

    // Add event listeners for dragging
    document.addEventListener("mousemove", drag);
    document.addEventListener("mouseup", stopDrag);
  }

  function drag(e) {
    if (!isDragging) return;

    // Calculate new position
    const dx = startX - e.clientX;
    const dy = e.clientY - startY;

    // Switch to left positioning during drag
    if (popupDiv.style.right !== "auto") {
      const rightVal = parseInt(popupDiv.style.right, 10) || 0;
      popupDiv.style.left = `${window.innerWidth - rightVal - popupRect.width}px`;
      popupDiv.style.right = "auto";
    }

    // Update position
    popupDiv.style.left = `${parseInt(popupDiv.style.left, 10) - dx}px`;
    popupDiv.style.top = `${startTop + dy}px`;

    // Update start position for next move
    startX = e.clientX;
  }

  function stopDrag() {
    isDragging = false;
    document.removeEventListener("mousemove", drag);
    document.removeEventListener("mouseup", stopDrag);
  }
}
