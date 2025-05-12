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
          showPopup(rect, `<div class='windsurf-summary'><b>Summary:</b><br>${summary}</div>`);
        } else {
          showPopup(rect, "<div class='windsurf-error'>Error fetching analysis.</div>");
        }
      }
    );
  }
});

function showPopup(rect, html) {
  if (popupDiv) popupDiv.remove();
  popupDiv = document.createElement("div");
  popupDiv.className = "windsurf-popup";
  popupDiv.style.position = "fixed";
  // Temporarily set offscreen to measure dimensions
  popupDiv.style.left = "-9999px";
  popupDiv.style.top = "-9999px";
  popupDiv.style.zIndex = 99999;
  popupDiv.style.background = "#fff";
  popupDiv.style.border = "1px solid #ccc";
  popupDiv.style.borderRadius = "8px";
  popupDiv.style.padding = "12px";
  popupDiv.style.boxShadow = "0 2px 12px rgba(0,0,0,0.15)";
  popupDiv.style.minWidth = "320px";
  popupDiv.style.maxWidth = "400px";
  popupDiv.innerHTML = html + `<div style='text-align:right;'><button class='windsurf-close' style='margin-top:8px;'>Close</button></div>`;
  document.body.appendChild(popupDiv);
  // Now measure and reposition
  const popupRect = popupDiv.getBoundingClientRect();
  let left = rect.left + window.scrollX;
  let top = rect.bottom + window.scrollY + 5;
  // Clamp left to at least 10px from left edge
  if (left < 10) left = 10;
  // Clamp right to at most 10px from right edge
  if (left + popupRect.width > window.innerWidth - 10) {
    left = window.innerWidth - popupRect.width - 10;
    if (left < 10) left = 10; // fallback if too wide
  }
  // If popup would go off the bottom, show above selection if possible
  if (top + popupRect.height > window.innerHeight - 10) {
    let above = rect.top + window.scrollY - popupRect.height - 5;
    if (above >= 10) {
      top = above;
    } else {
      top = 10; // fallback: clamp to top
    }
  }
  // Clamp top to at least 10px from top
  if (top < 10) top = 10;
  popupDiv.style.left = `${left}px`;
  popupDiv.style.top = `${top}px`;
  popupDiv.querySelector('.windsurf-close').onclick = () => popupDiv.remove();
}
