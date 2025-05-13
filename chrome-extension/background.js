chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze-context",
    title: "\uD83E\uDDE0 Analyze Political Context",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyze-context") {
    chrome.tabs.sendMessage(tab.id, { action: "analyzeSelection" });
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "fetchAnalysis") {
    // Show loading status in the response immediately
    if (request.progressUpdates) {
      sendResponse({ status: "loading" });
    }

    fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: request.text })
    })
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => sendResponse({ data }))
      .catch(err => {
        console.error("Error fetching analysis:", err);
        sendResponse({ error: err.toString() });
      });
    return true; // Keep the message channel open for async response
  }
});
