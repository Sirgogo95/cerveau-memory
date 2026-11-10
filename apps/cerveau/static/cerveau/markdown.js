// Markdown for Cerveau's replies, shared by the chat page and the memory viewer.
//
// Replies are rendered with marked and always sanitized with DOMPurify: a reply can
// contain HTML (say, from a stored memory) that must never run. Load after
// vendor/marked.umd.js and vendor/purify.min.js.

const markdown = {
  available: () => Boolean(window.marked && window.DOMPurify),

  // Sanitized HTML for a Markdown string. Images are dropped as well: loading one
  // means a request to another server, and an image URL is a classic way to leak
  // data out of a page (the page loads nothing from the internet, replies neither).
  html(text) {
    const raw = window.marked.parse(text, { gfm: true, breaks: true });
    return window.DOMPurify.sanitize(raw, { FORBID_TAGS: ["img", "style"] });
  },

  // Show `text` in `element` as rendered Markdown (plain text if the libraries
  // didn't load). The source is kept so streaming can re-render it as it grows.
  render(element, text) {
    element.dataset.source = text;
    if (markdown.available()) element.innerHTML = markdown.html(text);
    else element.textContent = text;
  },
};

// Links in replies open in a new tab, without giving that page access to this one.
document.addEventListener("DOMContentLoaded", () => {
  window.DOMPurify?.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
});

// Render every element marked data-markdown (server-rendered replies arrive as text).
document.addEventListener("DOMContentLoaded", () => {
  for (const element of document.querySelectorAll("[data-markdown]")) {
    markdown.render(element, element.textContent.trim());
  }
});
