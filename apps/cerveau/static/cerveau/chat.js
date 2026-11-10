// Cerveau chat: sends a message and renders the streamed reply.
// The server answers with newline-delimited JSON events (see views.send_message):
// {"episode", "new"} once, {"tool", "arguments"} when Cerveau searches its memory,
// {"files"} once if the message shared files, {"delta"} pieces of the reply, then
// {"done"} or {"error"}. Files go up with the message (multipart); a refused one
// returns 400 with {"error"} before anything is stored.
//
// Replies are rendered as Markdown by markdown.js (loaded before this file); your
// own messages stay plain text.

// One line of the recall trace, for a memory tool call. Built with DOM nodes, never
// innerHTML: the arguments come from the model.
const trace = {
  day(iso) {
    const date = new Date(`${iso}T00:00:00`);
    if (Number.isNaN(date.getTime())) return iso;
    const sameYear = date.getFullYear() === new Date().getFullYear();
    return date.toLocaleDateString("en", {
      month: "short", day: "numeric", ...(sameYear ? {} : { year: "numeric" }),
    });
  },

  line(name, args) {
    const li = document.createElement("li");
    li.className = "active";
    const add = (text, className) => {
      const node = document.createElement(className === "q" ? "q" : "span");
      if (className && className !== "q") node.className = className;
      node.textContent = text;
      li.append(node);
    };
    if (name === "search_memory") {
      add("Searching memory for ");
      add(args.query || "…", "q");
      const details = [];
      if (args.entities?.length) details.push(`about ${args.entities.join(", ")}`);
      if (args.after) details.push(`from ${trace.day(args.after)}`);
      if (args.before) details.push(`before ${trace.day(args.before)}`);
      if (args.kinds?.length === 1) details.push(`${args.kinds[0]}s only`);
      if (details.length) add(` · ${details.join(" · ")}`, "detail");
    } else if (name === "read_memory") {
      const id = String(args.id || "");
      const what = { M: `message ${id}`, F: `fact ${id}` }[id.slice(0, 1).toUpperCase()] || id;
      add(`Opening ${what}`);
      if (args.detail === "messages") add(" · its messages", "detail");
    } else {
      add(`Using ${name}`);
    }
    return li;
  },
};


// Files: chosen with the paperclip or dropped on the page, shown as chips until sent,
// then as chips on your message that follow the file's reading state.
const files = {
  STATES: { pending: "Being read", done: "Read", failed: "Couldn't be read", unsupported: "Not readable" },

  chip(name, { removable = false, href = "" } = {}) {
    const li = document.createElement("li");
    li.className = "file-chip";
    const label = document.createElement(href ? "a" : "span");
    if (href) label.href = href;
    else label.className = "file-chip-name";
    label.textContent = name;
    li.append(label);
    if (removable) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "file-chip-remove";
      remove.setAttribute("aria-label", `Remove ${name}`);
      remove.textContent = "×";
      li.append(remove);
    } else {
      const state = document.createElement("span");
      state.className = "file-chip-state";
      li.append(state);
    }
    return li;
  },

  // Show a file's state on its chip: the dot, and a title for the details.
  show(li, file) {
    li.dataset.status = file.status;
    if (file.status === "done" && !file.title) li.dataset.summarizing = "";
    else delete li.dataset.summarizing;
    const what = file.status === "done" && !file.title ? "Read; being summarized" : files.STATES[file.status];
    li.title = file.error ? `${what}: ${file.error}` : what || "";
    const link = li.querySelector("a");
    if (link && file.title) link.textContent = file.title;
  },

  // Ask the server until the file is read and summarized (or can't be), for a few minutes.
  follow(li, statusUrl, tries = 90) {
    const id = li.dataset.fileId;
    if (!id || tries <= 0) return;
    const busy = li.dataset.status === "pending" || "summarizing" in li.dataset;
    if (!busy) return;
    setTimeout(async () => {
      try {
        const response = await fetch(statusUrl.replace("/0/", `/${id}/`));
        if (response.ok) files.show(li, await response.json());
      } catch { /* try again next time */ }
      files.follow(li, statusUrl, tries - 1);
    }, 2000);
  },
};

document.addEventListener("DOMContentLoaded", () => {
  const app = document.getElementById("app");
  const form = document.getElementById("composer");
  const input = document.getElementById("text");
  const button = document.getElementById("send");
  const think = document.getElementById("think");
  const log = document.getElementById("log");
  const inner = document.getElementById("log-inner");
  const label = document.getElementById("episode-label");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;
  const picker = document.getElementById("files");
  const pendingList = document.getElementById("pending-files");
  const problem = document.getElementById("composer-error");
  const statusUrl = form.dataset.statusUrl;
  const fileUrl = form.dataset.fileUrl;
  const maxBytes = Number(form.dataset.maxMb || 25) * 1024 * 1024;
  let pending = [];   // File objects chosen, not sent yet

  const say = (text) => { problem.textContent = text; problem.hidden = !text; };

  function pinButton() {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pin";
    button.setAttribute("aria-pressed", "false");
    button.title = "Pin: keep this in Cerveau's view for the rest of the conversation";
    button.innerHTML = '<span class="visually-hidden">Pin</span><svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><path d="M9 3h6l-1 6 4 3v2H6v-2l4-3-1-6zM12 14v7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    return button;
  }

  // Pin or unpin a message: it stays in Cerveau's view after the conversation moves on.
  inner.addEventListener("click", async (event) => {
    const button = event.target.closest(".pin");
    if (!button) return;
    const article = button.closest(".turn");
    const id = article?.dataset.messageId;
    if (!id) return;   // not stored yet
    const pinned = button.getAttribute("aria-pressed") !== "true";
    try {
      const response = await fetch(form.dataset.pinUrl.replace("/0/", `/${id}/`), {
        method: "POST",
        headers: { "X-CSRFToken": csrf },
        body: new URLSearchParams({ pinned: pinned ? "1" : "0" }),
      });
      if (!response.ok) return;
      button.setAttribute("aria-pressed", String(pinned));
      if (pinned) article.dataset.pinned = "";
      else delete article.dataset.pinned;
    } catch { /* stays as it was */ }
  });

  function drawPending() {
    pendingList.replaceChildren(...pending.map((file) => files.chip(file.name, { removable: true })));
    pendingList.hidden = pending.length === 0;
  }

  function choose(list) {
    say("");
    for (const file of list) {
      if (file.size > maxBytes) { say(`${file.name} is larger than ${form.dataset.maxMb} MB.`); continue; }
      if (!pending.some((p) => p.name === file.name && p.size === file.size)) pending.push(file);
    }
    drawPending();
  }

  const timeNow = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  const dayNow = () => new Date().toLocaleDateString("en", { weekday: "short" });
  const scrollDown = () => { log.scrollTop = log.scrollHeight; };
  const setState = (state) => { app.dataset.state = state; };

  function addTurn(role, text, shared = []) {
    document.getElementById("empty")?.remove();
    const article = document.createElement("article");
    article.className = `turn turn--${role}`;

    const head = document.createElement("p");
    head.className = "turn-label";
    head.textContent = { user: "You", assistant: "Cerveau", error: "Not answered" }[role];
    const time = document.createElement("time");
    time.textContent = timeNow();
    head.append(" ", time);
    if (role !== "error") head.append(pinButton());

    const body = document.createElement("div");
    body.className = "turn-body";
    if (role === "assistant") markdown.render(body, text);
    else body.textContent = text;

    article.append(head);
    if (role === "assistant") {
      const steps = document.createElement("ol");
      steps.className = "trace";
      steps.setAttribute("aria-label", "What Cerveau looked up");
      article.append(steps);
    }
    article.append(body);
    if (shared.length) {
      const list = document.createElement("ul");
      list.className = "turn-files";
      list.setAttribute("aria-label", "Files shared");
      for (const name of shared) {
        const li = files.chip(name);
        li.dataset.status = "pending";
        list.append(li);
      }
      article.append(list);
    }
    inner.append(article);
    scrollDown();
    return article;
  }

  function addDivider(episodeId) {
    const divider = document.createElement("div");
    divider.className = "divider";
    divider.textContent = `New episode · ${timeNow()}`;
    // The divider goes above the message that opened the episode.
    const lastUser = [...inner.querySelectorAll(".turn--user")].pop();
    inner.insertBefore(divider, lastUser);
    label.textContent = `Episode ${episodeId}`;
    const since = document.createElement("span");
    since.className = "meta-since";
    since.textContent = ` · since ${dayNow()} ${timeNow()}`;
    label.append(since);
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  }

  async function send(text, attached) {
    const names = attached.map((f) => f.name);
    const mine = addTurn("user", text || `(shared ${names.join(", ")})`, names);
    const reply = addTurn("assistant", "");
    reply.classList.add("turn--waiting");
    const body = reply.querySelector(".turn-body");
    const steps = reply.querySelector(".trace");
    button.disabled = true;
    setState("thinking");

    try {
      const data = new FormData();
      data.append("text", text);
      if (think.checked) data.append("think", "1");     // the Think switch: this turn only
      for (const file of attached) data.append("files", file);
      const response = await fetch(form.action, {
        method: "POST",
        headers: { "X-CSRFToken": csrf },
        body: data,
      });
      if (response.status === 400 && attached.length) {
        // A file was refused: nothing was sent. Put everything back to fix and retry.
        const refusal = await response.json().catch(() => ({}));
        mine.remove();
        reply.remove();
        input.value = text;
        autosize();
        pending = attached;
        drawPending();
        say(refusal.error || "A file couldn't be sent.");
        setState("idle");
        return;
      }
      if (!response.ok) throw new Error(`The server answered ${response.status}.`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();  // keep a partial line for the next read
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.new) addDivider(event.episode);
          if (event.files) {
            // The files are stored: link their chips and follow their reading.
            const chips = [...mine.querySelectorAll(".file-chip")];
            event.files.forEach((file, i) => {
              const old = chips[i];
              if (!old) return;
              const li = files.chip(file.title || file.filename, { href: fileUrl.replace("/0/", `/${file.id}/`) });
              li.dataset.fileId = file.id;
              files.show(li, file);
              old.replaceWith(li);
              files.follow(li, statusUrl);
            });
          }
          if (event.notice) say(event.notice);
          if (event.ids) {
            // Stored: the two messages can be pinned now.
            if (event.ids.user) mine.dataset.messageId = event.ids.user;
            if (event.ids.reply) reply.dataset.messageId = event.ids.reply;
          }
          if (event.tool) {
            steps.querySelectorAll(".active").forEach((li) => li.classList.remove("active"));
            steps.append(trace.line(event.tool, event.arguments || {}));
            setState("searching");
            scrollDown();
          }
          if (event.delta) {
            reply.classList.remove("turn--waiting");
            steps.querySelectorAll(".active").forEach((li) => li.classList.remove("active"));
            setState("thinking");
            markdown.render(body, (body.dataset.source || "") + event.delta);
            scrollDown();
          }
          if (event.error) throw new Error(event.error);
        }
      }
      reply.classList.remove("turn--waiting");
      reply.classList.add("turn--done");
      setState("idle");
    } catch (error) {
      reply.remove();
      addTurn(
        "error",
        `Cerveau couldn't answer: ${error.message}\nYour message was saved; send another to try again.`,
      );
      setState("error");
    } finally {
      button.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if ((!text && !pending.length) || button.disabled) return;
    const attached = pending;
    pending = [];
    drawPending();
    say("");
    input.value = "";
    autosize();
    send(text, attached);
  });

  picker.addEventListener("change", () => { choose(picker.files); picker.value = ""; });
  pendingList.addEventListener("click", (event) => {
    const remove = event.target.closest(".file-chip-remove");
    if (!remove) return;
    const index = [...pendingList.children].indexOf(remove.closest(".file-chip"));
    pending.splice(index, 1);
    drawPending();
    input.focus();
  });

  // Drop files anywhere on the page to attach them.
  let depth = 0;
  const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  app.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    depth += 1;
    app.dataset.dropping = "";
  });
  app.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) delete app.dataset.dropping;
  });
  app.addEventListener("dragover", (event) => { if (hasFiles(event)) event.preventDefault(); });
  app.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    delete app.dataset.dropping;
    choose(event.dataTransfer.files);
    input.focus();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener("input", autosize);

  // Starter prompts on an empty page fill the composer and send.
  inner.addEventListener("click", (event) => {
    const suggestion = event.target.closest(".suggestion");
    if (!suggestion) return;
    input.value = suggestion.dataset.prompt;
    form.requestSubmit();
  });

  // Past replies carry the searches that led to them (stored tool calls): draw
  // their recall trace with the same wording as live ones.
  for (const article of inner.querySelectorAll(".turn--assistant[data-tools]")) {
    let calls = [];
    try { calls = JSON.parse(article.dataset.tools); } catch { continue; }
    const steps = document.createElement("ol");
    steps.className = "trace";
    steps.setAttribute("aria-label", "What Cerveau looked up");
    for (const call of calls) {
      const line = trace.line(call.tool, call.arguments || {});
      line.classList.remove("active");
      steps.append(line);
    }
    article.querySelector(".turn-label").after(steps);
  }

  // Replies already on the page arrive as plain text from the server; render them.
  for (const body of inner.querySelectorAll(".turn--assistant .turn-body")) {
    markdown.render(body, body.textContent.trim());
  }
  inner.querySelectorAll(".turn--assistant").forEach((t) => t.classList.add("turn--done"));
  // Files on past messages: show their state, and follow the ones still being read.
  for (const li of inner.querySelectorAll(".file-chip[data-file-id]")) {
    li.title = "summarizing" in li.dataset && li.dataset.status === "done"
      ? "Read; being summarized" : files.STATES[li.dataset.status] || "";
    files.follow(li, statusUrl);
  }
  scrollDown();
});
