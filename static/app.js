// Chat client for POST /chat. No build step, no dependencies.
const chat = document.getElementById("chat");
const form = document.getElementById("ask");
const input = document.getElementById("question");
const send = document.getElementById("send");

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// Render the answer as text, turning [POL-XX-000] markers into chips.
function renderAnswer(text, citations) {
  const urls = Object.fromEntries(citations.map(c => [c.doc_id, c.url]));
  const frag = document.createDocumentFragment();
  text.split("\n").forEach((line, i) => {
    if (i) frag.appendChild(document.createElement("br"));
    const parts = line.split(/(\[[^\]]*POL-[A-Z]+-\d{3}[^\]]*\])/g);
    parts.forEach(p => {
      const ids = p.match(/POL-[A-Z]+-\d{3}/g);
      if (p.startsWith("[") && ids) {
        ids.forEach(id => {
          const a = el("a", "chip", id);
          a.href = urls[id] || `/docs/${id}`;
          a.target = "_blank";
          frag.appendChild(a);
        });
      } else {
        frag.appendChild(document.createTextNode(p));
      }
    });
  });
  return frag;
}

function addUser(text) {
  const m = el("div", "msg user");
  m.appendChild(el("div", "bubble", text));
  chat.appendChild(m);
  m.scrollIntoView({block: "end"});
}

function addBot(data) {
  const m = el("div", "msg bot");
  const b = el("div", "bubble" + (data.refused ? " refused" : ""));
  b.appendChild(renderAnswer(data.answer, data.citations || []));
  m.appendChild(b);

  if (data.citations && data.citations.length) {
    const list = el("div", "sources");
    list.appendChild(el("div", "sources-title", data.citation_inferred ? "Source (inferred)" : "Sources"));
    data.citations.forEach(c => {
      const card = el("div", "source");
      const a = el("a", "", `${c.doc_id} · ${c.title}`);
      a.href = c.url; a.target = "_blank";
      card.appendChild(a);
      card.appendChild(el("div", "section", c.section));
      card.appendChild(el("blockquote", "", c.snippet));
      list.appendChild(card);
    });
    m.appendChild(list);
  }
  const meta = el("div", "meta",
    `${data.latency_ms} ms${data.cached ? " · cached" : ""}${data.refused ? " · refused" : ""}`);
  m.appendChild(meta);
  chat.appendChild(m);
  m.scrollIntoView({block: "end"});
}

function addError(text) {
  const m = el("div", "msg bot");
  m.appendChild(el("div", "bubble error", text));
  chat.appendChild(m);
}

async function ask(question) {
  addUser(question);
  send.disabled = true; input.disabled = true;
  const typing = el("div", "msg bot typing", "Searching the policies…");
  chat.appendChild(typing);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question}),
      signal: controller.signal,
    });
    const data = await res.json();
    typing.remove();
    if (!res.ok) addError(data.error || `Error ${res.status}`);
    else addBot(data);
  } catch (e) {
    typing.remove();
    addError(e.name === "AbortError"
      ? "The server took too long to answer (over 2 minutes). Please try again."
      : "Could not reach the server. If the app was idle, it may be waking up; try again in a minute.");
  } finally {
    clearTimeout(timer);
    send.disabled = false; input.disabled = false; input.focus();
  }
}

form.addEventListener("submit", e => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});

document.querySelectorAll(".example").forEach(btn =>
  btn.addEventListener("click", () => ask(btn.textContent)));

// Policy browser panel
const panel = document.getElementById("docs-panel");
document.getElementById("docs-toggle").addEventListener("click", async e => {
  e.preventDefault();
  if (!panel.hidden) { panel.hidden = true; return; }
  if (!panel.dataset.loaded) {
    const docs = await (await fetch("/api/docs")).json();
    const ul = el("ul");
    docs.forEach(d => {
      const li = el("li");
      const a = el("a", "", `${d.doc_id} · ${d.title}`);
      a.href = d.url; a.target = "_blank";
      li.appendChild(a);
      li.appendChild(el("span", "fmt", d.format.toUpperCase()));
      ul.appendChild(li);
    });
    panel.appendChild(ul);
    panel.dataset.loaded = "1";
  }
  panel.hidden = false;
});
