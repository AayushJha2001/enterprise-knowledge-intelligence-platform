// Knowledge Assistant UI. Plain JS, no build step.
// Security: every piece of server text is inserted with textContent, never innerHTML,
// so a document can't inject markup or script into the page.
"use strict";

const TOKEN_KEY = "ekip.token";
const SUGGESTIONS = [
  "How much parental leave do I get?",
  "What is the hotel limit in Paris?",
  "My VPN shows error VPN-809. What do I do?",
];

const $ = (id) => document.getElementById(id);
let token = sessionStorage.getItem(TOKEN_KEY);
let me = null;
let pollTimer = null;

// ------------------------------------------------------------------ API

class Unauthorized extends Error {}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    signOut();
    throw new Unauthorized("Signed out");
  }
  return response;
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (detail && detail.message) return detail.message;
    return `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

// ------------------------------------------------------------------ sign-in

async function signIn(event) {
  event.preventDefault();
  const error = $("login-error");
  error.hidden = true;
  const form = new URLSearchParams({
    username: $("login-email").value,
    password: $("login-password").value,
  });
  const response = await fetch("/auth/token", { method: "POST", body: form });
  if (!response.ok) {
    error.textContent = await errorMessage(response);
    error.hidden = false;
    return;
  }
  token = (await response.json()).access_token;
  sessionStorage.setItem(TOKEN_KEY, token);
  $("login-password").value = "";
  await start();
}

function signOut() {
  token = null;
  me = null;
  sessionStorage.removeItem(TOKEN_KEY);
  clearTimeout(pollTimer);
  $("app-view").hidden = true;
  $("login-view").hidden = false;
}

// ------------------------------------------------------------------ documents

async function loadDocuments() {
  clearTimeout(pollTimer);
  const response = await api("/documents?limit=200");
  if (!response.ok) return;
  const { items } = await response.json();
  const list = $("doc-list");
  list.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "muted";
    empty.textContent = "No documents shared with you yet.";
    list.append(empty);
  }
  for (const doc of items) {
    const li = document.createElement("li");
    const dot = document.createElement("span");
    dot.className = `dot ${doc.status}`;
    dot.title = doc.status + (doc.error ? `: ${doc.error}` : "");
    const title = document.createElement("span");
    title.textContent = doc.title;
    li.append(dot, title);
    list.append(li);
  }
  // Keep refreshing while anything is still being ingested.
  if (items.some((d) => d.status === "pending" || d.status === "processing")) {
    pollTimer = setTimeout(loadDocuments, 3000);
  }
}

async function upload(event) {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  const status = $("upload-status");
  status.hidden = false;
  status.textContent = `Uploading ${file.name}…`;
  const body = new FormData();
  body.append("file", file);
  const response = await api("/documents", { method: "POST", body });
  status.textContent = response.ok
    ? `${file.name} uploaded; indexing…`
    : `${file.name}: ${await errorMessage(response)}`;
  await loadDocuments();
}

// ------------------------------------------------------------------ asking

function addExchange(question) {
  $("empty").hidden = true;
  const node = $("exchange-template").content.firstElementChild.cloneNode(true);
  node.querySelector(".question").textContent = question;
  $("thread").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function renderSources(node, sources) {
  const details = node.querySelector(".sources");
  const list = node.querySelector(".source-list");
  list.replaceChildren();
  for (const source of sources) {
    const li = document.createElement("li");
    li.dataset.number = source.number;
    const label = document.createElement("span");
    label.className = "source-label";
    // The label already starts with "[n]"; the <ol> numbering shows it too, so strip it.
    label.textContent = source.label.replace(/^\[\d+\]\s*/, "");
    const text = document.createElement("p");
    text.className = "source-text";
    text.textContent = source.content;
    li.append(label, text);
    list.append(li);
  }
  details.querySelector("summary").textContent =
    `${sources.length} source${sources.length === 1 ? "" : "s"} searched`;
  details.hidden = sources.length === 0;
}

function segmentFor(node, blocks, block) {
  if (!blocks.has(block)) {
    const answer = node.querySelector(".answer-text");
    answer.querySelector(".typing")?.remove();
    const segment = document.createElement("span");
    const text = document.createElement("span");
    const cites = document.createElement("span");
    segment.append(text, cites);
    answer.append(segment);
    blocks.set(block, { text, cites, numbers: new Set() });
  }
  return blocks.get(block);
}

function addCitation(node, segment, number) {
  if (segment.numbers.has(number)) return;
  segment.numbers.add(number);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cite";
  button.textContent = String(number);
  button.title = `Show source ${number}`;
  button.addEventListener("click", () => {
    const details = node.querySelector(".sources");
    details.open = true;
    for (const li of details.querySelectorAll("li")) {
      li.classList.toggle("highlight", li.dataset.number === String(number));
    }
    details.querySelector(`li[data-number="${number}"]`)?.scrollIntoView({ block: "nearest" });
  });
  segment.cites.append(button);
  node.querySelector(`.source-list li[data-number="${number}"]`)?.classList.add("cited");
}

function describe(done) {
  const parts = [];
  if (done.cached) parts.push(`from cache in ${done.timings_ms.cache_lookup} ms`);
  else if (done.timings_ms.llm_total) parts.push(`${(done.timings_ms.llm_total / 1000).toFixed(1)} s`);
  if (done.model) parts.push(done.model);
  if (done.cost_usd) parts.push(`$${done.cost_usd.toFixed(4)}`);
  if (done.status === "refused") parts.push("declined");
  if (done.status === "truncated") parts.push("answer cut short");
  if (done.status === "answered" && !done.grounded) parts.push("no citations: verify before relying on it");
  return parts.join(" · ");
}

// Server-sent events over fetch (EventSource can't POST or send an Authorization header).
async function* readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length) yield [event, JSON.parse(data.join("\n"))];
    }
  }
}

async function ask(question) {
  const node = addExchange(question);
  const answer = node.querySelector(".answer-text");
  const meta = node.querySelector(".meta");
  const blocks = new Map();
  try {
    const response = await api("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error(await errorMessage(response));

    for await (const [event, data] of readEvents(response)) {
      if (event === "sources") {
        renderSources(node, data.sources);
      } else if (event === "text") {
        segmentFor(node, blocks, data.block).text.textContent += data.text;
      } else if (event === "citation") {
        addCitation(node, segmentFor(node, blocks, data.block), data.source);
      } else if (event === "done") {
        if (!blocks.size) {
          answer.replaceChildren(document.createTextNode(data.answer));
        }
        meta.textContent = describe(data);
      } else if (event === "error") {
        throw new Error(data.message);
      }
    }
  } catch (error) {
    if (error instanceof Unauthorized) return;
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = error.message || "Something went wrong.";
    answer.querySelector(".typing")?.remove();
    answer.append(message);
  }
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function submitQuestion(event) {
  event.preventDefault();
  const input = $("question");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  autoGrow();
  const button = $("ask-button");
  button.disabled = true;
  try {
    await ask(question);
  } finally {
    button.disabled = false;
    input.focus();
  }
}

function autoGrow() {
  const input = $("question");
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight + 2}px`; // +2: top and bottom border
  // Show a scrollbar only once the text outgrows the box's maximum height.
  input.classList.toggle("scrolls", input.scrollHeight > input.clientHeight + 2);
}

// ------------------------------------------------------------------ start

async function start() {
  if (!token) {
    signOut();
    return;
  }
  let response;
  try {
    response = await api("/auth/me");
  } catch {
    return; // 401 already handled
  }
  if (!response.ok) {
    signOut();
    return;
  }
  me = await response.json();
  $("me-name").textContent = me.display_name;
  $("me-groups").textContent = me.is_admin
    ? "admin"
    : me.groups.map((g) => g.name).join(", ") || "no groups";
  $("upload-label").hidden = !me.is_admin;
  $("login-view").hidden = true;
  $("app-view").hidden = false;
  $("question").focus();
  await loadDocuments();
}

function init() {
  $("login-form").addEventListener("submit", signIn);
  $("logout").addEventListener("click", signOut);
  $("ask-form").addEventListener("submit", submitQuestion);
  $("upload-input").addEventListener("change", upload);
  const input = $("question");
  input.addEventListener("input", autoGrow);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("ask-form").requestSubmit();
    }
  });
  for (const text of SUGGESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "ghost small";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      input.value = text;
      $("ask-form").requestSubmit();
    });
    $("suggestions").append(chip);
  }
  start();
}

document.addEventListener("DOMContentLoaded", init);
