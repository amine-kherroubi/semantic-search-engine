// ---------------------------------------------------------------------
// Semantic Search Engine — Console frontend
// No framework, no build step. Talks to FastAPI backend over REST.
// ---------------------------------------------------------------------

const API_BASE = "";

// ---------- tiny helpers ----------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Turn **term** markers from backend snippets.py into <mark>.
function renderSnippet(snippet) {
  const escaped = escapeHtml(snippet);
  return escaped.replace(/\*\*(.+?)\*\*/g, "<mark>$1</mark>");
}

async function apiFetch(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON, fall back to statusText */
    }
    throw new Error(detail);
  }
  return res.json();
}

// ---------- health check ----------

async function checkHealth() {
  const pill = $("#health-pill");
  try {
    const data = await apiFetch("/api/health");
    if (data.status === "ok") {
      pill.textContent = `API ok · ${data.document_count.toLocaleString()} docs`;
      pill.className = "pill pill-ok";
    } else {
      pill.textContent = "DB unreachable";
      pill.className = "pill pill-error";
    }
  } catch (err) {
    pill.textContent = "API unreachable";
    pill.className = "pill pill-error";
  }
}

// ---------- tab switching ----------

function setupTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      const view = tab.dataset.view;
      $$(".view").forEach((v) => v.classList.remove("active"));
      $(`#view-${view}`).classList.add("active");

      if (view === "browse" && !browseState.loadedOnce) {
        loadArticles();
      }
    });
  });
}

// =======================================================================
// SEARCH VIEW
// =======================================================================

function approachLabel(approach) {
  return { tfidf: "TF-IDF", semantic: "Semantic" }[approach] || approach;
}

function renderStats(stats) {
  const el = $("#search-stats");
  el.hidden = false;
  el.innerHTML = `
    <span class="stat">latency: <strong>${stats.latency_ms.toFixed(1)} ms</strong></span>
    <span class="stat">results: <strong>${stats.num_results}</strong></span>
    <span class="stat">approach: <strong>${approachLabel(stats.approach)}</strong></span>
    <span class="stat">index: <strong>${escapeHtml(stats.indexing_method)}</strong></span>
    ${stats.notes ? `<span class="stat stat-note">${escapeHtml(stats.notes)}</span>` : ""}
  `;
}

function renderResults(results) {
  const list = $("#results-list");
  list.innerHTML = "";

  if (results.length === 0) {
    $("#search-status").textContent =
      "No results. Try a different query or retrieval approach.";
    return;
  }
  $("#search-status").textContent = "";

  for (const r of results) {
    const li = document.createElement("li");
    li.className = "result-card";

    li.innerHTML = `
      <div class="result-top">
        <span>
          <span class="result-rank">#${r.rank}</span>
          <span class="result-title">${escapeHtml(r.title || "(untitled)")}</span>
        </span>
        <span class="result-score">score ${r.score.toFixed(4)}</span>
      </div>
      <div class="result-snippet">${renderSnippet(r.snippet)}</div>
      <div class="result-meta">
        <span class="tag">${escapeHtml(r.source || "unknown source")}</span>
        ${r.category ? `<span class="tag">${escapeHtml(r.category)}</span>` : ""}
      </div>
    `;

    list.appendChild(li);
  }
}

async function runSearch(e) {
  e.preventDefault();
  const query = $("#query-input").value.trim();
  if (!query) return;

  const approach = $("#approach-select").value;
  const topK = parseInt($("#topk-select").value, 10);
  const btn = $("#search-btn");

  btn.disabled = true;
  btn.textContent = "Searching…";
  $("#search-status").textContent = "Running query…";
  $("#search-status").className = "status-line";
  $("#results-list").innerHTML = "";
  $("#search-stats").hidden = true;

  try {
    const data = await apiFetch("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, approach, top_k: topK }),
    });
    renderStats(data.stats);
    renderResults(data.results);
  } catch (err) {
    $("#search-status").textContent = `Search failed: ${err.message}`;
    $("#search-status").className = "status-line error";
  } finally {
    btn.disabled = false;
    btn.textContent = "Search";
  }
}

// =======================================================================
// BROWSE VIEW
// =======================================================================

const browseState = {
  page: 1,
  pageSize: 25,
  totalPages: 1,
  category: "",
  query: "",
  loadedOnce: false,
  debounceTimer: null,
};

async function loadCategories() {
  try {
    const categories = await apiFetch("/api/categories");
    const sel = $("#filter-category");
    for (const c of categories) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    }
  } catch {
    // Non-fatal: filter dropdown just stays at "All categories".
  }
}

function renderArticleRows(items) {
  const tbody = $("#articles-tbody");
  tbody.innerHTML = "";

  if (items.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="4">No articles match this filter.</td></tr>`;
    return;
  }

  for (const item of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="col-id">${item.id}</td>
      <td class="title-cell">${escapeHtml(item.title || "(untitled)")}</td>
      <td class="col-source">${escapeHtml(item.source || "—")}</td>
      <td class="col-category">${item.category ? `<span class="category-badge">${escapeHtml(item.category)}</span>` : "—"}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function loadArticles() {
  browseState.loadedOnce = true;
  $("#browse-status").textContent = "Loading articles…";
  $("#browse-status").className = "status-line";

  const params = new URLSearchParams({
    page: browseState.page,
    page_size: browseState.pageSize,
  });
  if (browseState.category) params.set("category", browseState.category);
  if (browseState.query) params.set("q", browseState.query);

  try {
    const data = await apiFetch(`/api/articles?${params.toString()}`);
    renderArticleRows(data.items);
    browseState.totalPages = data.total_pages;
    $("#page-indicator").textContent =
      `Page ${data.page} of ${data.total_pages} · ${data.total.toLocaleString()} articles`;
    $("#prev-page").disabled = data.page <= 1;
    $("#next-page").disabled = data.page >= data.total_pages;
    $("#browse-status").textContent = "";
  } catch (err) {
    $("#browse-status").textContent = `Failed to load articles: ${err.message}`;
    $("#browse-status").className = "status-line error";
    renderArticleRows([]);
  }
}

function setupBrowseControls() {
  $("#refresh-btn").addEventListener("click", () => {
    browseState.page = 1;
    loadArticles();
  });

  $("#filter-text").addEventListener("input", (e) => {
    browseState.query = e.target.value.trim();
    clearTimeout(browseState.debounceTimer);
    browseState.debounceTimer = setTimeout(() => {
      browseState.page = 1;
      loadArticles();
    }, 350);
  });

  $("#filter-category").addEventListener("change", (e) => {
    browseState.category = e.target.value;
    browseState.page = 1;
    loadArticles();
  });

  $("#page-size-select").addEventListener("change", (e) => {
    browseState.pageSize = parseInt(e.target.value, 10);
    browseState.page = 1;
    loadArticles();
  });

  $("#prev-page").addEventListener("click", () => {
    if (browseState.page > 1) {
      browseState.page -= 1;
      loadArticles();
    }
  });

  $("#next-page").addEventListener("click", () => {
    if (browseState.page < browseState.totalPages) {
      browseState.page += 1;
      loadArticles();
    }
  });
}

// =======================================================================
// INIT
// =======================================================================

function init() {
  setupTabs();
  setupBrowseControls();
  $("#search-form").addEventListener("submit", runSearch);

  checkHealth();
  loadCategories();
  loadArticles(); // browse is the default view
}

document.addEventListener("DOMContentLoaded", init);
