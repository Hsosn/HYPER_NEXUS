/** Shared helpers: API client, DOM factory, markdown, modal. */

// ─────────────────────────────────────────────────────────────
// API client (with retry, dedup, and rate-limit handling)
// ─────────────────────────────────────────────────────────────
export function getAuthToken() {
  return localStorage.getItem('nexus.token') || '';
}

export function setAuthToken(token) {
  if (token) localStorage.setItem('nexus.token', token);
  else localStorage.removeItem('nexus.token');
}

// ── Request deduplication: identical concurrent GETs share one fetch ──
const _inflight = new Map();

// ── Retry configuration ──
const _RETRY_MAX_ATTEMPTS = 3;
const _RETRY_BASE_DELAY_MS = 1000;
const _RETRY_MAX_DELAY_MS = 10000;
const _RETRYABLE_STATUS = new Set([429, 502, 503, 504]);

function _sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function _parseRetryAfter(header) {
  if (!header) return null;
  const val = parseInt(header, 10);
  if (!isNaN(val)) return Math.min(val, 30) * 1000; // Cap at 30s
  // Try parse as HTTP-date (rare, but spec-compliant)
  const date = Date.parse(header);
  if (!isNaN(date)) return Math.min(Math.max(0, date - Date.now()), 30000);
  return null;
}

/**
 * Core fetch with automatic retry and exponential backoff for transient errors.
 * Retries on: 429 (rate limit), 502, 503, 504 (server errors).
 * Honors Retry-After header when present.
 */
async function _fetchWithRetry(path, opts, attempt = 0) {
  const token = getAuthToken();
  const headers = { ...(opts.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (!(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(path, {
    ...opts,
    headers,
    body: opts.body instanceof FormData ? opts.body
         : opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body,
  });

  if (!res.ok && _RETRYABLE_STATUS.has(res.status) && attempt < _RETRY_MAX_ATTEMPTS) {
    // Determine delay: prefer Retry-After header, else exponential backoff
    const retryAfter = _parseRetryAfter(res.headers.get('Retry-After'));
    const jitter = Math.random() * 500; // ±500ms jitter to avoid thundering herd
    const backoff = Math.min(_RETRY_BASE_DELAY_MS * Math.pow(2, attempt) + jitter, _RETRY_MAX_DELAY_MS);
    const delay = retryAfter !== null ? retryAfter : backoff;

    console.warn(`[api] ${res.status} on ${opts.method || 'GET'} ${path} — retry ${attempt + 1}/${_RETRY_MAX_ATTEMPTS} in ${Math.round(delay)}ms`);
    await _sleep(delay);
    return _fetchWithRetry(path, opts, attempt + 1);
  }

  if (!res.ok) {
    // Special handling for 429 — give a user-friendly message
    if (res.status === 429) {
      throw new Error(`429: Rate limit reached. Please wait a moment and try again.`);
    }
    // 401 — token may have expired
    if (res.status === 401 && path !== '/api/auth/login') {
      // Clear stale token so login modal shows on next action
      setAuthToken('');
    }
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }

  const ct = res.headers.get('Content-Type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

/**
 * Main API client with retry, dedup, and error handling.
 * - GET requests are deduplicated: concurrent identical GETs share one fetch.
 * - All requests get automatic retry on 429/5xx with exponential backoff.
 */
export async function api(path, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();

  // Deduplicate GET requests
  if (method === 'GET') {
    const cacheKey = `${method}:${path}`;
    if (_inflight.has(cacheKey)) {
      return _inflight.get(cacheKey);
    }
    const promise = _fetchWithRetry(path, opts).finally(() => _inflight.delete(cacheKey));
    _inflight.set(cacheKey, promise);
    return promise;
  }

  // Non-GET: always fire fresh (mutations should not be deduped)
  return _fetchWithRetry(path, opts);
}

// ─────────────────────────────────────────────────────────────
// DOM factory
// ─────────────────────────────────────────────────────────────
export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) {
      c.forEach(x => {
        if (x == null || x === false) return;
        e.appendChild(x instanceof Node ? x : document.createTextNode(String(x)));
      });
    } else {
      e.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
  }
  return e;
}

// ─────────────────────────────────────────────────────────────
// Formatters
// ─────────────────────────────────────────────────────────────
export function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

export function fmtNum(n) {
  if (!n) return '0';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function fmtDuration(ms) {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s ? ' ' + s + 's' : ''}`;
}

export function fmtAgo(tsSeconds) {
  const d = Math.max(0, Math.round(Date.now() / 1000 - tsSeconds));
  if (d < 60) return `${d}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

// ─────────────────────────────────────────────────────────────
// Markdown renderer
//   Supports: fenced code (with language label), inline code,
//   headings h1–h4, bold, italic, strike, links, lists (ul/ol),
//   blockquote, hr, tables, and paragraph breaks.
//   Output is sanitised — raw HTML from the model is escaped.
// ─────────────────────────────────────────────────────────────
export const _esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const _escAttr = s => String(s).replace(/["'&<>]/g, c => ({ '"': '&quot;', "'": '&#39;', '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

export function mdRender(text) {
  if (!text) return '';
  const src = String(text).replace(/\r\n?/g, '\n');

  // Pull out code fences first — their content must not be interpreted.
  const codeBlocks = [];
  let t = src.replace(/```([\w+.-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const i = codeBlocks.length;
    codeBlocks.push({ lang: (lang || '').toLowerCase().trim(), code });
    return `\u0000CODE${i}\u0000`;
  });

  // Escape everything else so we can splice safely.
  t = _esc(t);

  // Tables (very pragmatic GFM-style). Detect a header row followed by a
  // separator row then body rows, all consecutive.
  t = t.replace(
    /(^|\n)(\|[^\n]+\|)\n(\|[-:| ]+\|)\n((?:\|[^\n]+\|\n?)+)/g,
    (_, pre, header, sep, body) => {
      const hcells = header.trim().slice(1, -1).split('|').map(s => s.trim());
      const aligns = sep.trim().slice(1, -1).split('|').map(s => {
        const x = s.trim();
        if (x.startsWith(':') && x.endsWith(':')) return 'center';
        if (x.endsWith(':')) return 'right';
        if (x.startsWith(':')) return 'left';
        return '';
      });
      const rows = body.trim().split('\n').map(row =>
        row.trim().slice(1, -1).split('|').map(s => s.trim())
      );
      const thead = '<thead><tr>' + hcells.map((c, i) =>
        `<th${aligns[i] ? ` style="text-align:${aligns[i]}"` : ''}>${c}</th>`).join('') + '</tr></thead>';
      const tbody = '<tbody>' + rows.map(r =>
        '<tr>' + r.map((c, i) =>
          `<td${aligns[i] ? ` style="text-align:${aligns[i]}"` : ''}>${c}</td>`).join('') + '</tr>'
      ).join('') + '</tbody>';
      return `${pre}<table>${thead}${tbody}</table>`;
    }
  );

  // Headings (#, ##, ###, ####).
  t = t.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  t = t.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  t = t.replace(/^##\s+(.+)$/gm,  '<h2>$1</h2>');
  t = t.replace(/^#\s+(.+)$/gm,   '<h1>$1</h1>');

  // Horizontal rule (---, ***, ___).
  t = t.replace(/^(?:---|\*\*\*|___)\s*$/gm, '<hr>');

  // Blockquotes (collapse consecutive lines into one).
  t = t.replace(/(^|\n)((?:&gt;[^\n]*\n?)+)/g, (_, pre, block) => {
    const inner = block.split('\n')
      .map(l => l.replace(/^&gt;\s?/, ''))
      .filter(Boolean).join(' ');
    return `${pre}<blockquote>${inner}</blockquote>`;
  });

  // Lists — convert runs of "- " or "* " or "1. " into <ul>/<ol>.
  t = _renderLists(t);

  // Inline code.
  t = t.replace(/`([^`\n]+)`/g, (_, c) => `<code>${c}</code>`);

  // Bold / italic / strike.
  t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\b_([^_\n]+)_\b/g, '<em>$1</em>');
  t = t.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
  t = t.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');

  // Links [text](href).
  t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_, txt, href) => `<a href="${_escAttr(href)}" target="_blank" rel="noopener noreferrer">${txt}</a>`);
  // Bare URLs.
  t = t.replace(/(^|[\s(])((?:https?:\/\/)[^\s)]+)(?=[\s).,!?:;]|$)/g,
    (_, pre, href) => `${pre}<a href="${_escAttr(href)}" target="_blank" rel="noopener noreferrer">${href}</a>`);

  // Paragraphs: collapse blank-line-separated chunks.
  t = t.split(/\n{2,}/).map(chunk => {
    if (/^\s*<(h\d|ul|ol|blockquote|hr|table|pre|\u0000CODE)/.test(chunk)) return chunk;
    return `<p>${chunk.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');

  // Restore code blocks.
  t = t.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => {
    const { lang, code } = codeBlocks[Number(i)];
    const labelAttr = lang ? ` data-lang="${_escAttr(lang)}"` : '';
    return `<div class="code-wrapper"${labelAttr}>` +
      (lang ? `<span class="code-lang">${_esc(lang)}</span>` : '') +
      `<pre><code>${_esc(code).replace(/\n$/, '')}</code></pre></div>`;
  });

  return t;
}

function _renderLists(t) {
  const lines = t.split('\n');
  const out = [];
  let buf = null; // { type: 'ul'|'ol', items: [] }
  const flush = () => {
    if (!buf) return;
    out.push(`<${buf.type}>${buf.items.map(i => `<li>${i}</li>`).join('')}</${buf.type}>`);
    buf = null;
  };
  for (const line of lines) {
    const ul = line.match(/^\s*[-*]\s+(.*)/);
    const ol = line.match(/^\s*\d+\.\s+(.*)/);
    if (ul) {
      if (!buf || buf.type !== 'ul') { flush(); buf = { type: 'ul', items: [] }; }
      buf.items.push(ul[1]);
    } else if (ol) {
      if (!buf || buf.type !== 'ol') { flush(); buf = { type: 'ol', items: [] }; }
      buf.items.push(ol[1]);
    } else {
      flush();
      out.push(line);
    }
  }
  flush();
  return out.join('\n');
}

// ─────────────────────────────────────────────────────────────
// Lightweight modal
// ─────────────────────────────────────────────────────────────
export function modal(title, bodyEl, onConfirm, confirmText = 'Confirm') {
  const root = document.getElementById('modal-root');
  const backdrop = el('div', { class: 'modal-backdrop' });
  const box = el('div', { class: 'modal' });
  const cancelBtn = el('button', { class: 'btn ghost', onclick: close }, 'Cancel');
  const okBtn = el('button', {
    class: 'btn primary',
    onclick: async () => {
      try { await onConfirm(); close(); }
      catch (e) {
        // Surface inside the modal rather than window.alert.
        const err = box.querySelector('.modal-error') || el('div', {
          class: 'modal-error',
          style: {
            color: 'var(--danger)', fontSize: '12px', marginTop: '8px',
            padding: '8px 10px', background: 'var(--danger-soft)',
            border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px',
          },
        });
        err.textContent = e.message || String(e);
        if (!err.parentNode) box.insertBefore(err, box.querySelector('.modal-actions'));
      }
    },
  }, confirmText);

  box.appendChild(el('h2', {}, title));
  box.appendChild(bodyEl);
  box.appendChild(el('div', { class: 'modal-actions' }, cancelBtn, okBtn));
  backdrop.appendChild(box);
  root.appendChild(backdrop);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });
  document.addEventListener('keydown', escHandler);

  function close() {
    backdrop.remove();
    document.removeEventListener('keydown', escHandler);
  }
  function escHandler(e) { if (e.key === 'Escape') close(); }

  return { close };
}
