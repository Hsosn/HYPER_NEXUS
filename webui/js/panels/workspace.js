/**
 * Workspace panel — file tree, event feed, inline viewer, upload.
 */
import { api, el, fmtTime } from '../utils.js';
import { on } from '../state.js';
import { showSkeletons, toast } from '../enhancements.js';

const treeEl   = document.getElementById('workspace-tree');
const eventsEl = document.getElementById('workspace-events');
const viewerEl = document.getElementById('workspace-viewer');

export async function initWorkspace() {
  await refreshTree();
  await refreshEvents();

  document.getElementById('ws-refresh-btn')?.addEventListener('click', async () => {
    await refreshTree();
    await refreshEvents();
  });

  // Live updates
  on('ws:event', (msg) => {
    if (msg.kind === 'file_uploaded' || msg.kind === 'workspace_change') {
      prependEvent(msg.data || msg);
      refreshTree();
    }
  });

  // Upload
  const uploadBtn   = document.getElementById('upload-workspace-btn');
  const uploadInput = document.getElementById('workspace-file-input');
  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async () => {
      const files = Array.from(uploadInput.files);
      uploadInput.value = '';
      for (const file of files) await uploadFile(file);
      await refreshTree();
    });
  }

  // Drag & drop on viewer area
  if (viewerEl) {
    viewerEl.addEventListener('dragover', e => { e.preventDefault(); viewerEl.classList.add('drag-over'); });
    viewerEl.addEventListener('dragleave', () => viewerEl.classList.remove('drag-over'));
    viewerEl.addEventListener('drop', async e => {
      e.preventDefault();
      viewerEl.classList.remove('drag-over');
      const files = Array.from(e.dataTransfer.files);
      for (const f of files) await uploadFile(f);
      await refreshTree();
    });
  }

  window.addEventListener('panel:shown', (e) => {
    if (e.detail === 'workspace') { refreshTree(); refreshEvents(); }
  });
}

// ── Tree ─────────────────────────────────────────────────

async function refreshTree() {
  showSkeletons(treeEl, 5);
  try {
    const items = await api('/api/workspace/files');
    treeEl.innerHTML = '';
    if (!items.length) {
      treeEl.appendChild(_emptyState('No files yet', 'Files written by Hyper Nexus appear here.'));
      return;
    }
    items.forEach(item => treeEl.appendChild(_treeItem(item)));
  } catch {
    treeEl.innerHTML = '';
    treeEl.appendChild(_emptyState('Load error', 'Could not reach the workspace API.'));
  }
}

function _treeItem(item) {
  const row = el('div', { class: 'ws-row' + (item.is_dir ? ' ws-row--dir' : '') });

  const icon = el('span', { class: 'ws-row-icon' });
  icon.textContent = item.is_dir ? '📁' : fileIcon(item.name);

  const name = el('span', { class: 'ws-row-name' });
  name.textContent = item.name;

  row.appendChild(icon);
  row.appendChild(name);

  if (!item.is_dir) {
    const size = el('span', { class: 'ws-row-size' });
    size.textContent = fmtBytes(item.size);
    row.appendChild(size);
    row.addEventListener('click', () => loadFile(item.path, item.name));
  }
  return row;
}

// ── Events ───────────────────────────────────────────────

async function refreshEvents() {
  try {
    const events = await api('/api/workspace/events?limit=30');
    eventsEl.innerHTML = '';
    if (!events.length) {
      eventsEl.appendChild(_emptyState('No events yet', ''));
      return;
    }
    events.forEach(e => eventsEl.appendChild(_eventRow(e)));
  } catch {}
}

function _eventRow(e) {
  const TYPE_CLS = { created: 'ev-created', modified: 'ev-modified', deleted: 'ev-deleted' };
  const row = el('div', { class: 'ws-ev-row' });

  const dot = el('span', { class: 'ws-ev-dot ' + (TYPE_CLS[e.event_type] || 'ev-modified') });
  const path = el('span', { class: 'ws-ev-path' });
  path.textContent = e.path;
  const ago = el('span', { class: 'ws-ev-ago' });
  ago.textContent = fmtAgo(e.created_at);

  row.append(dot, path, ago);
  return row;
}

function prependEvent(data) {
  const empty = eventsEl.querySelector('.ws-empty');
  if (empty) empty.remove();
  const row = _eventRow({
    event_type: data.event_type || 'modified',
    path: data.path || data.name || '?',
    created_at: Date.now() / 1000,
  });
  eventsEl.insertBefore(row, eventsEl.firstChild);
}

// ── File viewer ──────────────────────────────────────────

async function loadFile(path, name) {
  // Clear viewer
  viewerEl.innerHTML = '';
  viewerEl.classList.remove('drag-over');

  const loading = el('div', { class: 'ws-viewer-loading' });
  loading.innerHTML = `<span class="ws-spinner"></span><span>Loading ${_esc(name)}…</span>`;
  viewerEl.appendChild(loading);

  try {
    const text = await api(`/api/workspace/file?path=${encodeURIComponent(path)}`);
    loading.remove();

    const wrap = el('div', { class: 'ws-viewer-wrap' });

    const header = el('div', { class: 'ws-viewer-header' });
    const left   = el('div', { class: 'ws-viewer-title' });
    left.innerHTML = `<span>${fileIcon(name)}</span><span>${_esc(name)}</span>`;
    const actions = el('div', { class: 'ws-viewer-actions' });
    const closeBtn = el('button', { class: 'ws-viewer-close' });
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', () => { viewerEl.innerHTML = ''; _showDropHint(); });
    actions.appendChild(closeBtn);
    header.appendChild(left);
    header.appendChild(actions);

    const lineCount = (text.match(/\n/g) || []).length + 1;
    const lineNums = el('div', { class: 'ws-line-nums' });
    for (let i = 1; i <= Math.min(lineCount, 2000); i++) {
      const ln = el('div', { class: 'ws-ln' });
      ln.textContent = i;
      lineNums.appendChild(ln);
    }

    const pre  = el('pre', { class: 'ws-viewer-pre' });
    const code = el('code');
    code.textContent = text;
    pre.appendChild(code);

    const body = el('div', { class: 'ws-viewer-body' });
    body.appendChild(lineNums);
    body.appendChild(pre);

    const footer = el('div', { class: 'ws-viewer-footer' });
    footer.textContent = `${lineCount} lines · ${fmtBytes(new Blob([text]).size)}`;

    wrap.append(header, body, footer);
    viewerEl.appendChild(wrap);

  } catch {
    loading.remove();
    const err = el('div', { class: 'ws-viewer-err' });
    err.textContent = 'Could not load file.';
    viewerEl.appendChild(err);
  }
}

function _showDropHint() {
  viewerEl.innerHTML = `
    <div class="ws-drop-hint">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" opacity=".3"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>
      <span>Select a file to preview, or drop files here to upload</span>
    </div>`;
}

// ── Upload ───────────────────────────────────────────────

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('subfolder', '');
  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      toast(`✓ Uploaded: ${data.name}`, 'success');
    } else {
      toast(`Upload failed: ${file.name}`, 'error');
    }
  } catch {
    toast(`Upload error: ${file.name}`, 'error');
  }
}

// ── Utilities ────────────────────────────────────────────

function _emptyState(title, sub) {
  const d = el('div', { class: 'ws-empty' });
  d.innerHTML = `<span class="ws-empty-title">${_esc(title)}</span>${sub ? `<span class="ws-empty-sub">${_esc(sub)}</span>` : ''}`;
  return d;
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  const map = {
    pdf:'📕', doc:'📝', docx:'📝', odt:'📝', txt:'📄', md:'📝', rtf:'📝',
    py:'🐍', js:'🟨', ts:'🔷', jsx:'🟨', tsx:'🔷', html:'🌐', css:'🎨',
    json:'📋', xml:'📋', yaml:'📋', yml:'📋', toml:'📋', ini:'⚙️', env:'🔑',
    sh:'⚙️', bat:'⚙️', ps1:'⚙️', sql:'🗄️', rs:'🦀', go:'🐹', cpp:'💻',
    c:'💻', h:'💻', cs:'💻', java:'☕', php:'🐘', rb:'💎', swift:'🍎',
    csv:'📊', xlsx:'📊', xls:'📊', ods:'📊', parquet:'📊', jsonl:'📊',
    jpg:'🖼️', jpeg:'🖼️', png:'🖼️', gif:'🖼️', svg:'🖼️', webp:'🖼️',
    bmp:'🖼️', ico:'🖼️', tiff:'🖼️',
    zip:'📦', tar:'📦', gz:'📦', bz2:'📦', rar:'📦', xz:'📦',
    mp3:'🎵', wav:'🎵', ogg:'🎵', flac:'🎵', aac:'🎵',
    mp4:'🎬', mov:'🎬', avi:'🎬', mkv:'🎬', webm:'🎬',
    epub:'📚', pptx:'📊', ppt:'📊',
  };
  return map[ext] || '📄';
}

function fmtBytes(n) {
  if (n < 1024)     return `${n} B`;
  if (n < 1048576)  return `${(n/1024).toFixed(1)} KB`;
  return `${(n/1048576).toFixed(1)} MB`;
}

function fmtAgo(ts) {
  const ago = Math.round(Date.now() / 1000 - ts);
  if (ago < 60)   return `${ago}s ago`;
  if (ago < 3600) return `${Math.floor(ago/60)}m ago`;
  return `${Math.floor(ago/3600)}h ago`;
}

function _esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str != null ? str : '');
  return d.innerHTML;
}
