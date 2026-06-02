/**
 * Memory Panel v6 — Professional category-driven layout
 * - Category pill tabs (All | Factual | Semantic | Episodic | Procedural)
 * - Search bar with live filter
 * - Clean card design per memory with colored stripe, badge, importance, delete
 * - Stats summary row
 */
import { api } from '../utils.js';
import { toast } from '../enhancements.js';

const container = document.getElementById('memories-list');
const scrollWrap = document.querySelector('.mem-scroll-wrap');
const statsEl   = document.getElementById('memory-stats-bar');
const filterEl  = document.getElementById('memory-filter');

const TYPES = ['factual','semantic','episodic','procedural'];

const TYPE_META = {
  factual:    { color: 'var(--factual)',    label: 'Factual',    icon: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>' },
  semantic:   { color: 'var(--semantic)',   label: 'Semantic',   icon: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>' },
  episodic:   { color: 'var(--episodic)',   label: 'Episodic',   icon: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>' },
  procedural: { color: 'var(--procedural)', label: 'Procedural', icon: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>' },
};

let _activeType = '';
let _searchQuery = '';
let _allMems = [];

export async function initMemory() {
  await refresh();
  if (filterEl) filterEl.addEventListener('change', () => {
    _activeType = filterEl.value;
    renderList();
  });
  window.addEventListener('panel:shown', (e) => {
    if (e.detail === 'memory') refresh();
  });
  // Auto-refresh memories when a chat completes so the user sees
  // newly consolidated facts without having to switch tabs manually.
  window.addEventListener('ws:final_answer', () => {
    refresh().catch(() => {});
  });
  window.addEventListener('ws:memory_consolidated', () => {
    refresh().catch(() => {});
  });
}

async function refresh() {
  if (!container) return;

  // Skeleton
  container.innerHTML = '';
  for (let i = 0; i < 4; i++) {
    const s = document.createElement('div');
    s.className = 'skeleton skeleton-card';
    s.style.opacity = String(1 - i * 0.2);
    container.appendChild(s);
  }

  try {
    _allMems = await api('/api/memories');
    renderStatsBar();
    renderTabs();
    renderSearch();
    renderList();
    // Scroll to bottom on new memories
    if (scrollWrap) scrollWrap.scrollTop = scrollWrap.scrollHeight;
  } catch (e) {
    container.innerHTML = `
      <div class="mem-empty">
        <div class="mem-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
        </div>
        <div class="mem-empty-title">Failed to load memories</div>
        <div class="mem-empty-sub">${esc(e.message || 'Check server connection')}</div>
      </div>`;
  }
}

// ── Stats row ──────────────────────────────────────────────
function renderStatsBar() {
  if (!statsEl) return;
  statsEl.innerHTML = '';
  statsEl.className = 'mem-stat-row';

  const total = _allMems.length;
  const allPill = makeStatPill('All', total, 'var(--a)');
  allPill.classList.add('active');
  allPill.addEventListener('click', () => { _activeType = ''; if (filterEl) filterEl.value = ''; renderTabs(); renderList(); updateStatActive(); });
  statsEl.appendChild(allPill);

  TYPES.forEach(t => {
    const count = _allMems.filter(m => (m.kind || 'factual') === t).length;
    const pill = makeStatPill(TYPE_META[t].label, count, TYPE_META[t].color);
    pill.addEventListener('click', () => { _activeType = t; if (filterEl) filterEl.value = t; renderTabs(); renderList(); updateStatActive(); });
    statsEl.appendChild(pill);
  });
}

function makeStatPill(label, count, color) {
  const div = document.createElement('div');
  div.className = 'mem-stat-pill';
  div.dataset.label = label;
  div.innerHTML = `<span class="mem-stat-dot" style="background:${color}"></span><span class="mem-stat-count">${count}</span><span class="mem-stat-label">${label}</span>`;
  return div;
}

function updateStatActive() {
  if (!statsEl) return;
  statsEl.querySelectorAll('.mem-stat-pill').forEach(pill => {
    pill.classList.toggle('active', pill.dataset.label === (_activeType ? TYPE_META[_activeType]?.label : 'All'));
  });
}

// ── Category tabs ──────────────────────────────────────────
function renderTabs() {
  // Remove old tabs to prevent DOM leak on repeated refresh() calls
  const oldTabs = container.parentElement.querySelectorAll('.mem-tabs');
  oldTabs.forEach(t => { if (t !== container.parentElement.querySelector('.mem-tabs')) t.remove(); });

  let tabsEl = container.parentElement.querySelector('.mem-tabs');
  if (!tabsEl) {
    tabsEl = document.createElement('div');
    tabsEl.className = 'mem-tabs';
    // Insert before memories-list
    container.parentElement.insertBefore(tabsEl, container);
  }
  tabsEl.innerHTML = '';

  const allTab = makeTab('All', '');
  if (_activeType === '') allTab.classList.add('active');
  allTab.addEventListener('click', () => { _activeType = ''; if (filterEl) filterEl.value = ''; renderTabs(); renderList(); updateStatActive(); });
  tabsEl.appendChild(allTab);

  TYPES.forEach(t => {
    const tab = makeTab(TYPE_META[t].label, t);
    tab.style.setProperty('--tab-color', TYPE_META[t].color);
    if (_activeType === t) tab.classList.add('active');
    tab.addEventListener('click', () => { _activeType = t; if (filterEl) filterEl.value = t; renderTabs(); renderList(); updateStatActive(); });
    tabsEl.appendChild(tab);
  });
}

function makeTab(label, type) {
  const btn = document.createElement('button');
  btn.className = 'mem-tab';
  btn.dataset.type = type;
  const meta = type ? TYPE_META[type] : null;
  const dotHtml = meta ? `<span class="mem-tab-dot" style="background:${meta.color}"></span>` : '';
  btn.innerHTML = `${dotHtml}${label}`;
  return btn;
}

// ── Search bar ─────────────────────────────────────────────
function renderSearch() {
  // Remove old search wraps to prevent DOM leak on repeated refresh() calls
  const oldSearches = container.parentElement.querySelectorAll('.mem-search-wrap');
  oldSearches.forEach(s => { if (s !== container.parentElement.querySelector('.mem-search-wrap')) s.remove(); });

  let searchWrap = container.parentElement.querySelector('.mem-search-wrap');
  if (!searchWrap) {
    searchWrap = document.createElement('div');
    searchWrap.className = 'mem-search-wrap';
    searchWrap.innerHTML = `
      <svg class="mem-search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input type="text" class="mem-search" placeholder="Search memories…">
    `;
    container.parentElement.insertBefore(searchWrap, container);
  }
  const searchInput = searchWrap.querySelector('.mem-search');
  if (searchInput && !searchInput._bound) {
    searchInput._bound = true;
    searchInput.addEventListener('input', () => {
      _searchQuery = searchInput.value.toLowerCase().trim();
      renderList();
    });
  }
}

// ── Memory list ────────────────────────────────────────────
function renderList() {
  if (!container) return;
  container.innerHTML = '';

  let filtered = _allMems;
  if (_activeType) {
    filtered = filtered.filter(m => (m.kind || 'factual') === _activeType);
  }
  if (_searchQuery) {
    filtered = filtered.filter(m => {
      const content = (m.content || '').toLowerCase();
      const tags = (m.tags || []).join(' ').toLowerCase();
      return content.includes(_searchQuery) || tags.includes(_searchQuery);
    });
  }

  if (!filtered.length) {
    container.innerHTML = `
      <div class="mem-empty">
        <div class="mem-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--tx4)" stroke-width="1.2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
        </div>
        <div class="mem-empty-title">No memories found</div>
        <div class="mem-empty-sub">${_searchQuery ? 'Try a different search term' : 'Hyper Nexus will remember important things as you chat.'}</div>
      </div>`;
    return;
  }

  // Group by type
  const groups = {};
  filtered.forEach(m => {
    const k = m.kind || 'factual';
    if (!groups[k]) groups[k] = [];
    groups[k].push(m);
  });

  const orderedKeys = _activeType
    ? [_activeType]
    : TYPES.filter(t => groups[t]?.length);

  let delay = 0;
  orderedKeys.forEach(type => {
    const items = groups[type] || [];
    if (!items.length) return;
    const meta = TYPE_META[type];

    // Group label
    const gl = document.createElement('div');
    gl.className = 'mem-group-label';
    gl.style.color = meta.color;
    gl.innerHTML = `${meta.icon} ${meta.label} <span class="mem-group-count">${items.length}</span>`;
    container.appendChild(gl);

    items.forEach(m => {
      const imp = Math.round((m.importance || 0.5) * 100);
      const tags = Array.isArray(m.tags) ? m.tags : [];
      const timeStr = m.created_at ? formatTime(m.created_at) : '';

      const card = document.createElement('div');
      card.className = 'mem-card';
      card.dataset.type = type;
      card.dataset.id = m.id;
      card.style.animationDelay = `${delay}ms`;
      delay += 20;

      card.innerHTML = `
        <div class="mem-card-stripe" style="background:${meta.color}"></div>
        <div class="mem-card-body">
          <div class="mem-card-top">
            <span class="mem-badge mem-badge--${type}">${meta.icon} ${meta.label}</span>
            <span class="mem-card-importance">${imp}%</span>
          </div>
          <div class="mem-card-content">${esc(m.content?.substring(0, 300) || '')}</div>
          <div class="mem-card-meta">
            <div class="mem-imp-bar">
              <div class="mem-imp-fill" style="width:0%;background:${meta.color}" data-target="${imp}"></div>
            </div>
            ${tags.length ? `<div class="mem-card-tags">${tags.slice(0, 3).map(t => `<span class="mem-tag">${esc(t)}</span>`).join('')}</div>` : ''}
            ${timeStr ? `<span class="mem-card-time">${timeStr}</span>` : ''}
            <button class="mem-del-btn" data-id="${m.id}" title="Delete memory">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
            </button>
          </div>
        </div>
      `;

      // Delete handler
      const delBtn = card.querySelector('.mem-del-btn');
      if (delBtn) {
        delBtn.addEventListener('click', async (ev) => {
          ev.stopPropagation();
          try {
            await api(`/api/memories/${m.id}`, { method: 'DELETE' });
            card.style.opacity = '0';
            card.style.transform = 'translateX(20px)';
            card.style.transition = 'all 200ms ease';
            setTimeout(() => card.remove(), 200);
            _allMems = _allMems.filter(x => x.id !== m.id);
            renderStatsBar();
            toast('Memory deleted', 'success');
          } catch (e) {
            toast('Delete failed: ' + e.message, 'error');
          }
        });
      }

      container.appendChild(card);

      // Animate importance bar
      requestAnimationFrame(() => {
        const fill = card.querySelector('.mem-imp-fill');
        if (fill) setTimeout(() => { fill.style.width = `${imp}%`; }, delay + 50);
      });
    });
  });
}

// ── Helpers ────────────────────────────────────────────────
function esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str != null ? str : '');
  return d.innerHTML;
}

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(typeof ts === 'number' ? ts * 1000 : ts);
  const now = new Date();
  const diffMs = now - d;
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHrs = Math.floor(diffMins / 60);
  if (diffHrs < 24) return `${diffHrs}h ago`;
  const diffDays = Math.floor(diffHrs / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}
