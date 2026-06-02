/**
 * Tools Panel — Professional redesign v3
 * Real icons, clean grid, capability indicators
 */
import { api } from '../utils.js';
import { state } from '../state.js';
import { toast } from '../enhancements.js';

const RISK_CFG = {
  low:    { cls: 'risk-low',    label: 'Low Risk',    color: '#17c964' },
  medium: { cls: 'risk-medium', label: 'Med Risk',    color: '#f5a623' },
  high:   { cls: 'risk-high',   label: 'High Risk',   color: '#f31260' },
};

// Professional SVG icons per category
const CAT_ICONS = {
  web:      `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`,
  file:     `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
  shell:    `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`,
  code:     `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  memory:   `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
  research: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  git:      `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><line x1="6" y1="9" x2="6" y2="21"/></svg>`,
  email:    `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="4" width="20" height="16" rx="2"/><polyline points="2,4 12,13 22,4"/></svg>`,
  system:   `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`,
  browser:  `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>`,
  monitor:  `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
  '3d_engine':`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
  general:  `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93A10 10 0 1 0 4.93 19.07 10 10 0 0 0 19.07 4.93z"/></svg>`,
};

const grid   = document.getElementById('tools-grid');
const addBtn = document.getElementById('add-tool');

export async function initTools() {
  await refresh();
  addBtn?.addEventListener('click', showCreateModal);
  window.addEventListener('panel:shown', e => { if (e.detail === 'tools') refresh(); });

  // Refresh when tools are added/removed via WebSocket
  window.addEventListener('ws:event', e => {
    const msg = e.detail;
    if (msg.kind === 'custom_tool_added') refresh();
  });

  const searchEl = document.getElementById('tools-search');
  searchEl?.addEventListener('input', () => filterCards(searchEl.value));
}

async function refresh() {
  if (!grid) return;
  grid.innerHTML = `<div class="tx-skeleton-grid">${Array(6).fill('<div class="tx-skeleton"></div>').join('')}</div>`;

  try {
    const [tools, custom] = await Promise.all([
      api('/api/tools').catch(() => []),
      api('/api/custom_tools').catch(() => []),
    ]);
    state.toolsCount = tools.length;
    renderTools(tools, custom);
  } catch (e) {
    grid.innerHTML = `<div class="tx-empty">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".4"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <p>Failed to load tools</p><span>${_esc(e.message)}</span></div>`;
  }
}

function renderTools(tools, custom) {
  grid.innerHTML = '';
  if (!tools.length) {
    grid.innerHTML = `<div class="tx-empty">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".4"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
      <p>No tools loaded</p></div>`;
    return;
  }

  // Stats bar
  const customSet = new Set(custom.map(c => c.name));
  const highRiskN = tools.filter(t => t.risk === 'high').length;
  const statsBar  = el('div', { class: 'tx-stats' });
  statsBar.innerHTML = `
    <span><b>${tools.length}</b> tools available</span>
    <span class="tx-dot">·</span>
    <span><b>${custom.length}</b> custom</span>
    <span class="tx-dot">·</span>
    <span class="${highRiskN ? 'tx-risk-warn' : ''}"><b>${highRiskN}</b> high-risk</span>`;
  grid.appendChild(statsBar);

  // Group by category
  const byCategory = {};
  tools.forEach(t => {
    const cat = (t.category || 'general').toLowerCase();
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push(t);
  });

  const cardsWrap = el('div', { class: 'tx-grid', id: 'tx-cards' });

  Object.entries(byCategory).sort(([a],[b]) => a.localeCompare(b)).forEach(([cat, catTools]) => {
    const section = el('div', { class: 'tx-category' });
    const hdr = el('div', { class: 'tx-cat-header' });
    hdr.innerHTML = `
      <span class="tx-cat-icon">${CAT_ICONS[cat] || CAT_ICONS.general}</span>
      <span class="tx-cat-label">${cat}</span>
      <span class="tx-cat-count">${catTools.length}</span>`;
    section.appendChild(hdr);

    const cardsRow = el('div', { class: 'tx-cards-row' });
    catTools.forEach((t, i) => {
      const isCustom = customSet.has(t.name);
      const risk = RISK_CFG[t.risk] || RISK_CFG.low;
      const card = el('div', { class: 'tx-card', 'data-name': (t.name||'').toLowerCase(), 'data-cat': cat });
      card.style.animationDelay = Math.min(i, 12) * 22 + 'ms';
      card.innerHTML = `
        <div class="tx-card-top">
          <span class="tx-card-icon">${CAT_ICONS[cat] || CAT_ICONS.general}</span>
          <div class="tx-card-badges">
            <span class="tx-risk ${risk.cls}">${risk.label}</span>
            ${isCustom ? '<span class="tx-badge-custom">Custom</span>' : ''}
          </div>
        </div>
        <div class="tx-card-name">${_esc(t.name)}</div>
        <div class="tx-card-desc">${_esc((t.description||'').slice(0,100))}</div>`;
      cardsRow.appendChild(card);
    });

    section.appendChild(cardsRow);
    cardsWrap.appendChild(section);
  });

  grid.appendChild(cardsWrap);
}

function filterCards(q) {
  const term = q.toLowerCase();
  document.querySelectorAll('#tx-cards .tx-card').forEach(card => {
    const match = !term || card.dataset.name?.includes(term) || card.dataset.cat?.includes(term);
    card.style.display = match ? '' : 'none';
  });
}

function showCreateModal() {
  const root     = document.getElementById('modal-root') || document.body;
  const backdrop = el('div', { class: 'modal-backdrop' });
  backdrop.innerHTML = `
    <div class="modal" style="max-width:520px">
      <div class="modal-header">
        <div class="modal-title">New Custom Tool</div>
        <button class="modal-close" id="ct-close" type="button">✕</button>
      </div>
      <div class="modal-body" style="display:flex;flex-direction:column;gap:12px">
        <div class="mfield"><label>Tool Name <span class="req">*</span></label>
          <input type="text" id="ct-name" placeholder="my_custom_tool" autocomplete="off"/></div>
        <div class="mfield"><label>Description</label>
          <input type="text" id="ct-desc" placeholder="What does this tool do?"/></div>
        <div class="mfield"><label>Schema (JSON)</label>
          <textarea id="ct-schema" rows="4" placeholder='{"type":"object","properties":{"query":{"type":"string"}}}'></textarea></div>
        <div class="mfield"><label>Code (Python)</label>
          <textarea id="ct-code" rows="6" placeholder="def run(query: str) -> str:\n    return f'Result: {query}'"></textarea></div>
      </div>
      <div class="modal-footer">
        <button class="topbar-btn" id="ct-cancel" type="button">Cancel</button>
        <button class="topbar-btn accent" id="ct-save" type="button">Create Tool</button>
      </div>
    </div>`;
  root.appendChild(backdrop);

  const close = () => backdrop.remove();
  backdrop.querySelector('#ct-close').addEventListener('click', close);
  backdrop.querySelector('#ct-cancel').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });

  backdrop.querySelector('#ct-save').addEventListener('click', async () => {
    const name   = backdrop.querySelector('#ct-name').value.trim();
    const desc   = backdrop.querySelector('#ct-desc').value.trim();
    const schema = backdrop.querySelector('#ct-schema').value.trim();
    const code   = backdrop.querySelector('#ct-code').value.trim();
    if (!name) { toast('Name required', 'warn'); return; }
    let parsedSchema = {};
    try { parsedSchema = schema ? JSON.parse(schema) : {}; } catch { toast('Invalid JSON schema', 'error'); return; }
    try {
      await api('/api/custom_tools', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ name, description: desc, schema: parsedSchema, code }),
      });
      toast('Tool created ✓', 'success');
      close(); refresh();
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
  });
}

function _esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s ?? '');
  return d.innerHTML;
}
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (text) e.textContent = text;
  return e;
}

