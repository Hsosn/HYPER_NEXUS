/**
 * Skills Panel — with custom skill upload support
 * Two-column card grid, working toggles, proper category filtering, upload section
 */
import { api, el } from '../utils.js';
import { toast } from '../enhancements.js';

const CATS = {
  all:         { label: 'All',         color: '#6366f1' },
  ai_media:    { label: 'AI & Media',  color: '#7c3aed' },
  documents:   { label: 'Documents',   color: '#0891b2' },
  development: { label: 'Development', color: '#059669' },
  research:    { label: 'Research',    color: '#d97706' },
  content:     { label: 'Content',     color: '#db2777' },
  finance:     { label: 'Finance',     color: '#dc2626' },
  automation:  { label: 'Automation',  color: '#0284c7' },
  '3d_engine': { label: '3D Engine',   color: '#9333ea' },
  custom:      { label: 'Custom',      color: '#f59e0b' },
};

let allSkills = [];
let activeTab = 'all';
let searchQ   = '';

export async function initSkills() {
  const panel = document.querySelector('.panel[data-panel="skills"]');
  if (!panel) return;
  panel.innerHTML = '';

  // ── Topbar ──
  panel.insertAdjacentHTML('beforeend', `
    <div class="panel-topbar">
      <div class="panel-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
        Skills
        <span class="sk2-count" id="sk2-count">…</span>
      </div>
      <div class="panel-actions">
        <div class="sk2-search-wrap">
          <svg class="sk2-search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" id="sk2-search" class="sk2-search-input" placeholder="Search skills…" autocomplete="off"/>
        </div>
      </div>
    </div>`);

  // ── Upload Section ──
  const uploadSection = document.createElement('div');
  uploadSection.className = 'sk2-upload-section';
  uploadSection.innerHTML = `
    <div class="sk2-upload-header">
      <span class="sk2-upload-label">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Upload Custom Skill
      </span>
    </div>
    <div class="sk2-upload-dropzone" id="sk2-dropzone">
      <input type="file" id="sk2-file-input" accept=".md,.zip,.skill" hidden />
      <div class="sk2-drop-content">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><polyline points="9 15 12 12 15 15"/></svg>
        <p class="sk2-drop-text">Drop <strong>SKILL.md</strong> or <strong>.zip</strong> here</p>
        <p class="sk2-drop-sub">or click to browse</p>
      </div>
    </div>
    <div class="sk2-upload-progress" id="sk2-upload-progress" style="display:none;">
      <div class="sk2-progress-bar"><div class="sk2-progress-fill" id="sk2-progress-fill"></div></div>
      <span class="sk2-progress-text" id="sk2-progress-text">Uploading…</span>
    </div>
  `;
  panel.appendChild(uploadSection);

  // ── Upload Event Handlers ──
  const dropzone = document.getElementById('sk2-dropzone');
  const fileInput = document.getElementById('sk2-file-input');

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('sk2-drop-active');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('sk2-drop-active');
  });
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('sk2-drop-active');
    const files = e.dataTransfer.files;
    if (files.length) handleUpload(files[0]);
  });
  fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleUpload(e.target.files[0]);
  });

  // ── Category tabs ──
  const tabBar = document.createElement('div');
  tabBar.className = 'sk2-tabs';
  Object.entries(CATS).forEach(([key, cfg]) => {
    const b = document.createElement('button');
    b.className = 'sk2-tab' + (key === activeTab ? ' active' : '');
    b.dataset.tab = key;
    b.textContent = cfg.label;
    b.style.setProperty('--tab-color', cfg.color);
    b.addEventListener('click', () => {
      activeTab = key;
      tabBar.querySelectorAll('.sk2-tab').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      render();
    });
    tabBar.appendChild(b);
  });
  panel.appendChild(tabBar);

  // ── Grid ──
  const grid = document.createElement('div');
  grid.id = 'sk2-grid';
  grid.className = 'sk2-grid';
  panel.appendChild(grid);

  // ── Events ──
  document.getElementById('sk2-search').addEventListener('input', e => {
    searchQ = e.target.value.toLowerCase();
    render();
  });
  window.addEventListener('panel:shown', e => { if (e.detail === 'skills') load(); });

  // Refresh when skills are toggled via WebSocket
  window.addEventListener('ws:event', e => {
    const msg = e.detail;
    if (msg.kind === 'settings_updated' && msg.data && msg.data.updates && msg.data.updates.includes('skills_enabled_state')) {
      load();
    }
    if (msg.kind === 'custom_skill_added' || msg.kind === 'custom_skill_removed') {
      load();
      toast(msg.kind === 'custom_skill_added' ? 'Custom skill installed!' : 'Custom skill removed', 'info');
    }
  });

  await load();
}

async function handleUpload(file) {
  const progressEl = document.getElementById('sk2-upload-progress');
  const progressFill = document.getElementById('sk2-progress-fill');
  const progressText = document.getElementById('sk2-progress-text');

  const fname = file.name.toLowerCase();
  if (!fname.endsWith('.md') && !fname.endsWith('.zip') && !fname.endsWith('.skill')) {
    toast('Only SKILL.md or .zip files are accepted', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    toast('File too large (max 10MB)', 'error');
    return;
  }

  progressEl.style.display = 'flex';
  progressFill.style.width = '20%';
  progressText.textContent = 'Uploading…';

  try {
    const formData = new FormData();
    formData.append('file', file);

    progressFill.style.width = '50%';
    progressText.textContent = 'Processing…';

    const result = await api('/api/skills/upload', {
      method: 'POST',
      body: formData,
    });

    progressFill.style.width = '100%';
    progressText.textContent = 'Done!';

    if (result.results && result.results.length > 0) {
      // Zip upload with multiple skills
      const ok = result.results.filter(r => r.ok).length;
      toast(`Installed ${ok} skill(s) from zip`, 'success');
    } else {
      toast(`Skill "${result.name}" installed!`, 'success');
    }

    // Reload skills
    await load();
  } catch (err) {
    progressFill.style.width = '0%';
    progressText.textContent = 'Failed';
    toast('Upload failed: ' + (err.message || 'Unknown error'), 'error');
  }

  setTimeout(() => {
    progressEl.style.display = 'none';
  }, 2000);
}

async function load() {
  const grid = document.getElementById('sk2-grid');
  if (!grid) return;
  grid.innerHTML = `<div class="sk2-loading">${'<div class="sk2-skel"></div>'.repeat(8)}</div>`;
  try {
    allSkills = await api('/api/skills');
    if (!Array.isArray(allSkills)) allSkills = [];
    render();
  } catch (e) {
    grid.innerHTML = `<div class="sk2-empty"><span>⚠️</span><p>Failed to load skills</p><small>${e.message}</small></div>`;
  }
}

function render() {
  const grid = document.getElementById('sk2-grid');
  const countEl = document.getElementById('sk2-count');
  if (!grid) return;

  const filtered = allSkills.filter(s => {
    const catOk  = activeTab === 'all' || (s.category||'').toLowerCase() === activeTab;
    const termOk = !searchQ || s.name.toLowerCase().includes(searchQ) || (s.description||'').toLowerCase().includes(searchQ);
    return catOk && termOk;
  });

  const enabled = allSkills.filter(s => s.enabled !== false).length;
  if (countEl) countEl.textContent = `${enabled}/${allSkills.length} enabled`;

  grid.innerHTML = '';

  if (!filtered.length) {
    grid.innerHTML = `<div class="sk2-empty"><span>🔍</span><p>No skills match</p></div>`;
    return;
  }

  filtered.forEach((s, i) => {
    const enabled = s.enabled !== false;
    const cat = (s.category || 'general').toLowerCase();
    const cfg = CATS[cat] || { color: '#6366f1' };
    const isCustom = s.is_custom;

    const card = document.createElement('div');
    card.className = 'sk2-card' + (enabled ? '' : ' sk2-off') + (isCustom ? ' sk2-custom' : '');
    card.style.animationDelay = Math.min(i, 20) * 20 + 'ms';

    const icon = s.icon || (isCustom ? 'PLG' : '🔧');

    card.innerHTML = `
      <div class="sk2-card-top">
        <span class="sk2-icon">${esc(icon)}</span>
        <div class="sk2-card-actions">
          ${isCustom ? `<button class="sk2-delete-btn" title="Delete skill" data-name="${esc(s.name)}" data-id="${s.custom_skill_id || ''}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>` : ''}
          <label class="sk2-toggle" title="${enabled ? 'Disable' : 'Enable'}">
            <input type="checkbox" ${enabled ? 'checked' : ''}>
            <span class="sk2-rail"><span class="sk2-thumb"></span></span>
          </label>
        </div>
      </div>
      <div class="sk2-name">${esc(s.name)}${isCustom ? ' <span class="sk2-custom-badge">custom</span>' : ''}</div>
      <div class="sk2-desc">${esc((s.description || '—').slice(0, 90))}</div>
      <div class="sk2-footer">
        <span class="sk2-cat-chip" style="--chip-color:${cfg.color}">${cat.replace(/_/g,' ')}</span>
      </div>`;

    // Toggle handler
    card.querySelector('input[type="checkbox"]').addEventListener('change', async e => {
      const checked = e.target.checked;
      try {
        await api(`/api/skills/${encodeURIComponent(s.name)}/toggle`, { method: 'POST' });
        s.enabled = checked;
        card.classList.toggle('sk2-off', !checked);
        const countEl = document.getElementById('sk2-count');
        if (countEl) {
          const en = allSkills.filter(x => x.enabled !== false).length;
          countEl.textContent = `${en}/${allSkills.length} enabled`;
        }
        toast(`${s.name} ${checked ? 'enabled' : 'disabled'}`, checked ? 'success' : 'info');
      } catch (err) {
        toast('Toggle failed: ' + err.message, 'error');
        e.target.checked = !checked;
        card.classList.toggle('sk2-off', checked);
      }
    });

    // Delete handler for custom skills
    const deleteBtn = card.querySelector('.sk2-delete-btn');
    if (deleteBtn) {
      deleteBtn.addEventListener('click', async e => {
        e.stopPropagation();
        const skillName = deleteBtn.dataset.name;
        const skillId = deleteBtn.dataset.id;
        if (!confirm(`Delete custom skill "${skillName}"?`)) return;
        try {
          await api(`/api/skills/custom/${skillId}`, { method: 'DELETE' });
          toast(`Skill "${skillName}" deleted`, 'info');
          await load();
        } catch (err) {
          toast('Delete failed: ' + err.message, 'error');
        }
      });
    }

    grid.appendChild(card);
  });
}

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
