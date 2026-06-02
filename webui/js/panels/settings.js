import { api, el } from '../utils.js';
import { state } from '../state.js';
import { toast } from '../enhancements.js';

const form    = document.getElementById('settings-form');
const saveBtn = document.getElementById('save-settings');

// ── Section definitions ─────────────────────────────────────────────────────
const SECTIONS = [
  {
    id: 'api', title: 'API & Provider',
    desc: 'Credentials and endpoint for the language model',
    color: '#5b5bd6',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0 3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>`,
    fields: [
      {
        key: 'provider', label: 'Provider', type: 'select',
        options: ['openrouter', 'openai', 'together', 'nvidia', 'custom', 'groq'],
        hint: 'openrouter = OpenRouter.ai · openai = Direct OpenAI · together = Together AI · nvidia = NVIDIA NIM · custom = Any OpenAI-compatible endpoint · groq = Groq LPU Inference',
      },
      { key: 'openrouter_api_key', label: 'OpenRouter API Key', type: 'password', hint: 'Get yours at openrouter.ai/keys', section: 'openrouter' },
      { key: 'openrouter_base_url', label: 'OpenRouter Base URL', type: 'text', section: 'openrouter' },
      { key: 'openai_api_key', label: 'OpenAI API Key', type: 'password', hint: 'sk-...', section: 'openai' },
      { key: 'openai_base_url', label: 'OpenAI Base URL', type: 'text', hint: 'Default: https://api.openai.com/v1', section: 'openai' },
      { key: 'openai_org', label: 'OpenAI Organization ID', type: 'text', hint: 'Optional', section: 'openai' },
      { key: 'together_api_key', label: 'Together AI API Key', type: 'password', hint: 'Get yours at api.together.xyz', section: 'together' },
      { key: 'together_base_url', label: 'Together AI Base URL', type: 'text', hint: 'Default: https://api.together.xyz/v1', section: 'together' },
      { key: 'groq_api_key', label: 'Groq API Key', type: 'password', hint: 'Get yours at console.groq.com/keys', section: 'groq' },
      { key: 'groq_base_url', label: 'Groq Base URL', type: 'text', hint: 'Default: https://api.groq.com/openai/v1', section: 'groq' },
      { key: 'nvidia_api_key', label: 'NVIDIA API Key', type: 'password', hint: 'Get yours at build.nvidia.com', section: 'nvidia' },
      { key: 'nvidia_base_url', label: 'NVIDIA Base URL', type: 'text', hint: 'Default: https://integrate.api.nvidia.com/v1', section: 'nvidia' },
      { key: 'custom_base_url', label: 'Custom Endpoint URL', type: 'text', hint: 'e.g. http://localhost:11434/v1 or https://xxx.openai.azure.com/...', section: 'custom' },
      { key: 'custom_api_key', label: 'Custom API Key', type: 'password', hint: 'Leave empty if no auth needed', section: 'custom' },
      { key: 'custom_auth_type', label: 'Auth Type', type: 'select', options: ['bearer', 'basic', 'apikey'], hint: 'bearer = Authorization: Bearer … · apikey = X-API-Key header', section: 'custom' },
      { key: 'custom_no_auth', label: 'No Auth Required', type: 'bool', hint: 'Enable for local endpoints (LM Studio, etc.)', section: 'custom' },
      { key: 'default_model', label: 'Default Model', type: 'model_select', hint: 'Model used for all agent work' },
    ],
  },
  {
    id: 'agents', title: 'Agent Models',
    desc: 'Model configuration for the agent',
    color: '#06b6d4',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>`,
    fields: [
      { key: 'memory_model',   label: 'Memory Model',   type: 'model_select' },
    ],
  },
  {
    id: 'generation', title: 'Generation',
    desc: 'Sampling and token budget for LLM output',
    color: '#f5a623',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
    fields: [
      { key: 'temperature', label: 'Temperature', type: 'number', hint: '0 = deterministic · 1 = creative' },
      { key: 'max_tokens',  label: 'Max Tokens',  type: 'number', hint: 'Max output tokens per request' },
      { key: 'top_p',       label: 'Top-p',       type: 'number', hint: 'Nucleus sampling (0–1)' },
    ],
  },
  {
    id: 'identity', title: 'Identity',
    desc: 'Agent name, personality, traits, and communication style',
    color: '#17c964',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>`,
    fields: [
      { key: 'agent_name',          label: 'Agent Name',          type: 'text' },
      { key: 'personality',         label: 'Personality Prompt',  type: 'textarea' },
      { key: 'traits',              label: 'Traits',              type: 'csv',  hint: 'Comma-separated: curious, precise, helpful' },
      { key: 'communication_style', label: 'Communication Style', type: 'text' },
    ],
  },
  {
    id: 'reasoning', title: 'Reasoning',
    desc: 'How the agent thinks, plans, and executes multi-step tasks',
    color: '#8b5cf6',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/><line x1="9" y1="21" x2="15" y2="21"/></svg>`,
    fields: [
      { key: 'reasoning_mode',         label: 'Reasoning Strategy',   type: 'text', hint: 'Nexus Framework — adaptive (always active, cannot be changed)' },
      { key: 'enable_planning',        label: 'Enable Planning',       type: 'bool' },
      { key: 'enable_self_reflection', label: 'Self-Reflection',       type: 'bool' },
      // Single-agent architecture — no multi-agent settings
      { key: 'max_token_budget',       label: 'Token Budget (0=∞)',    type: 'number' },
      { key: 'step_timeout_seconds',   label: 'Step Timeout (sec)',    type: 'number' },
    ],
  },
  {
    id: 'memory', title: 'Memory',
    desc: 'Long-term memory, vector retrieval, and embedding settings',
    color: '#0070f3',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
    fields: [
      { key: 'enable_long_term_memory',     label: 'Enable Long-Term Memory', type: 'bool' },
      { key: 'memory_importance_threshold', label: 'Importance Threshold',    type: 'number', hint: '0–1 · lower = remember more' },
      { key: 'memory_retrieval_k',          label: 'Retrieval K',             type: 'number', hint: 'How many memories to retrieve per query' },
    ],
  },
  {
    id: 'heartbeat', title: 'Heartbeat',
    desc: 'Background loop — scheduling, goal monitoring, environment checks, data integrity',
    color: '#ef4444',
    icon: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
    fields: [
      { key: 'enable_heartbeat',           label: 'Enable Heartbeat',     type: 'bool' },
      { key: 'heartbeat_interval_seconds', label: 'Interval (seconds)',   type: 'number', hint: 'Minimum 5 — lower = more responsive but more CPU' },
    ],
  },
];

let cachedModels = [];
let activeSection = 'api';
let currentSettings = {};

export async function initSettings() {
  currentSettings = await api('/api/settings');
  state.settings = currentSettings;
  try {
    const { models } = await api('/api/models');
    cachedModels = models.map(m => m.id).sort();
  } catch { cachedModels = []; }
  render(currentSettings);
  saveBtn?.addEventListener('click', save);
}

// ── Render layout ──────────────────────────────────────────────────────────

function render(s) {
  form.innerHTML = '';
  form.className = 'sx-layout';

  // Sidebar nav
  const nav = el('nav', { class: 'sx-nav', role: 'navigation', 'aria-label': 'Settings sections' });
  SECTIONS.forEach(sec => {
    const item = el('button', {
      class: 'sx-nav-item' + (sec.id === activeSection ? ' active' : ''),
      'data-sec': sec.id, type: 'button',
    });
    item.innerHTML = `
      <span class="sx-nav-icon" style="color:${sec.color}">${sec.icon}</span>
      <span class="sx-nav-label">${sec.title}</span>
    `;
    item.addEventListener('click', () => {
      nav.querySelectorAll('.sx-nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      activeSection = sec.id;
      document.querySelectorAll('.sx-section').forEach(e => {
        e.style.display = e.dataset.sec === sec.id ? '' : 'none';
      });
      // Update provider-specific field visibility
      if (sec.id === 'api') updateProviderVisibility();
    });
    nav.appendChild(item);
  });

  const content = el('div', { class: 'sx-content' });

  SECTIONS.forEach(sec => {
    const section = el('section', { class: 'sx-section', 'data-sec': sec.id, 'aria-label': sec.title });
    section.style.display = sec.id === activeSection ? '' : 'none';

    // Header
    const hdr = el('div', { class: 'sx-section-header' });
    hdr.innerHTML = `
      <span class="sx-section-icon" style="color:${sec.color}">${sec.icon}</span>
      <div>
        <div class="sx-section-title">${sec.title}</div>
        <div class="sx-section-desc">${sec.desc}</div>
      </div>`;
    section.appendChild(hdr);

    // Fields
    const grid = el('div', { class: 'sx-fields' });
    sec.fields.forEach(f => grid.appendChild(renderField(f, s[f.key])));
    section.appendChild(grid);
    content.appendChild(section);
  });

  form.appendChild(nav);
  form.appendChild(content);

  // Initial provider visibility
  updateProviderVisibility();

  // Wire provider select to show/hide relevant fields and refresh models
  const providerSel = form.querySelector('[data-key="provider"]');
  if (providerSel) {
    providerSel.addEventListener('change', async () => {
      updateProviderVisibility();
      // Fetch models for the new provider after saving
    });
  }

  // Also refresh models when settings are saved
  const saveBtn = document.getElementById('sx-save');
  if (saveBtn) {
    const originalSave = saveBtn.onclick;
    saveBtn.addEventListener('click', async () => {
      await save();
      // Refresh models after save to get new provider's models
      try {
        const { models } = await api('/api/models');
        cachedModels = models.map(m => m.id).sort();
        // Re-render model select fields with new models
        render(currentSettings);
      } catch {}
    });
  }
}

function updateProviderVisibility() {
  const providerEl = form.querySelector('[data-key="provider"]');
  const provider   = providerEl ? providerEl.value : (currentSettings.provider || 'openrouter');

  form.querySelectorAll('[data-provider-section]').forEach(row => {
    const sec = row.dataset.providerSection;
    const show = !sec || sec === provider;
    row.style.display = show ? '' : 'none';
  });
}

// ── Field renderer ──────────────────────────────────────────────────────────

function renderField(f, value) {
  const wrap = el('div', { class: 'sx-field' });
  if (f.section) {
    wrap.dataset.providerSection = f.section;
  }

  const labelEl = el('label', { class: 'sx-label', for: `set-${f.key}` });
  labelEl.textContent = f.label;

  let input;

  if (f.type === 'bool') {
    const row = el('div', { class: 'sx-bool-row' });
    row.appendChild(labelEl);
    const toggle = el('label', { class: 'sx-toggle' });
    const cb = el('input', { type: 'checkbox', id: `set-${f.key}`, 'data-key': f.key });
    if (value) cb.checked = true;
    toggle.appendChild(cb);
    toggle.appendChild(el('span', { class: 'sx-slider' }));
    row.appendChild(toggle);
    wrap.appendChild(row);
    if (f.hint) wrap.appendChild(el('p', { class: 'sx-hint' }, f.hint));
    return wrap;
  }

  wrap.appendChild(labelEl);

  if (f.type === 'textarea') {
    input = el('textarea', { id: `set-${f.key}`, 'data-key': f.key, rows: 5 });
    input.value = Array.isArray(value) ? value.join('\n') : (value || '');
  } else if (f.type === 'select') {
    input = el('select', { id: `set-${f.key}`, 'data-key': f.key });
    f.options.forEach(o => {
      const opt = el('option', { value: o });
      opt.textContent = o;
      if (o === value) opt.selected = true;
      input.appendChild(opt);
    });
  } else if (f.type === 'model_select') {
    // Wrapper for dropdown + refresh button
    const modelWrap = el('div', { class: 'sx-model-wrap' });
    input = el('select', { id: `set-${f.key}`, 'data-key': f.key });
    const opts = cachedModels.length ? cachedModels : [value || ''].filter(Boolean);
    const currentVal = value || '';
    if (!opts.includes(currentVal) && currentVal) opts.unshift(currentVal);
    opts.forEach(o => {
      const opt = el('option', { value: o });
      opt.textContent = o;
      if (o === currentVal) opt.selected = true;
      input.appendChild(opt);
    });
    // Refresh button
    const refreshBtn = el('button', {
      type: 'button',
      class: 'sx-model-refresh',
      title: 'Refresh model list',
    }, '↻');
    refreshBtn.addEventListener('click', async () => {
      refreshBtn.textContent = '…';
      try {
        const { models } = await api('/api/models');
        cachedModels = models.map(m => m.id).sort();
        render(currentSettings);
      } catch {
        refreshBtn.textContent = '↻';
      }
    });
    modelWrap.appendChild(input);
    modelWrap.appendChild(refreshBtn);
    input = modelWrap;
    // Manual input fallback
    const manualWrap = el('div', { class: 'sx-model-manual' });
    const manualInput = el('input', { type: 'text', placeholder: 'Or type a model name…', 'data-key': `${f.key}_manual` });
    manualInput.addEventListener('input', () => {
      if (manualInput.value.trim()) {
        input.value = '';  // deselect dropdown
      }
    });
    manualWrap.appendChild(manualInput);
    wrap.appendChild(input);
    wrap.appendChild(manualWrap);
    if (f.hint) wrap.appendChild(el('p', { class: 'sx-hint' }, f.hint));
    return wrap;
  } else if (f.type === 'csv') {
    const arrVal = Array.isArray(value) ? value.join(', ') : (value || '');
    input = el('input', { type: 'text', id: `set-${f.key}`, 'data-key': f.key, value: arrVal, placeholder: f.hint || '' });
  } else if (f.type === 'number') {
    input = el('input', { type: 'number', id: `set-${f.key}`, 'data-key': f.key,
      value: value != null ? String(value) : '', step: 'any', min: '0' });
  } else if (f.type === 'password') {
    input = el('input', { type: 'password', id: `set-${f.key}`, 'data-key': f.key,
      value: value || '', autocomplete: 'new-password', spellcheck: 'false' });
    // Eye toggle
    const row = el('div', { class: 'sx-password-row' });
    row.appendChild(input);
    const eye = el('button', { type: 'button', class: 'sx-eye', title: 'Toggle visibility' });
    eye.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
    eye.addEventListener('click', () => {
      input.type = input.type === 'password' ? 'text' : 'password';
    });
    row.appendChild(eye);
    wrap.appendChild(labelEl);
    wrap.appendChild(row);
    if (f.hint) wrap.appendChild(el('p', { class: 'sx-hint' }, f.hint));
    return wrap;
  } else {
    input = el('input', { type: 'text', id: `set-${f.key}`, 'data-key': f.key,
      value: value != null ? String(value) : '', placeholder: f.hint || '' });
  }

  wrap.appendChild(input);
  if (f.hint) wrap.appendChild(el('p', { class: 'sx-hint' }, f.hint));
  return wrap;
}

// ── Save ───────────────────────────────────────────────────────────────────

async function save() {
  const _API_KEY_FIELDS = new Set([
    'openrouter_api_key', 'openai_api_key', 'nvidia_api_key', 'custom_api_key', 'together_api_key',
  ]);
  const patch = {};
  form.querySelectorAll('[data-key]').forEach(inp => {
    const key = inp.dataset.key;
    if (!key || key.endsWith('_manual')) return;
    
    // FIX: Skip API key fields that still show masked values (bullets from GET).
    // If the value is mostly bullets, the user didn't change it — don't send it.
    if (_API_KEY_FIELDS.has(key)) {
      const val = inp.value || '';
      if (val && (val.replace(/•/g, '').length < 5)) {
        return;  // Mostly masked — skip to preserve existing key on server
      }
    }
    
    if (inp.type === 'checkbox') {
      patch[key] = inp.checked;
    } else if (inp.tagName === 'TEXTAREA') {
      patch[key] = inp.value;
    } else if (inp.type === 'number') {
      const v = parseFloat(inp.value);
      if (!isNaN(v)) patch[key] = v;
    } else if (_API_KEY_FIELDS.has(key)) {
      // FIX: Always send API key values (including empty to clear), unless masked
      const val = inp.value || '';
      if (val && !val.includes('•')) {
        patch[key] = val;  // Real key value (no bullets)
      }
      // Empty string or masked = don't send (preserve existing)
    } else {
      // For model_select: prefer manual text input if filled
      const manualEl = form.querySelector(`[data-key="${key}_manual"]`);
      const v = manualEl?.value.trim() || inp.value;
      if (v !== '') patch[key] = v;
    }
  });

  saveBtn.disabled = true;
  saveBtn.textContent = 'Saving…';
  const oldProvider = currentSettings.provider;
  try {
    await api('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates: patch }),
    });
    state.settings = { ...state.settings, ...patch };
    currentSettings = { ...currentSettings, ...patch };

    // Refresh models if provider changed
    const newProvider = patch.provider || oldProvider;
    if (newProvider !== oldProvider || patch.default_model) {
      try {
        const { models } = await api('/api/models');
        cachedModels = models.map(m => m.id).sort();
        render(currentSettings);
      } catch {}
    }

    toast('Settings saved ✓', 'success');
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = 'Save Settings';
  }
}
