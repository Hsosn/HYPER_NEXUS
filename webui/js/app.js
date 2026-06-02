/**
 * Nexus WebUI - main application.
 */
import { state } from './state.js';
import { connectWS } from './ws.js';
import { initChat }      from './panels/chat.js';
import { initTools }     from './panels/tools.js';
import { initSkills }    from './panels/skills.js';
import { initMemory }    from './panels/memory.js';
import { initGoals }     from './panels/goals.js';
import { initActivity }  from './panels/activity.js';
import { initMetrics }   from './panels/metrics.js';
import { initSettings }  from './panels/settings.js';
import { initWorkspace }     from './panels/workspace.js';
import { initNotifications } from './panels/notifications.js';
import { initIntegrations, initTriggers, initAutomation } from './panels/integrations.js';
import { initMCP } from './panels/mcp.js';
import { initBrowserLive }   from './panels/browser-live.js';
import { initVirtualComputer } from './panels/virtual-computer.js';
import { initVMLiveViewer }   from './vm-live-viewer.js';
import { api, el, setAuthToken, getAuthToken } from './utils.js';

// ════════════════════════════════════════════════════════════════════════════
// COMMAND PALETTE (Cmd/Ctrl+K) — switch panels + run quick commands
// ════════════════════════════════════════════════════════════════════════════
const PALETTE_COMMANDS = [
  { id: 'panel:chat',            label: '→ Chat',                hint: '1',      keywords: 'chat conversation main' },
  { id: 'panel:tools',           label: '→ Tools',               hint: '2',      keywords: 'tools tool registry' },
  { id: 'panel:skills',          label: '→ Skills',              hint: '3',      keywords: 'skills ml 3d' },
  { id: 'panel:memory',          label: '→ Memory',              hint: '4',      keywords: 'memory memories' },
  { id: 'panel:goals',           label: '→ Goals',               hint: '5',      keywords: 'goals tasks' },
  { id: 'panel:activity',        label: '→ Activity',            hint: '6',      keywords: 'activity log' },
  { id: 'panel:workspace',       label: '→ Workspace',           hint: '7',      keywords: 'workspace files' },
  { id: 'panel:metrics',         label: '→ Metrics',             hint: '8',      keywords: 'metrics tokens cost' },
  { id: 'panel:settings',        label: '→ Settings',            hint: '9',      keywords: 'settings config' },
  { id: 'open:integrations',     label: '→ Integrations',        hint: 'I',      keywords: 'integrations services oauth' },
  { id: 'open:triggers',         label: '→ Triggers',            hint: 'T',      keywords: 'triggers webhooks' },
  { id: 'open:automation',       label: '→ Automation',          hint: 'A',      keywords: 'automation watches monitors schedules' },
  { id: 'open:mcp',              label: '→ MCP Servers',         hint: 'M',      keywords: 'mcp servers model context' },
  { id: 'open:browser',          label: '→ Browser Live View',   hint: 'B',      keywords: 'browser live view' },
  { id: 'open:vm',               label: '→ Virtual Computer',    hint: 'V',      keywords: 'vm virtual computer desktop' },
  { id: 'open:vmviewer',         label: '→ VM Live Viewer',      hint: '⇧V',     keywords: 'vm live viewer vnc' },
  { id: 'open:notifications',    label: '→ Notifications',       hint: 'N',      keywords: 'notifications inbox' },
  { id: 'action:newchat',        label: '+ New Chat',            hint: '⌘N',     keywords: 'new chat session' },
  { id: 'action:clear',          label: '✕ Clear all chats',     hint: '',       keywords: 'clear all delete wipe' },
];

let paletteIndex = 0;
let paletteMatches = [];

function openPalette() {
  const palette = document.getElementById('nx-palette');
  if (!palette) return;
  palette.classList.add('open');
  const input = document.getElementById('nx-palette-input');
  input.value = '';
  paletteMatches = PALETTE_COMMANDS.slice();
  paletteIndex = 0;
  renderPalette();
  setTimeout(() => input.focus(), 0);
}

function closePalette() {
  const palette = document.getElementById('nx-palette');
  if (palette) palette.classList.remove('open');
  const input = document.getElementById('input');
  if (input) input.focus();
}

function renderPalette() {
  const list = document.getElementById('nx-palette-list');
  if (!list) return;
  if (paletteMatches.length === 0) {
    list.innerHTML = '<div class="nx-palette-empty">no matches</div>';
    return;
  }
  list.innerHTML = paletteMatches.map((c, i) => `
    <div class="nx-palette-row ${i === paletteIndex ? 'active' : ''}" data-idx="${i}">
      <span class="nx-row-id">[${c.id}]</span>
      <span>${c.label}</span>
      <span class="nx-row-hint">${c.hint || ''}</span>
    </div>
  `).join('');
  list.querySelectorAll('.nx-palette-row').forEach(row => {
    row.addEventListener('click', () => {
      paletteIndex = Number(row.dataset.idx);
      runPaletteCommand();
    });
  });
}

function filterPalette(q) {
  q = (q || '').toLowerCase().trim();
  if (!q) { paletteMatches = PALETTE_COMMANDS.slice(); }
  else {
    paletteMatches = PALETTE_COMMANDS.filter(c =>
      c.label.toLowerCase().includes(q) ||
      c.id.toLowerCase().includes(q) ||
      (c.keywords || '').toLowerCase().includes(q)
    );
  }
  paletteIndex = 0;
  renderPalette();
}

function runPaletteCommand() {
  const cmd = paletteMatches[paletteIndex];
  if (!cmd) return;
  closePalette();
  const [kind, target] = cmd.id.split(':');
  if (kind === 'panel') {
    const nav = document.querySelector(`.nav-item[data-panel="${target}"]`);
    if (nav) nav.click();
  } else if (kind === 'open') {
    const triggerMap = {
      integrations: 'integrations-trigger',
      triggers: 'triggers-trigger',
      automation: 'automation-trigger',
      mcp: 'mcp-trigger',
      browser: 'browser-live-toggle',
      vm: 'vm-live-toggle', // Note: there are two vm triggers; vm-live-toggle opens live viewer
      vmviewer: 'vm-live-toggle',
      notifications: 'notifications-trigger',
    };
    const btnId = triggerMap[target];
    if (btnId) {
      const btn = document.getElementById(btnId);
      if (btn) btn.click();
    }
  } else if (kind === 'action') {
    if (target === 'newchat') {
      const b = document.getElementById('new-chat');
      if (b) b.click();
    } else if (target === 'clear') {
      const b = document.getElementById('clear-all-chats');
      if (b) b.click();
    }
  }
}

function initPalette() {
  document.addEventListener('keydown', (e) => {
    const palette = document.getElementById('nx-palette');
    const isOpen = palette && palette.classList.contains('open');
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      if (isOpen) closePalette(); else openPalette();
      return;
    }
    if (!isOpen) return;
    if (e.key === 'Escape') { closePalette(); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); paletteIndex = Math.min(paletteIndex + 1, paletteMatches.length - 1); renderPalette(); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); paletteIndex = Math.max(paletteIndex - 1, 0); renderPalette(); }
    if (e.key === 'Enter')     { e.preventDefault(); runPaletteCommand(); }
  });
  const input = document.getElementById('nx-palette-input');
  if (input) input.addEventListener('input', (e) => filterPalette(e.target.value));
  // close on backdrop click
  document.getElementById('nx-palette')?.addEventListener('click', (e) => {
    if (e.target.id === 'nx-palette') closePalette();
  });
}

// ════════════════════════════════════════════════════════════════════════════
// HEADER + STATUS BAR — reflect current state in the new chrome
// ════════════════════════════════════════════════════════════════════════════
function updateHeaderPanel() {
  const cur = document.getElementById('nx-current-panel');
  const st  = document.getElementById('nx-st-panel');
  if (!cur || !st) return;
  const name = (state.activePanel || 'chat').toUpperCase();
  cur.textContent = `[${name}]`;
  st.textContent  = name;
}

function updateStatusBar() {
  // Time
  const t = document.getElementById('nx-st-time');
  if (t) {
    const d = new Date();
    t.textContent = d.toISOString().substring(11, 19);
  }
  // WS state
  const ws = document.getElementById('ws-status');
  const wsLabel = document.getElementById('nx-hdr-ws-label');
  const wsDot   = document.getElementById('nx-hdr-ws');
  if (ws && wsLabel && wsDot) {
    const txt = (ws.textContent || '').toLowerCase();
    const online = txt.includes('connect') || txt === 'connected' || txt === 'online';
    const error  = txt.includes('reconnect') || txt.includes('error');
    wsDot.className = 'nx-status-dot ' + (error ? 'error' : online ? 'online' : 'warn');
    wsLabel.textContent = online ? 'online' : error ? 'reconnect' : 'connecting';
  }
}

function startStatusTicker() {
  updateStatusBar();
  updateHeaderPanel();
  setInterval(updateStatusBar, 1000);
  // Re-sync panel label whenever nav changes
  document.addEventListener('panel:shown', updateHeaderPanel);
}

// ════════════════════════════════════════════════════════════════════════════
// NAVIGATION (extended — also reflects in header + opens sessions drawer)
// ════════════════════════════════════════════════════════════════════════════

// ------- Panel switching -------
function initNav() {
  const navButtons = document.querySelectorAll('.nav-item');
  const panels     = document.querySelectorAll('.panel');
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.panel;
      navButtons.forEach(b => b.classList.toggle('active', b === btn));
      panels.forEach(p => p.classList.toggle('active', p.dataset.panel === target));
      state.activePanel = target;
      window.dispatchEvent(new CustomEvent('panel:shown', { detail: target }));
    });
  });
}

// ------- WS status classes -------
function initWsStatus() {
  const statusEl = document.getElementById('ws-status');
  if (!statusEl) return;
  const observer = new MutationObserver(() => {
    const text = statusEl.textContent;
    document.body.classList.toggle('ws-error', text.includes('reconnect'));
    document.body.classList.toggle('ws-warn',  text.includes('connecting'));
  });
  observer.observe(statusEl, { childList: true, characterData: true, subtree: true });
}

// ------- Login modal -------
async function tryAutoLogin() {
  // Check if auth is even enabled by hitting /api/status (optional_auth)
  try {
    const status = await api('/api/status');
    if (status.auth_enabled && !getAuthToken()) {
      showLoginModal();
      return false;
    }
    return true;
  } catch (e) {
    // If status fails with 401, auth is required
    if (e.message.includes('401')) {
      showLoginModal();
      return false;
    }
    return true; // server error, not auth error
  }
}

function showLoginModal() {
  const root = document.getElementById('modal-root');
  root.innerHTML = '';

  const backdrop = el('div', { class: 'modal-backdrop', style: { zIndex: 9999 } });
  const box = el('div', { class: 'modal', style: { maxWidth: '380px' } });

  const title = el('h2', {}, '🔐 Login to Hyper Nexus');
  const desc = el('p', { style: { color: 'var(--tx2)', fontSize: '13px', marginBottom: '16px' } },
    'Authentication is enabled. Enter your credentials.');

  const userLabel = el('label', { style: { fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '4px' } }, 'Username');
  const userInput = el('input', { type: 'text', value: 'admin', placeholder: 'admin',
    style: { width: '100%', padding: '8px 10px', marginBottom: '12px', borderRadius: '6px',
             border: '1px solid var(--line)', background: 'var(--surface2)', color: 'var(--tx1)' } });

  const passLabel = el('label', { style: { fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '4px' } }, 'Password');
  const passInput = el('input', { type: 'password', placeholder: 'admin',
    style: { width: '100%', padding: '8px 10px', marginBottom: '16px', borderRadius: '6px',
             border: '1px solid var(--line)', background: 'var(--surface2)', color: 'var(--tx1)' } });

  const errorEl = el('div', { style: { color: 'var(--red)', fontSize: '12px', marginBottom: '8px', display: 'none' } });
  const loginBtn = el('button', { class: 'btn primary', style: { width: '100%' } }, 'Login');
  const skipBtn = el('button', { class: 'btn ghost', style: { width: '100%', marginTop: '8px' } }, 'Skip (auth may be disabled)');

  async function doLogin() {
    loginBtn.disabled = true;
    loginBtn.textContent = 'Logging in…';
    try {
      const data = await api('/api/auth/login', {
        method: 'POST',
        body: { username: userInput.value || 'admin', password: passInput.value || 'admin' },
      });
      setAuthToken(data.token);
      backdrop.remove();
      startApp();
    } catch (e) {
      errorEl.textContent = 'Login failed: ' + e.message;
      errorEl.style.display = 'block';
      loginBtn.disabled = false;
      loginBtn.textContent = 'Login';
    }
  }

  loginBtn.addEventListener('click', doLogin);
  passInput.addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
  skipBtn.addEventListener('click', () => { backdrop.remove(); startApp(); });

  box.appendChild(title);
  box.appendChild(desc);
  box.appendChild(userLabel);
  box.appendChild(userInput);
  box.appendChild(passLabel);
  box.appendChild(passInput);
  box.appendChild(errorEl);
  box.appendChild(loginBtn);
  box.appendChild(skipBtn);
  backdrop.appendChild(box);
  root.appendChild(backdrop);
  userInput.focus();
}

// ------- Welcome splash -------
function showWelcomeModal() {
  const root = document.getElementById('modal-root');
  // Don't stack if already open
  if (root.querySelector('.welcome-backdrop')) return;

  const backdrop = el('div', { class: 'modal-backdrop welcome-backdrop' });
  const box = el('div', { class: 'modal', style: { maxWidth: '480px', textAlign: 'center' } });

  const logo = el('div', { style: { fontSize: '32px', marginBottom: '8px' } }, '⚡');
  const brand = el('h2', { style: { margin: '0 0 4px', fontSize: '18px', letterSpacing: '2px' } },
    'CREATED BY VESKO LABS');
  const thanks = el('p', { style: { color: 'var(--tx2)', fontSize: '13px', margin: '0 0 16px' } },
    'Thank you for downloading and using Hyper Nexus.');

  const divider = el('hr', { style: { border: 'none', borderTop: '1px solid var(--line)', margin: '0 0 16px' } });

  const desc = el('div', { style: { fontSize: '13px', lineHeight: '1.6', color: 'var(--tx2)', textAlign: 'left' } },
    el('p', { style: { margin: '0 0 8px' } },
      'Hyper Nexus is an autonomous AI agent platform with adaptive reasoning, long-term semantic memory, 40+ built-in tools, a pure-Python 3D engine, and a real-time Web UI — all in a single codebase.'),
    el('p', { style: { margin: '0' } },
      'It runs entirely on your hardware with no cloud dependency. Explore, automate, and create.')
  );

  const gotItBtn = el('button', {
    class: 'btn primary',
    style: { marginTop: '20px', minWidth: '120px' },
    onclick: () => backdrop.remove()
  }, 'Get Started');

  box.appendChild(logo);
  box.appendChild(brand);
  box.appendChild(thanks);
  box.appendChild(divider);
  box.appendChild(desc);
  box.appendChild(gotItBtn);
  backdrop.appendChild(box);
  root.appendChild(backdrop);

  // Dismiss on click outside
  backdrop.addEventListener('click', e => { if (e.target === backdrop) backdrop.remove(); });
}

// ------- App init -------
async function startApp() {
  console.log('[boot] Starting panels...');

  // Init the new shell chrome (palette, status ticker, nav)
  try {
    initPalette();
    startStatusTicker();
    console.log('[boot] Shell chrome initialized');
  } catch (e) { console.error('[boot] shell init failed:', e); }

  try {
    initNav();
    console.log('[boot] Nav initialized');
  } catch (e) { console.error('[boot] initNav failed:', e); }

  try {
    initWsStatus();
    console.log('[boot] WS status initialized');
  } catch (e) { console.error('[boot] initWsStatus failed:', e); }

  // Connect WebSocket FIRST — this is critical for all real-time features
  try {
    connectWS();
    console.log('[boot] WebSocket connect called');
  } catch (e) { console.error('[boot] connectWS failed:', e); }

  // Panel initializations — run in parallel with Promise.allSettled() so that
  // one failure doesn't block others. Each panel reports its own status.
  const panelInits = [
    ['Chat', () => initChat()],
    ['Tools', () => initTools()],
    ['Skills', () => initSkills()],
    ['Memory', () => initMemory()],
    ['Goals', () => initGoals()],
    ['Activity', () => initActivity()],
    ['Metrics', () => initMetrics()],
    ['Settings', () => initSettings()],
    ['Workspace', () => initWorkspace()],
    ['Notifications', () => initNotifications()],
    ['Integrations', () => initIntegrations()],
    ['Triggers', () => initTriggers()],
    ['Automation', () => initAutomation()],
    ['MCP', () => initMCP()],
    ['BrowserLive', () => initBrowserLive()],
    ['VirtualComputer', () => initVirtualComputer()],
    ['VMLiveViewer', () => initVMLiveViewer()],
  ];

  const initPromises = panelInits.map(([name, initFn]) =>
    Promise.resolve().then(() => initFn()).then(
      () => { console.log(`[boot] ${name} initialized`); },
      (e) => { console.error(`[boot] ${name} init failed:`, e); }
    )
  );
  await Promise.allSettled(initPromises);

  console.log('[boot] All panels initialized!');

  // Auto-focus the omni-input
  setTimeout(() => { try { document.getElementById('input')?.focus(); } catch {} }, 100);

  // Show welcome splash after a short delay so panels have rendered
  setTimeout(showWelcomeModal, 600);
}

async function boot() {
  console.log('[boot] Hyper Nexus WebUI boot starting...');
  try {
    const canProceed = await tryAutoLogin();
    console.log('[boot] Auto-login result:', canProceed);
    if (canProceed) startApp();
  } catch (e) {
    console.error('[boot] CRITICAL boot failure:', e);
    // Attempt to start app anyway as fallback
    try {
      startApp();
    } catch (e2) {
      console.error('[boot] Fallback boot also failed:', e2);
    }
  }
}

// Use DOMContentLoaded to ensure all HTML elements exist
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
