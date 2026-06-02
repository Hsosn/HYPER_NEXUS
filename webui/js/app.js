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
import { initIntegrations, initTriggers } from './panels/integrations.js';
import { initMCP } from './panels/mcp.js';
import { initBrowserLive }   from './panels/browser-live.js';
import { initVirtualComputer } from './panels/virtual-computer.js';
import { initVMLiveViewer }   from './vm-live-viewer.js';
import { initParticles } from './particles.js';
import { api, el, setAuthToken, getAuthToken } from './utils.js';

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

  // Init sci-fi particle background
  try {
    initParticles();
    console.log('[boot] Particle network initialized');
  } catch (e) { console.error('[boot] initParticles failed:', e); }

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
