/**
 * UI Enhancements v3: toasts, typing indicator, copy buttons, skeletons,
 * confirm dialog, and premium floating thinking animation.
 */

import { showThinkingBlock } from './thinking.js';

// ── Toast ─────────────────────────────────────────────────────────────────
let _tc = null;
function getTC() {
  if (!_tc) {
    _tc = document.getElementById('toast-container');
    if (!_tc) {
      _tc = document.createElement('div');
      _tc.id = 'toast-container';
      document.body.appendChild(_tc);
    }
  }
  return _tc;
}

const TOAST_ICONS = { success: '✓', error: '✕', warn: '⚠', info: 'ℹ' };

export function toast(message, type = 'info', duration = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `
    <span class="toast-icon">${(TOAST_ICONS[type] || TOAST_ICONS.info)}</span>
    <span class="toast-msg">${message}</span>`;
  getTC().appendChild(el);

  const dismiss = () => {
    el.classList.add('exiting');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  };
  const t = setTimeout(dismiss, duration);
  el.addEventListener('click', () => { clearTimeout(t); dismiss(); });
  return el;
}

// ── Confirm dialog ─────────────────────────────────────────────────────────
export function confirmDialog(message) {
  return new Promise(resolve => {
    const root = document.getElementById('modal-root');
    const bd = document.createElement('div');
    bd.className = 'modal-backdrop';
    bd.innerHTML = `
      <div class="modal" style="max-width:360px">
        <p style="font-size:14px;line-height:1.65;margin-bottom:1.5rem;color:var(--fg-dim)">${message}</p>
        <div class="modal-actions">
          <button class="btn ghost" id="_conf-cancel">Cancel</button>
          <button class="btn danger" id="_conf-ok">Confirm</button>
        </div>
      </div>`;
    root.appendChild(bd);

    const close = (v) => {
      bd.remove();
      document.removeEventListener('keydown', escHandler);
      resolve(v);
    };
    const escHandler = (e) => { if (e.key === 'Escape') close(false); };

    bd.querySelector('#_conf-ok').addEventListener('click',     () => close(true));
    bd.querySelector('#_conf-cancel').addEventListener('click', () => close(false));
    bd.addEventListener('click', e => { if (e.target === bd) close(false); });
    document.addEventListener('keydown', escHandler);

    // Auto-focus the Cancel button so keyboard is immediately useful
    bd.querySelector('#_conf-cancel').focus();
  });
}

// ── Floating Thinking Indicator (Premium) ─────────────────────────────────
let _thinkingEl = null;
let _thinkingPhase = '';
let _thinkingTimeout = null;

const THINKING_PHASES = {
  default:   'Thinking',
  planning:  'Planning',
  tools:     'Using tools',
  reasoning: 'Reasoning',
};

const THINKING_STATUSES = [];

/**
 * Show the premium floating thinking animation in a container.
 * @param {HTMLElement} container - The #messages element
 * @param {string} phase - Optional phase: 'planning' | 'tools' | 'reasoning'
 * @param {string} content - Optional actual thought content to display
 */
export function showThinking(container, phase = '', content = '') {
  if (_thinkingEl) {
    // If already showing, update with new content
    if (content) {
      const status = _thinkingEl.querySelector('.nx-thinking-status');
      if (status) status.textContent = content.slice(0, 100);
    }
    if (phase && phase !== _thinkingPhase) {
      _thinkingEl.dataset.phase = phase;
      const label = _thinkingEl.querySelector('.nx-thinking-label');
      if (label) label.textContent = THINKING_PHASES[phase] || 'Thinking';
      _thinkingPhase = phase;
    }
    return;
  }

  _thinkingPhase = phase || '';

  const el = document.createElement('div');
  el.className = 'nx-thinking';
  if (phase) el.dataset.phase = phase;
  
  const statusText = content ? content.slice(0, 100) : (THINKING_STATUSES[0] || '');
  el.innerHTML = `
    <div class="nx-orb-wrap">
      <div class="nx-orb"></div>
      <div class="nx-orbit"><div class="nx-orbit-dot"></div></div>
      <div class="nx-orbit"><div class="nx-orbit-dot"></div></div>
      <div class="nx-orbit"><div class="nx-orbit-dot"></div></div>
      <div class="nx-orbit"><div class="nx-orbit-dot"></div></div>
      <div class="nx-ring"></div>
    </div>
    <div class="nx-thinking-text">
      <span class="nx-thinking-label">${THINKING_PHASES[phase] || 'Thinking'}</span>
      <span class="nx-thinking-status">${statusText}</span>
    </div>`;

  container.appendChild(el);
  _thinkingEl = el;

  // Cycle through status messages (only if there are entries)
  const statusEl = el.querySelector('.nx-thinking-status');

  _cancelThinkingCycle();
  if (THINKING_STATUSES.length > 0) {
    let statusIdx = 0;
    _thinkingTimeout = setInterval(() => {
      statusIdx = (statusIdx + 1) % THINKING_STATUSES.length;
      if (statusEl) statusEl.textContent = THINKING_STATUSES[statusIdx];
    }, 2800);
  }
}

/**
 * Update the phase of an existing thinking indicator.
 * @param {string} phase - 'planning' | 'tools' | 'reasoning'
 */
export function setThinkingPhase(phase) {
  if (!_thinkingEl) return;
  _thinkingPhase = phase;
  _thinkingEl.dataset.phase = phase;
  const label = _thinkingEl.querySelector('.nx-thinking-label');
  if (label) label.textContent = THINKING_PHASES[phase] || 'Thinking';
}

/**
 * Hide the floating thinking indicator with exit animation.
 */
export function hideThinking() {
  _cancelThinkingCycle();
  if (!_thinkingEl) return;

  const el = _thinkingEl;
  _thinkingEl = null;
  _thinkingPhase = '';

  el.classList.add('exiting');
  el.addEventListener('animationend', () => el.remove(), { once: true });

  // Safety: remove after timeout even if animation doesn't fire
  setTimeout(() => { if (el.parentNode) el.remove(); }, 400);
}

function _cancelThinkingCycle() {
  if (_thinkingTimeout) {
    clearInterval(_thinkingTimeout);
    _thinkingTimeout = null;
  }
}

// ── Typing indicator (kept as lightweight fallback) ────────────────────────
let _typingEl = null;

export function showTyping(container) {
  if (_typingEl) return;
  _typingEl = document.createElement('div');
  _typingEl.className = 'typing-indicator';
  _typingEl.innerHTML = '<span></span><span></span><span></span>';
  container.appendChild(_typingEl);
}

export function hideTyping() {
  if (_typingEl) { _typingEl.remove(); _typingEl = null; }
}

// ── Copy buttons ───────────────────────────────────────────────────────────
export function injectCopyButtons(container) {
  container.querySelectorAll('pre').forEach(pre => {
    if (pre.parentElement.classList.contains('code-wrapper')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'code-wrapper';
    pre.replaceWith(wrapper);
    wrapper.appendChild(pre);
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'copy';
    btn.addEventListener('click', () => {
      const text = (pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText);
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'copied!';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 2000);
      }).catch(() => toast('Copy failed', 'error'));
    });
    wrapper.appendChild(btn);
  });
}

// ── Skeleton loaders ───────────────────────────────────────────────────────
export function showSkeletons(container, count = 3) {
  container.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const s = document.createElement('div');
    s.className = 'skeleton skeleton-card';
    s.style.opacity = String(1 - i * 0.18);
    s.style.animationDelay = `${i * 80}ms`;
    container.appendChild(s);
  }
}

export function showMsgSkeletons(container, count = 5) {
  container.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const s = document.createElement('div');
    const isRight = i % 3 === 2;
    s.className = `skeleton skeleton-msg${isRight ? ' right' : ''}`;
    s.style.opacity = String(0.9 - i * 0.12);
    s.style.width = `${50 + Math.random() * 30}%`;
    container.appendChild(s);
  }
}

// Start reasoning group - delegate to thinking.js
export function startReasoningGroup() { 
  showThinkingBlock();
}
