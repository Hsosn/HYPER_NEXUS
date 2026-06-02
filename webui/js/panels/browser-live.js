/**
 * BrowserLiveViewer — Draggable live browser rendering block for the chat panel.
 *
 * Deep wiring:
 * - Listens to main WS bus for browser_action, tool_start events
 * - Auto-popup with animation when agent starts using browser tools
 * - Dedicated /api/browser/stream WebSocket for continuous ~8fps JPEG streaming
 * - Cross-panel sync: also polls /api/browser/status on init (catches manual navigations)
 * - Toggle button in chat topbar (#browser-live-toggle)
 * - Keyboard shortcut: Ctrl+Shift+B
 *
 * Interaction:
 * - Click/right-click/double-click/mousemove forwarded to browser page
 * - Direct keyboard typing (printable chars batched for efficiency)
 * - Dedicated input bar for sending longer text strings
 * - Scroll wheel forwarded to page
 * - URL bar with navigate capability
 * - Click ripple visual feedback on canvas
 * - Agent activity badge when agent uses browser_* tools
 * - Back/Forward/Refresh navigation buttons
 *
 * Performance:
 * - ImageBitmap decode path (faster than Image element)
 * - Frame drop: skips pending renders to prevent queue buildup
 * - requestAnimationFrame rendering pipeline
 * - Opaque canvas context for GPU compositing
 * - Keyboard batching: collects rapid printable chars, sends as single message
 * - Page Visibility API: pauses streaming when tab is backgrounded
 * - Panel watcher: pauses when chat panel hidden, resumes when shown
 * - Cleanup: all timers/intervals cleared on page unload
 *
 * UX:
 * - Draggable within #chat-main bounds only (mouse + touch)
 * - Resizable via bottom-right handle
 * - Minimize/close controls
 * - Glass-morphism dark theme with teal/cyan accent matching Nexus design system
 * - Position + width persisted in localStorage (scales with window resize)
 * - State-aware: connected/connecting/offline/error/reconnecting
 * - FPS counter, status overlay, keyboard hint
 */

import { on } from '../state.js';
import { api, getAuthToken } from '../utils.js';

// ── State ─────────────────────────────────────────────────────────────
let _ws = null;            // Browser stream WebSocket
let _isVisible = false;
let _isMinimized = false;
let _browserActive = false; // Tracks whether we believe browser is active
let _canvas = null;
let _ctx = null;
let _viewerEl = null;
let _frameCount = 0;
let _fpsTimer = null;
let _reconnectTimer = null;
let _statusPollTimer = null;
let _autoHideTimer = null;
let _agentActiveTimer = null;
let _renderPending = false;
let _tabVisible = true;    // Page Visibility API
let _currentUrl = '';
let _currentTitle = '';

// Keyboard batching state
let _kbBuffer = [];         // Buffered printable characters
let _kbFlushTimer = null;   // Timeout to flush the buffer
const _KB_FLUSH_DELAY = 50; // ms — reduced from 80ms for snappier typing

// Drag/resize state
const _dragState = { active: false, offsetX: 0, offsetY: 0 };
const _resizeState = { active: false, startX: 0, startY: 0, startW: 0, startH: 0 };
let _kbFocus = false;       // Whether viewer has keyboard capture
let _connecting = false;    // Reconnect guard — prevents duplicate WS connections

// Cleanup support
let _panelObserver = null;  // MutationObserver reference for cleanup
const _abortController = new AbortController(); // For document-level listeners

// Constants
const STORAGE_KEY = 'nexus.browser-viewer.pos';
const DISPLAY_W = 1280;
const DISPLAY_H = 720;
const MIN_WIDTH = 360;
const MIN_HEIGHT = 240;

// ── Initialization ──────────────────────────────────────────────────────
export function initBrowserLive() {
    console.log('[browser-live] initializing...');
    setupEventListeners();
    setupPageVisibility();
    setupPanelWatcher();
    setupTopbarButton();
    setupCleanup();

    // Pre-create viewer DOM (hidden) so it's ready when browser starts
    createViewerElement();
    loadSavedPosition();

    // Deep connect: check if browser is already active on page load
    checkInitialBrowserStatus();
}

/**
 * Deep connect: On page load, check if browser is already active.
 */
async function checkInitialBrowserStatus() {
    try {
        const status = await api('/api/browser/status');
        if (status.open && status.url && status.url !== 'about:blank') {
            console.log('[browser-live] Browser already active on init, showing viewer');
            _browserActive = true;
            _currentUrl = status.url || '';
            _currentTitle = status.title || '';
            updateUrlBar(_currentUrl);
            showWithAnimation();
        }
    } catch (e) {
        // Browser status endpoint might not be available (server starting up)
        console.log('[browser-live] Initial browser status check skipped:', e.message);
    }
}

/* ── Event listeners ─────────────────────────────────────────────────────── */
function setupEventListeners() {
    // ── Deep connect: Listen for browser actions via the main WS bus ──
    on('ws:event', (msg) => {
        if (!msg || !msg.kind) return;

        // browser_action events emitted by backend when agent uses browser_* tools
        if (msg.kind === 'browser_action') {
            const data = msg.data || {};
            const action = data.action;

            if (action === 'open' || action === 'screenshot') {
                console.log('[browser-live] Browser action (via WS):', action);
                _browserActive = true;
                if (data.url) {
                    _currentUrl = data.url;
                    updateUrlBar(data.url);
                }
                if (data.title) _currentTitle = data.title;
                clearTimeout(_autoHideTimer);
                _autoHideTimer = null;
                if (!_isVisible) {
                    showWithAnimation();
                } else if (_isMinimized) {
                    restoreFromMinimize();
                }
            } else if (action === 'click') {
                // Agent clicked something — just show viewer if hidden
                if (!_isVisible) showWithAnimation();
            } else if (action === 'close') {
                _browserActive = false;
                if (_isVisible) {
                    updateViewerState('offline');
                    clearTimeout(_autoHideTimer);
                    _autoHideTimer = setTimeout(() => {
                        if (!_browserActive) hideWithAnimation();
                    }, 5000);
                }
            } else if (action === 'error') {
                console.warn('[browser-live] Browser error:', data.error);
            }
        }

        // ── Deep connect: Show agent activity badge when agent uses browser tools ──
        if (msg.kind === 'tool_start') {
            const toolName = msg.data?.name || '';
            if (toolName.startsWith('browser_') && _isVisible && !_isMinimized) {
                showAgentBadge(toolName.replace('browser_', ''));
            }
        }
    });

    // Keyboard shortcut: Ctrl+Shift+B to toggle viewer
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'B') {
            e.preventDefault();
            toggle();
        }
    });

    // Escape to release keyboard focus from viewer
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && _kbFocus) {
            releaseKeyboardFocus();
        }
    });
}

/**
 * Deep connect: Wire the #browser-live-toggle button in chat topbar.
 */
function setupTopbarButton() {
    const toggleBtn = document.getElementById('browser-live-toggle');
    if (!toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
        toggle();
        toggleBtn.classList.toggle('active', _isVisible && !_isMinimized);
    });
}

/**
 * Deep connect: Page Visibility API — pause streaming when tab is backgrounded.
 */
function setupPageVisibility() {
    const handler = () => {
        _tabVisible = !document.hidden;

        if (_tabVisible && _browserActive && _isVisible && !_isMinimized && !_ws) {
            console.log('[browser-live] Tab visible, resuming stream');
            reconnectStream();
        } else if (!_tabVisible && _ws) {
            console.log('[browser-live] Tab hidden, pausing stream');
            stopStream();
        }
    };

    document.addEventListener('visibilitychange', handler, { signal: _abortController.signal });
}

/**
 * Deep connect: Watch for chat panel visibility to pause/resume streaming.
 */
function setupPanelWatcher() {
    const chatPanel = document.querySelector('[data-panel="chat"]');
    if (chatPanel) {
        _panelObserver = new MutationObserver(() => {
            const isActive = chatPanel.classList.contains('active');
            if (isActive && _browserActive && _isVisible && !_isMinimized && !_ws && _tabVisible) {
                reconnectStream();
            } else if (!isActive && _ws) {
                stopStream();
            }
        });
        _panelObserver.observe(chatPanel, { attributes: true, attributeFilter: ['class'] });
    }

    // Backup: Listen for DOM CustomEvent
    window.addEventListener('panel:shown', (e) => {
        if (e.detail === 'chat' && _browserActive && _isVisible && !_isMinimized && !_ws && _tabVisible) {
            reconnectStream();
        }
    }, { signal: _abortController.signal });

    // Also listen for panel:hidden to ensure stream stops
    window.addEventListener('panel:hidden', (e) => {
        if (e.detail === 'chat' && _ws) {
            stopStream();
        }
    }, { signal: _abortController.signal });
}

/**
 * Deep connect: Clean up all resources on page unload.
 */
function setupCleanup() {
    const cleanup = () => {
        stopStream();
        clearTimeout(_autoHideTimer);
        clearTimeout(_agentActiveTimer);
        clearTimeout(_reconnectTimer);
        clearTimeout(_kbFlushTimer);
        clearInterval(_statusPollTimer);
        clearInterval(_fpsTimer);
        flushKeyboardBuffer();
        // Disconnect MutationObserver
        _panelObserver?.disconnect();
        _panelObserver = null;
        // Abort all document-level listeners
        _abortController.abort();
    };
    window.addEventListener('beforeunload', cleanup);
}

/* ── Show / Hide with animations ────────────────────────────────────── */
export function show() {
    if (_isVisible && !_isMinimized) return;
    if (!_viewerEl) createViewerElement();

    _isVisible = true;
    _isMinimized = false;
    _viewerEl.style.display = '';
    _viewerEl.classList.remove('minimized');

    // Update toggle button
    const toggleBtn = document.getElementById('browser-live-toggle');
    if (toggleBtn) toggleBtn.classList.add('active');

    // Start streaming
    if (_tabVisible) {
        reconnectStream();
    } else {
        updateViewerState('connecting');
    }

    // Focus the viewer for keyboard input (deferred)
    setTimeout(() => {
        if (_viewerEl && !_isMinimized) _viewerEl.focus();
    }, 100);
}

function showWithAnimation() {
    if (!_viewerEl) createViewerElement();
    _isVisible = true;
    _isMinimized = false;
    _viewerEl.style.display = '';
    _viewerEl.classList.remove('minimized');
    // Set initial opacity BEFORE showing to prevent flash
    _viewerEl.style.opacity = '0';
    _viewerEl.style.transform = 'scale(0.95) translateY(-8px)';

    // Update toggle button
    const toggleBtn = document.getElementById('browser-live-toggle');
    if (toggleBtn) toggleBtn.classList.add('active');

    // Start streaming
    if (_tabVisible) {
        reconnectStream();
    } else {
        updateViewerState('connecting');
    }

    // Animate in
    requestAnimationFrame(() => {
        _viewerEl.style.transition = 'opacity 0.3s var(--ease, ease-out), transform 0.3s var(--ease, ease-out), box-shadow 0.3s var(--ease, ease-out)';
        _viewerEl.style.opacity = '1';
        _viewerEl.style.transform = 'scale(1) translateY(0)';
        setTimeout(() => {
            if (_viewerEl) _viewerEl.style.transition = '';
        }, 350);
    });

    // Focus the viewer for keyboard input (deferred)
    setTimeout(() => {
        if (_viewerEl && !_isMinimized) _viewerEl.focus();
    }, 100);
}

function hide() {
    if (!_viewerEl) return;
    _isVisible = false;
    _isMinimized = false;
    _viewerEl.style.display = 'none';
    stopStream();
    releaseKeyboardFocus();
    flushKeyboardBuffer();

    const toggleBtn = document.getElementById('browser-live-toggle');
    if (toggleBtn) toggleBtn.classList.remove('active');
}

function hideWithAnimation() {
    if (!_viewerEl) return;
    _viewerEl.style.transition = 'opacity 0.25s var(--ease, ease-out), transform 0.25s var(--ease, ease-out)';
    _viewerEl.style.opacity = '0';
    _viewerEl.style.transform = 'scale(0.95) translateY(-8px)';
    setTimeout(() => {
        hide();
        if (_viewerEl) {
            _viewerEl.style.transition = '';
            _viewerEl.style.transform = '';
        }
    }, 260);
}

function toggle() {
    if (_isVisible && !_isMinimized) {
        minimize();
    } else {
        show();
    }
}

function minimize() {
    if (!_viewerEl || !_isVisible) return;
    _isMinimized = true;
    _viewerEl.classList.add('minimized');
    _viewerEl.setAttribute('tabindex', '-1');
    stopStream();
    releaseKeyboardFocus();
    flushKeyboardBuffer();
}

function restoreFromMinimize() {
    if (!_viewerEl || !_isVisible || !_isMinimized) return;
    _isMinimized = false;
    _viewerEl.classList.remove('minimized');
    _viewerEl.setAttribute('tabindex', '0');
    if (_tabVisible) reconnectStream();
}

/* ── Create DOM element ────────────────────────────────────────────────── */
function createViewerElement() {
    if (_viewerEl) return;

    const chatMain = document.getElementById('chat-main');
    if (!chatMain) {
        console.warn('[browser-live] chat-main element not found');
        return;
    }

    _viewerEl = document.createElement('div');
    _viewerEl.className = 'browser-live-viewer';
    _viewerEl.id = 'browser-live-viewer';
    _viewerEl.setAttribute('tabindex', '0');
    _viewerEl.style.display = 'none';
    _viewerEl.innerHTML = buildViewerHTML();

    chatMain.appendChild(_viewerEl);

    // Cache canvas and context (opaque for GPU compositing optimization)
    _canvas = _viewerEl.querySelector('#browser-live-canvas');
    _ctx = _canvas.getContext('2d', { alpha: false });

    // Bind all event subsystems
    bindDragEvents();
    bindResizeEvents();
    bindCanvasEvents();
    bindButtonEvents();
    bindInputBarEvents();
    bindUrlBarEvents();
    loadSavedPosition();
}

function buildViewerHTML() {
    return `
    <div class="browser-live-header" id="browser-live-header">
      <div class="browser-live-header-left">
        <span class="browser-live-indicator" id="browser-live-indicator"></span>
        <span class="browser-live-title">Browser</span>
        <span class="browser-live-fps" id="browser-live-fps"></span>
      </div>
      <div class="browser-live-header-right">
        <button class="browser-live-btn" id="browser-live-back-btn" title="Back">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <button class="browser-live-btn" id="browser-live-forward-btn" title="Forward">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
        </button>
        <button class="browser-live-btn" id="browser-live-refresh-btn" title="Refresh">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        </button>
        <button class="browser-live-btn" id="browser-live-minimize-btn" title="Minimize">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <button class="browser-live-btn browser-live-close-btn" id="browser-live-close-btn" title="Close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    </div>
    <div class="browser-live-url-bar" id="browser-live-url-bar">
      <span class="browser-live-url-icon">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      </span>
      <input type="text" id="browser-live-url-input" class="browser-live-url-input" placeholder="Navigate to URL..." autocomplete="off" spellcheck="false" />
      <button class="browser-live-url-go" id="browser-live-url-go" title="Go">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </button>
    </div>
    <div class="browser-live-body" id="browser-live-body">
      <canvas id="browser-live-canvas" width="${DISPLAY_W}" height="${DISPLAY_H}"></canvas>
      <div class="browser-live-overlay" id="browser-live-overlay">
        <div class="browser-live-overlay-content">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".4">
            <circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
          </svg>
          <span id="browser-live-status-text">Waiting for browser...</span>
        </div>
      </div>
      <div class="browser-live-keyboard-hint" id="browser-live-kb-hint">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8h.01M10 8h.01M14 8h.01M18 8h.01M8 12h.01M12 12h.01M16 12h.01M18 12h.01M8 16h.01M12 16h.01M16 16h.01"/></svg>
        <span>Click to interact</span>
      </div>
      <div class="browser-live-agent-badge" id="browser-live-agent-badge">Agent</div>
      <div class="browser-live-resize-handle" id="browser-live-resize-handle"></div>
    </div>
    <div class="browser-live-input-bar" id="browser-live-input-bar">
      <input type="text" id="browser-live-type-input" placeholder="Type text to send to browser..." autocomplete="off" spellcheck="false" />
      <button id="browser-live-type-send" title="Send text">Send</button>
    </div>
  `;
}

/* ── Dragging (constrained to #chat-main bounds) ────────────────────── */
function bindDragEvents() {
    const header = _viewerEl.querySelector('#browser-live-header');
    if (!header) return;

    header.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', onDrag, { signal: _abortController.signal });
    document.addEventListener('mouseup', endDrag, { signal: _abortController.signal });
    header.addEventListener('touchstart', startDragTouch, { passive: false });
    document.addEventListener('touchmove', onDragTouch, { passive: false, signal: _abortController.signal });
    document.addEventListener('touchend', endDrag, { signal: _abortController.signal });
}

function startDrag(e) {
    if (e.target.closest('.browser-live-btn')) return;
    e.preventDefault();
    initDrag(e.clientX, e.clientY);
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'grabbing';
}

function startDragTouch(e) {
    if (e.target.closest('.browser-live-btn')) return;
    e.preventDefault();
    const touch = e.touches[0];
    initDrag(touch.clientX, touch.clientY);
}

function initDrag(clientX, clientY) {
    _dragState.active = true;
    const rect = _viewerEl.getBoundingClientRect();
    _dragState.offsetX = clientX - rect.left;
    _dragState.offsetY = clientY - rect.top;
    _viewerEl.style.transition = 'none';
    _viewerEl.style.willChange = 'top, left';
}

function onDrag(e) {
    if (!_dragState.active || !_viewerEl) return;
    e.preventDefault();
    applyDrag(e.clientX, e.clientY);
}

function onDragTouch(e) {
    if (!_dragState.active || !_viewerEl) return;
    e.preventDefault();
    const touch = e.touches[0];
    applyDrag(touch.clientX, touch.clientY);
}

function applyDrag(clientX, clientY) {
    const chatRect = getChatPanelBounds();
    const newX = clientX - _dragState.offsetX - chatRect.left;
    const newY = clientY - _dragState.offsetY - chatRect.top;
    const maxX = chatRect.width - _viewerEl.offsetWidth;
    const maxY = chatRect.height - _viewerEl.offsetHeight;
    const clampedX = Math.max(0, Math.min(newX, maxX));
    const clampedY = Math.max(0, Math.min(newY, maxY));
    _viewerEl.style.left = clampedX + 'px';
    _viewerEl.style.top = clampedY + 'px';
    _viewerEl.style.right = 'auto';
}

function endDrag() {
    if (!_dragState.active) return;
    _dragState.active = false;
    _viewerEl.style.transition = '';
    _viewerEl.style.willChange = '';
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
    savePosition();
}

/* ── Resizing ─────────────────────────────────────────────────────────── */
function bindResizeEvents() {
    const handle = _viewerEl.querySelector('#browser-live-resize-handle');
    if (!handle) return;
    handle.addEventListener('mousedown', startResize);
    document.addEventListener('mousemove', onResize, { signal: _abortController.signal });
    document.addEventListener('mouseup', endResize, { signal: _abortController.signal });
    handle.addEventListener('touchstart', startResizeTouch, { passive: false });
    document.addEventListener('touchmove', onResizeTouch, { passive: false, signal: _abortController.signal });
    document.addEventListener('touchend', endResize, { signal: _abortController.signal });
}

function startResize(e) {
    e.preventDefault(); e.stopPropagation();
    const rect = _viewerEl.getBoundingClientRect();
    _resizeState.active = true;
    _resizeState.startX = e.clientX;
    _resizeState.startY = e.clientY;
    _resizeState.startW = rect.width;
    _resizeState.startH = rect.height;
    _viewerEl.style.transition = 'none';
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'nwse-resize';
}

function startResizeTouch(e) {
    e.preventDefault(); e.stopPropagation();
    const touch = e.touches[0];
    const rect = _viewerEl.getBoundingClientRect();
    _resizeState.active = true;
    _resizeState.startX = touch.clientX;
    _resizeState.startY = touch.clientY;
    _resizeState.startW = rect.width;
    _resizeState.startH = rect.height;
    _viewerEl.style.transition = 'none';
}

function onResize(e) {
    if (!_resizeState.active || !_viewerEl) return;
    applyResize(e.clientX, e.clientY);
}

function onResizeTouch(e) {
    if (!_resizeState.active || !_viewerEl) return;
    const touch = e.touches[0];
    applyResize(touch.clientX, touch.clientY);
}

function applyResize(clientX, clientY) {
    const chatRect = getChatPanelBounds();
    const dw = clientX - _resizeState.startX;
    const dh = clientY - _resizeState.startY;
    const newW = Math.max(MIN_WIDTH, Math.min(_resizeState.startW + dw, chatRect.width));
    const newH = Math.max(MIN_HEIGHT, Math.min(_resizeState.startH + dh, chatRect.height));
    _viewerEl.style.width = newW + 'px';
    _viewerEl.style.height = newH + 'px';
}

function endResize() {
    if (!_resizeState.active) return;
    _resizeState.active = false;
    _viewerEl.style.transition = '';
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
    savePosition();
}

function getChatPanelBounds() {
    const chatMain = document.getElementById('chat-main');
    if (!chatMain) return { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
    return chatMain.getBoundingClientRect();
}

function savePosition() {
    if (!_viewerEl) return;
    const chatRect = getChatPanelBounds();
    const x = parseFloat(_viewerEl.style.left) || 0;
    const y = parseFloat(_viewerEl.style.top) || 0;
    const w = _viewerEl.offsetWidth;
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ x, y, w, chatW: chatRect.width, chatH: chatRect.height }));
    } catch (_) { /* ignore */ }
}

function loadSavedPosition() {
    if (!_viewerEl) return;
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (!saved) return;
        const chatRect = getChatPanelBounds();
        const scaleX = chatRect.width / (saved.chatW || chatRect.width);
        const scaleY = chatRect.height / (saved.chatH || chatRect.height);
        const x = Math.max(0, Math.min(saved.x * scaleX, chatRect.width - _viewerEl.offsetWidth));
        const y = Math.max(0, Math.min(saved.y * scaleY, chatRect.height - _viewerEl.offsetHeight));
        _viewerEl.style.left = x + 'px';
        _viewerEl.style.top = y + 'px';
        _viewerEl.style.right = 'auto';
        if (saved.w && saved.w >= MIN_WIDTH) _viewerEl.style.width = saved.w + 'px';
    } catch (_) {
        if (_viewerEl) { _viewerEl.style.left = 'auto'; _viewerEl.style.right = '16px'; _viewerEl.style.top = '16px'; }
    }
}

/* ── URL bar ─────────────────────────────────────────────────────────── */
function updateUrlBar(url) {
    const input = _viewerEl?.querySelector('#browser-live-url-input');
    if (input) input.value = url || '';
}

function bindUrlBarEvents() {
    const urlInput = _viewerEl.querySelector('#browser-live-url-input');
    const goBtn = _viewerEl.querySelector('#browser-live-url-go');
    if (!urlInput || !goBtn) return;

    const navigateToUrl = async () => {
        let url = urlInput.value.trim();
        if (!url) return;
        // Auto-add https:// if no protocol
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            url = 'https://' + url;
        }
        urlInput.value = url;
        _currentUrl = url;
        _browserActive = true;

        // Navigate via WebSocket if connected (faster), otherwise REST
        // Note: URL bar is set optimistically — the backend will send page_info via WS
        // with the confirmed URL if navigation succeeds or a different URL if redirected.
        if (_ws && _ws.readyState === WebSocket.OPEN) {
            _ws.send(JSON.stringify({ type: 'navigate', url }));
        } else {
            try {
                const result = await api('/api/browser/navigate', { method: 'POST', body: JSON.stringify({ url }) });
                // Update URL from backend response (not optimistic)
                if (result.url) _currentUrl = result.url;
                if (result.title) _currentTitle = result.title;
            } catch (e) {
                console.error('[browser-live] Navigate error:', e);
            }
        }

        // Start streaming if not already
        if (!_ws || _ws.readyState !== WebSocket.OPEN) {
            reconnectStream();
        }

        urlInput.blur();
    };

    goBtn.addEventListener('click', navigateToUrl);
    urlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            navigateToUrl();
        }
        if (e.key === 'Escape') {
            e.stopPropagation();
            urlInput.blur();
        }
        // Block keys from reaching the browser keyboard handler
        e.stopPropagation();
    });

    // Click on url input — don't steal keyboard focus from viewer
    urlInput.addEventListener('focus', () => {
        releaseKeyboardFocus();
    });
}

/* ── Canvas events: click, double-click, right-click, mousemove, scroll ─────── */
function bindCanvasEvents() {
    if (!_canvas) return;

    // Click → send to browser page + show ripple
    // Double-click is handled natively by the remote browser via Playwright's click() —
    // we do NOT manually send extra clicks. The mousedown event handles both single
    // and double clicks — Playwright distinguishes them internally.
    _canvas.addEventListener('mousedown', (e) => {
        if (!_kbFocus || !_browserActive) return;
        e.preventDefault();
        e.stopPropagation();

        if (e.button === 0) { // Left click
            const { pageX, pageY, canvasX, canvasY } = getPageCoords(e);
            spawnClickRipple(canvasX, canvasY);
            sendStreamMessage({ type: 'click', x: pageX, y: pageY, button: 'left' });
        } else if (e.button === 2) { // Right click
            const { pageX, pageY } = getPageCoords(e);
            sendStreamMessage({ type: 'click', x: pageX, y: pageY, button: 'right' });
        } else if (e.button === 1) { // Middle click
            const { pageX, pageY } = getPageCoords(e);
            sendStreamMessage({ type: 'click', x: pageX, y: pageY, button: 'middle' });
        }
    });

    // Right-click context menu → prevent + send to browser
    _canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    // Touch support for canvas (tap = click, long-press = right-click, 2-finger = scroll)
    _canvas.addEventListener('touchstart', (e) => {
        if (!_kbFocus || !_browserActive) return;
        if (e.touches.length === 1) {
            e.preventDefault();
            const touch = e.touches[0];
            const fakeEvent = { clientX: touch.clientX, clientY: touch.clientY, button: 0 };
            const { pageX, pageY, canvasX, canvasY } = getPageCoords(fakeEvent);
            spawnClickRipple(canvasX, canvasY);
            sendStreamMessage({ type: 'click', x: pageX, y: pageY, button: 'left' });
        }
    }, { passive: false });

    // Mouse move → forward to browser page (throttled via rAF)
    let _moveRAF = null;
    _canvas.addEventListener('mousemove', (e) => {
        if (!_kbFocus || !_browserActive) return;
        if (_moveRAF) return;
        _moveRAF = requestAnimationFrame(() => {
            _moveRAF = null;
            const { pageX, pageY } = getPageCoords(e);
            sendStreamMessage({ type: 'mouse_move', x: pageX, y: pageY });
        });
    });

    // Scroll wheel → forward to browser page
    _canvas.addEventListener('wheel', (e) => {
        if (!_kbFocus || !_browserActive) return;
        e.preventDefault();
        e.stopPropagation();
        sendStreamMessage({
            type: 'scroll',
            delta_x: e.deltaX,
            delta_y: e.deltaY,
        });
    }, { passive: false });
}

function getPageCoords(e) {
    const rect = _canvas.getBoundingClientRect();
    const scaleX = DISPLAY_W / rect.width;
    const scaleY = DISPLAY_H / rect.height;
    const pageX = Math.max(0, Math.min(Math.round((e.clientX - rect.left) * scaleX), DISPLAY_W - 1));
    const pageY = Math.max(0, Math.min(Math.round((e.clientY - rect.top) * scaleY), DISPLAY_H - 1));
    return {
        pageX,
        pageY,
        canvasX: e.clientX - rect.left,
        canvasY: e.clientY - rect.top,
    };
}

function sendStreamMessage(msg) {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
        _ws.send(JSON.stringify(msg));
    }
}

function spawnClickRipple(x, y) {
    const body = document.getElementById('browser-live-body');
    if (!body) return;
    const ripple = document.createElement('div');
    ripple.className = 'browser-live-click-ripple';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    body.appendChild(ripple);
    const cleanup = () => ripple.remove();
    ripple.addEventListener('animationend', cleanup, { once: true });
    // Fallback: remove after 1s in case animation doesn't fire (prefers-reduced-motion)
    setTimeout(cleanup, 1000);
}

// ── Agent activity badge ─────────────────────────────────────────────
function showAgentBadge(toolName) {
    const badge = document.getElementById('browser-live-agent-badge');
    if (!badge) return;
    badge.textContent = toolName || 'Agent';
    badge.classList.add('visible');
    clearTimeout(_agentActiveTimer);
    _agentActiveTimer = setTimeout(() => badge.classList.remove('visible'), 3000);
}

// ── Button events ────────────────────────────────────────────────────
function bindButtonEvents() {
    _viewerEl.querySelector('#browser-live-minimize-btn')?.addEventListener('click', minimize);
    _viewerEl.querySelector('#browser-live-close-btn')?.addEventListener('click', hide);
    _viewerEl.querySelector('#browser-live-refresh-btn')?.addEventListener('click', async () => {
        if (_ws && _ws.readyState === WebSocket.OPEN) {
            // Navigate to current URL to refresh
            if (_currentUrl && _currentUrl !== 'about:blank') {
                sendStreamMessage({ type: 'navigate', url: _currentUrl });
            }
        }
    });
    _viewerEl.querySelector('#browser-live-back-btn')?.addEventListener('click', async () => {
        if (_kbFocus && _browserActive) {
            if (_ws && _ws.readyState === WebSocket.OPEN) {
                sendStreamMessage({ type: 'keyboard', key: 'Alt+ArrowLeft' });
            } else {
                await sendKeyViaApi('Alt+ArrowLeft');
            }
        }
    });
    _viewerEl.querySelector('#browser-live-forward-btn')?.addEventListener('click', async () => {
        if (_kbFocus && _browserActive) {
            if (_ws && _ws.readyState === WebSocket.OPEN) {
                sendStreamMessage({ type: 'keyboard', key: 'Alt+ArrowRight' });
            } else {
                await sendKeyViaApi('Alt+ArrowRight');
            }
        }
    });

    // Click on viewer body → acquire keyboard focus
    _viewerEl.querySelector('.browser-live-body')?.addEventListener('mousedown', (e) => {
        if (e.target.closest('.browser-live-btn') || e.target.closest('.browser-live-resize-handle')) return;
        acquireKeyboardFocus();
    });

    _viewerEl.addEventListener('focus', () => { if (!_isMinimized) acquireKeyboardFocus(); });
    _viewerEl.addEventListener('blur', () => {
        setTimeout(() => {
            if (!_viewerEl.contains(document.activeElement)) releaseKeyboardFocus();
        }, 10);
    });

    _viewerEl.addEventListener('keydown', handleKeyDown);
}

function acquireKeyboardFocus() {
    _kbFocus = true;
    _viewerEl.classList.add('kb-active');
    _viewerEl.setAttribute('tabindex', '0');
    const hint = _viewerEl.querySelector('#browser-live-kb-hint');
    if (hint) {
        hint.style.opacity = '0';
        setTimeout(() => { if (hint) hint.style.display = 'none'; }, 300);
    }
}

function releaseKeyboardFocus() {
    if (_kbFocus) {
        // Await the flush — use .then() since this is sync
        flushKeyboardBuffer().catch(() => {});
    }
    _kbFocus = false;
    if (_viewerEl) _viewerEl.classList.remove('kb-active');
}

/* ── Keyboard handler with batching optimization ──────────────────────── */
function handleKeyDown(e) {
    if (!_kbFocus || !_browserActive) return;
    e.stopPropagation();

    // Escape → release focus
    if (e.key === 'Escape') {
        releaseKeyboardFocus();
        return;
    }

    // Don't interfere with the type input bar or URL bar
    if (e.target.id === 'browser-live-type-input' || e.target.id === 'browser-live-url-input') return;

    e.preventDefault();

    // ── Special keys: flush buffer, then send individual press ──
    if (e.key === 'Enter' || e.key === 'Return' ||
        e.key === 'Tab' ||
        e.key === 'Backspace' || e.key === 'Delete' ||
        (e.key.startsWith('F') && e.key.length >= 2 && e.key.length <= 3 && !isNaN(e.key.slice(1))) ||
        e.key.startsWith('Arrow') ||
        e.ctrlKey || e.altKey || e.metaKey) {

        flushKeyboardBuffer();

        // Build the key string for Playwright
        if (e.key === 'Enter' || e.key === 'Return') {
            sendKeyToStream('Enter');
        } else if (e.key === 'Tab') {
            sendKeyToStream('Tab');
        } else if (e.key === 'Backspace' || e.key === 'Delete') {
            sendKeyToStream('Backspace');
        } else if (e.key.startsWith('F') && !isNaN(e.key.slice(1))) {
            sendKeyToStream(e.key);
        } else if (e.key.startsWith('Arrow')) {
            sendKeyToStream(e.key);
        } else if (e.ctrlKey || e.altKey || e.metaKey) {
            // Modifier combo: convert to Playwright format
            const parts = [];
            if (e.ctrlKey) parts.push('Control');
            if (e.altKey) parts.push('Alt');
            if (e.shiftKey) parts.push('Shift');
            if (e.metaKey) parts.push('Meta');
            parts.push(e.key.length === 1 ? e.key.toUpperCase() : e.key);
            sendKeyToStream(parts.join('+'));
        }
        return;
    }

    // ── Printable character: buffer for batched sending ──
    if (e.key === ' ') {
        _kbBuffer.push(' ');
    } else if (e.key.length === 1 && !e.ctrlKey && !e.altKey && !e.metaKey) {
        _kbBuffer.push(e.key);
    }

    if (_kbBuffer.length === 1) {
        clearTimeout(_kbFlushTimer);
        _kbFlushTimer = setTimeout(flushKeyboardBuffer, _KB_FLUSH_DELAY);
    }
}

/**
 * Flush keyboard buffer via WebSocket stream.
 */
async function flushKeyboardBuffer() {
    clearTimeout(_kbFlushTimer);
    _kbFlushTimer = null;

    if (_kbBuffer.length === 0 || !_browserActive) return;
    const text = _kbBuffer.join('');
    _kbBuffer = [];

    // Prefer sending via WebSocket stream (lower latency than REST)
    if (_ws && _ws.readyState === WebSocket.OPEN) {
        sendStreamMessage({ type: 'type', text });
    } else {
        try {
            await api('/api/browser/type', {
                method: 'POST',
                body: JSON.stringify({ text }),
            });
        } catch (err) {
            console.error('[browser-live] Batch type error:', err);
        }
    }
}

function sendKeyToStream(key) {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
        sendStreamMessage({ type: 'keyboard', key });
    } else {
        sendKeyViaApi(key);
    }
}

async function sendKeyViaApi(key) {
    try {
        await api('/api/browser/keyboard', {
            method: 'POST',
            body: JSON.stringify({ key }),
        });
    } catch (err) {
        console.error('[browser-live] Key press error:', err);
    }
}

/* ── Input bar (type text to browser) ────────────────────────────────── */
function bindInputBarEvents() {
    const input = _viewerEl.querySelector('#browser-live-type-input');
    const sendBtn = _viewerEl.querySelector('#browser-live-type-send');
    if (!input || !sendBtn) return;

    const sendTypedText = async () => {
        const text = input.value;
        if (!text || !_browserActive) return;
        input.value = '';
        input.placeholder = 'Sending...';

        if (_ws && _ws.readyState === WebSocket.OPEN) {
            sendStreamMessage({ type: 'type', text });
        } else {
            try {
                await api('/api/browser/type', {
                    method: 'POST',
                    body: JSON.stringify({ text }),
                });
            } catch (err) {
                console.error('[browser-live] Type bar error:', err);
            }
        }
        input.placeholder = 'Type text to send to browser...';
        input.focus();
    };

    sendBtn.addEventListener('click', sendTypedText);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            sendTypedText();
        }
        if (e.key === 'Escape') {
            e.stopPropagation();
            input.blur();
        }
        e.stopPropagation();
    });
}

/* ── WebSocket streaming ────────────────────────────────────────────── */
function reconnectStream() {
    // Guard against duplicate connection attempts
    if (_connecting) return;
    _connecting = true;
    stopStream();
    updateViewerState('connecting');

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = getAuthToken();
    // NOTE: matches browser_routes.py router prefix "/api/browser" + websocket "/stream"
    const url = token
        ? `${proto}//${location.host}/api/browser/stream?token=${encodeURIComponent(token)}`
        : `${proto}//${location.host}/api/browser/stream`;

    try {
        _ws = new WebSocket(url);
    } catch (e) {
        console.error('[browser-live] WebSocket creation failed:', e);
        _connecting = false;  // FIX M4: Reset guard on creation failure
        updateViewerState('error');
        _reconnectTimer = setTimeout(reconnectStream, 3000);
        return;
    }

    _ws.binaryType = 'arraybuffer';
    _frameCount = 0;
    _fpsTimer = null;
    _renderPending = false;

    _ws.onopen = () => {
        console.log('[browser-live] Stream WS connected');
        updateViewerState('connected');
        _browserActive = true;
        _connecting = false;  // Clear guard
        clearTimeout(_reconnectTimer);
        _reconnectTimer = null;
    };

    _ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            try {
                handleStreamMessage(JSON.parse(event.data));
            } catch (_) { /* ignore non-JSON */ }
        } else if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
            handleBinaryFrame(event.data);
        }
    };

    _ws.onclose = () => {
        console.log('[browser-live] Stream WS disconnected');
        _connecting = false;  // Clear guard
        if (_browserActive && _isVisible && _tabVisible) {
            updateViewerState('reconnecting');
            _reconnectTimer = setTimeout(reconnectStream, 2000);
        } else {
            updateViewerState('offline');
        }
    };

    _ws.onerror = () => {
        console.error('[browser-live] Stream WS error');
        _connecting = false;  // Clear guard
        if (_browserActive && _isVisible && _tabVisible) {
            updateViewerState('error');
            _reconnectTimer = setTimeout(reconnectStream, 3000);
        }
    };

    // Backup status polling
    clearInterval(_statusPollTimer);
    _statusPollTimer = setInterval(async () => {
        if (!_isVisible) return;
        try {
            const status = await api('/api/browser/status');
            if (!status.open || status.url === 'about:blank') {
                if (_isVisible && _browserActive) {
                    _browserActive = false;
                    updateViewerState('offline');
                    clearTimeout(_autoHideTimer);
                    _autoHideTimer = setTimeout(() => { if (!_browserActive) hideWithAnimation(); }, 5000);
                }
            }
        } catch (_) { /* ignore */ }
    }, 10000);
}

function stopStream() {
    if (_ws) {
        try { _ws.close(); } catch (_) {}
        _ws = null;
    }
    clearTimeout(_reconnectTimer);
    _reconnectTimer = null;
    clearInterval(_statusPollTimer);
    _statusPollTimer = null;
    clearInterval(_fpsTimer);
    _fpsTimer = null;
    _renderPending = false;
}

function handleStreamMessage(msg) {
    if (msg.type === 'info') {
        if (msg.open && msg.url && msg.url !== 'about:blank') {
            updateViewerState('connected');
            _browserActive = true;
            if (msg.url) {
                _currentUrl = msg.url;
                updateUrlBar(msg.url);
            }
            if (msg.title) _currentTitle = msg.title;
        } else {
            updateViewerState('connecting');
        }
    } else if (msg.type === 'page_info') {
        // Page navigated (from agent mirroring or user action)
        if (msg.url) {
            _currentUrl = msg.url;
            updateUrlBar(msg.url);
        }
        if (msg.title) _currentTitle = msg.title;
    } else if (msg.type === 'error') {
        console.error('[browser-live] Stream error:', msg.message);
        updateViewerState('error');
    } else if (msg.type === 'input_result') {
        if (!msg.success) {
            console.warn('[browser-live] Input failed:', msg);
        }
    }
}

/**
 * Optimized frame rendering pipeline:
 * 1. Skip if a render is already pending (frame drop)
 * 2. Create Blob from binary data
 * 3. Try ImageBitmap decode (GPU-accelerated in modern browsers)
 * 4. Render via requestAnimationFrame (vsync-aligned, non-blocking)
 * 5. Close bitmap immediately to free memory
 */
function handleBinaryFrame(data) {
    if (!_canvas || !_ctx || !_isVisible || _isMinimized) return;

    // Frame drop: skip if previous frame is still rendering
    if (_renderPending) return;

    _renderPending = true;
    const blob = new Blob([data], { type: 'image/jpeg' });

    if (typeof createImageBitmap === 'function') {
        createImageBitmap(blob).then(bitmap => {
            if (!_ctx || !_isVisible || _isMinimized) {
                bitmap.close();
                _renderPending = false;
                return;
            }
            requestAnimationFrame(() => {
                _ctx.drawImage(bitmap, 0, 0, DISPLAY_W, DISPLAY_H);
                bitmap.close();
                _renderPending = false;
                onFrameRendered();
            });
        }).catch(() => renderFrameFallback(blob));
    } else {
        renderFrameFallback(blob);
    }
}

function renderFrameFallback(blob) {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
        if (!_ctx || !_isVisible || _isMinimized) {
            URL.revokeObjectURL(url);
            _renderPending = false;
            return;
        }
        requestAnimationFrame(() => {
            _ctx.drawImage(img, 0, 0, DISPLAY_W, DISPLAY_H);
            URL.revokeObjectURL(url);
            _renderPending = false;
            onFrameRendered();
        });
    };
    img.onerror = () => {
        URL.revokeObjectURL(url);
        _renderPending = false;
    };
    img.src = url;
}

function onFrameRendered() {
    _frameCount++;
    // FPS counter
    if (!_fpsTimer) {
        let count = 0;
        _fpsTimer = setInterval(() => {
            const fps = _frameCount - count;
            count = _frameCount;
            const fpsEl = document.getElementById('browser-live-fps');
            if (fpsEl) fpsEl.textContent = fps > 0 ? `${fps} fps` : '';
        }, 1000);
    }
    // Hide overlay on first frame
    const overlay = document.getElementById('browser-live-overlay');
    if (overlay && overlay.style.display !== 'none') {
        overlay.style.opacity = '0';
        setTimeout(() => { if (overlay) overlay.style.display = 'none'; }, 300);
    }
}

/* ── Viewer state management ─────────────────────────────────────────── */
function updateViewerState(state) {
    const indicator = document.getElementById('browser-live-indicator');
    const statusText = document.getElementById('browser-live-status-text');
    const overlay = document.getElementById('browser-live-overlay');

    if (indicator) indicator.className = 'browser-live-indicator ' + state;

    if (statusText) {
        const labels = {
            'connected': 'Connected',
            'connecting': 'Connecting...',
            'offline': 'Browser idle',
            'error': 'Connection error',
            'reconnecting': 'Reconnecting...',
        };
        statusText.textContent = labels[state] || state;
    }

    // Show overlay for offline/error states
    if (overlay && (state === 'offline' || state === 'error' || state === 'connecting' || state === 'reconnecting')) {
        overlay.style.display = 'flex';
        overlay.style.opacity = '1';
    }

    // Border accent based on state
    if (_viewerEl) {
        _viewerEl.classList.remove('state-connected', 'state-connecting', 'state-offline', 'state-error', 'state-reconnecting');
        _viewerEl.classList.add('state-' + state);
    }
}

/* ── Public API ──────────────────────────────────────────────────────── */
export function getOrCreatePopup() {
    // Legacy compatibility — create viewer if not exists
    if (!_viewerEl) createViewerElement();
    return _viewerEl;
}
