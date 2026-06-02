/**
 * VMLiveViewer — Draggable live VM rendering block for the chat panel.
 *
 * Deep wiring:
 * - Listens to main WS bus for vm_action, tool_start, vm_screenshot events
 * - Auto-popup with animation when agent starts VM
 * - Auto-hides 5s after VM stops
 * - Cross-panel sync: also polls /api/vm/status on init (catches manual VM starts)
 * - Toggle button in chat topbar (#vm-live-toggle)
 * - Keyboard shortcut: Ctrl+Shift+V
 *
 * Interaction:
 * - Click/double-click/right-click/mousemove forwarded to VM
 * - Direct keyboard typing (printable chars batched for efficiency)
 * - Dedicated input bar for sending longer text strings
 * - Click ripple visual feedback on VM canvas
 * - Agent activity badge when agent uses vm_* tools
 *
 * Performance:
 * - ImageBitmap decode path (faster than Image element)
 * - Frame drop: skips pending renders to prevent queue buildup
 * - requestAnimationFrame rendering pipeline
 * - Opaque canvas context for GPU compositing
 * - Keyboard batching: collects rapid printable chars, sends as single API call
 * - Page Visibility API: pauses streaming when tab is backgrounded
 * - Panel watcher: pauses when chat panel hidden, resumes when shown
 * - Cleanup: all timers/intervals cleared on page unload
 *
 * UX:
 * - Draggable within #chat-main bounds only (mouse + touch)
 * - Resizable via bottom-right handle
 * - Minimize/close/full-VNC controls
 * - Glass-morphism dark theme matching Nexus design system
 * - Position + width persisted in localStorage (scales with window resize)
 * - State-aware: connected/connecting/offline/error/reconnecting
 * - FPS counter, status overlay, keyboard hint
 */

import { on } from './state.js';
import { api, getAuthToken } from './utils.js';

// ── State ─────────────────────────────────────────────────────────────
let _ws = null;            // VM stream WebSocket
let _isVisible = false;
let _isMinimized = false;
let _vmRunning = false;    // Tracks whether we believe VM is running
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

// Keyboard batching state
let _kbBuffer = [];         // Buffered printable characters
let _kbFlushTimer = null;   // Timeout to flush the buffer
const _KB_FLUSH_DELAY = 80; // ms — wait before sending batched chars

// Drag/resize state
const _dragState = { active: false, offsetX: 0, offsetY: 0 };
const _resizeState = { active: false, startX: 0, startY: 0, startW: 0, startH: 0 };
let _kbFocus = false;       // Whether viewer has keyboard capture
let _connecting = false;     // Reconnect guard — prevents duplicate WS connections

// Cleanup support
const _abortController = new AbortController(); // For document-level listeners
let _panelObserver = null;   // MutationObserver reference for cleanup

// Constants
const STORAGE_KEY = 'nexus.vm-viewer.pos';
const DISPLAY_W = 1280;   // FIX M5: Match VM's actual resolution for better quality
const DISPLAY_H = 720;
const MIN_WIDTH = 320;
const MIN_HEIGHT = 200;

// ── Initialization ──────────────────────────────────────────────────────
export function initVMLiveViewer() {
    console.log('[vm-live-viewer] initializing...');
    setupEventListeners();
    setupPageVisibility();
    setupPanelWatcher();
    setupTopbarButton();
    setupCleanup();

    // Pre-create viewer DOM (hidden) so it's ready when VM starts
    createViewerElement();
    loadSavedPosition();

    // Deep connect: check if VM is already running on page load
    checkInitialVMStatus();
}

/**
 * Deep connect: On page load, check if VM is already running.
 * If so, show the viewer immediately. This catches the case where
 * the user manually started the VM from the Virtual Computer panel,
 * or if the VM was running before the page refreshed.
 */
async function checkInitialVMStatus() {
    try {
        const status = await api('/api/vm/status');
        if (status.running) {
            console.log('[vm-live-viewer] VM already running on init, showing viewer');
            _vmRunning = true;
            showWithAnimation();
        }
    } catch (e) {
        // VM status endpoint might not be available (server starting up)
        console.log('[vm-live-viewer] Initial VM status check skipped:', e.message);
    }
}

/* ── Event listeners ─────────────────────────────────────────────────────── */
function setupEventListeners() {
    // ── Deep connect: Listen for VM lifecycle events via the main WS bus ──
    on('ws:event', (msg) => {
        if (!msg || !msg.kind) return;

        // vm_action events emitted by backend vm_routes.py for all VM operations
        // These fire whether the agent triggers them OR the user clicks buttons in the VM panel
        if (msg.kind === 'vm_action') {
            const action = msg.data?.action;
            const status = msg.data?.status;

            if (action === 'start' && (status === 'running' || status === 'starting')) {
                console.log('[vm-live-viewer] VM starting (via WS), showing viewer');
                _vmRunning = true;
                clearTimeout(_autoHideTimer);
                _autoHideTimer = null;
                showWithAnimation();
            } else if (action === 'stop' || action === 'destroy') {
                console.log('[vm-live-viewer] VM stopped/destroyed (via WS)');
                _vmRunning = false;
                if (_isVisible) {
                    updateViewerState('offline');
                    clearTimeout(_autoHideTimer);
                    _autoHideTimer = setTimeout(() => {
                        if (!_vmRunning) hideWithAnimation();
                    }, 5000);
                }
            } else if (action === 'restart') {
                console.log('[vm-live-viewer] VM restarting (via WS)');
                _vmRunning = false;
                updateViewerState('connecting');
                setTimeout(() => {
                    _vmRunning = true;
                    if (_isVisible) reconnectStream();
                }, 3000);
            }
            // Ignore 'mouse', 'keyboard', 'execute' sub-actions (user interaction, not lifecycle)
        }

        // ── Deep connect: Show agent activity badge when agent uses VM tools ──
        if (msg.kind === 'tool_start') {
            const toolName = msg.data?.name || '';
            if (toolName.startsWith('vm_') && _isVisible && !_isMinimized) {
                showAgentBadge(toolName.replace('vm_', ''));
            }
        }

        // Track screenshot events for frame counting and render as fallback frame
        if (msg.kind === 'vm_screenshot') {
            _frameCount++;
            // If the stream WebSocket isn't connected, render the screenshot
            // as a static frame so the user can still see the VM display.
            if (!_ws || _ws.readyState !== WebSocket.OPEN) {
                const imgData = msg.data?.image;
                if (imgData && typeof imgData === 'string' && _canvas && _ctx) {
                    const img = new Image();
                    img.onload = () => {
                        _ctx.drawImage(img, 0, 0, DISPLAY_W, DISPLAY_H);
                        img.onload = null;
                    };
                    img.src = imgData;
                    updateViewerState('connected');
                }
            }
        }
    });

    // Keyboard shortcut: Ctrl+Shift+V to toggle viewer
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'V') {
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
 * Deep connect: Wire the #vm-live-toggle button in chat topbar.
 */
function setupTopbarButton() {
    const toggleBtn = document.getElementById('vm-live-toggle');
    if (!toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
        toggle();

        // Visual feedback on the button
        toggleBtn.classList.toggle('active', _isVisible && !_isMinimized);
    });

}

/**
 * Deep connect: Page Visibility API — pause streaming when tab is backgrounded.
 * Saves bandwidth, CPU, and prevents WebSocket buffer buildup.
 */
function setupPageVisibility() {
    const handler = () => {
        _tabVisible = !document.hidden;

        if (_tabVisible && _vmRunning && _isVisible && !_isMinimized && !_ws) {
            // Tab became visible again, resume streaming
            console.log('[vm-live-viewer] Tab visible, resuming stream');
            reconnectStream();
        } else if (!_tabVisible && _ws) {
            // Tab went to background, pause streaming
            console.log('[vm-live-viewer] Tab hidden, pausing stream');
            stopStream();
        }
    };

    document.addEventListener('visibilitychange', handler, { signal: _abortController.signal });
}

/**
 * Deep connect: Watch for chat panel visibility to pause/resume streaming.
 * Uses both MutationObserver (reliable) and state event bus (backup).
 */
function setupPanelWatcher() {
    // Primary: MutationObserver on the chat panel's class attribute
    const chatPanel = document.querySelector('[data-panel="chat"]');
    if (chatPanel) {
        _panelObserver = new MutationObserver(() => {
            const isActive = chatPanel.classList.contains('active');
            if (isActive && _vmRunning && _isVisible && !_isMinimized && !_ws && _tabVisible) {
                reconnectStream();
            } else if (!isActive && _ws) {
                stopStream();
            }
        });
        _panelObserver.observe(chatPanel, { attributes: true, attributeFilter: ['class'] });
    }

    // Backup: Listen for DOM CustomEvent (app.js dispatches this)
    window.addEventListener('panel:shown', (e) => {
        if (e.detail === 'chat' && _vmRunning && _isVisible && !_isMinimized && !_ws && _tabVisible) {
            reconnectStream();
        }
    }, { signal: _abortController.signal });
}

/**
 * Deep connect: Clean up all resources on page unload to prevent memory leaks.
 */
function setupCleanup() {
    const cleanup = () => {
        stopStream();
        _connecting = false;
        clearTimeout(_autoHideTimer);
        clearTimeout(_agentActiveTimer);
        clearTimeout(_reconnectTimer);
        clearTimeout(_kbFlushTimer);
        clearInterval(_statusPollTimer);
        clearInterval(_fpsTimer);
        flushKeyboardBuffer();
        _panelObserver?.disconnect();
        _panelObserver = null;
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
    const toggleBtn = document.getElementById('vm-live-toggle');
    if (toggleBtn) toggleBtn.classList.add('active');

    // If VM is running, start streaming
    if (_vmRunning && _tabVisible) {
        reconnectStream();
    } else if (_vmRunning && !_tabVisible) {
        updateViewerState('connecting');
    } else {
        updateViewerState('offline');
    }

    // Focus the viewer for keyboard input (deferred to avoid focus race)
    setTimeout(() => {
        if (_viewerEl && !_isMinimized) _viewerEl.focus();
    }, 100);
}

function showWithAnimation() {
    show();
    if (!_viewerEl) return;
    _viewerEl.style.opacity = '0';
    _viewerEl.style.transform = 'scale(0.95) translateY(-8px)';
    requestAnimationFrame(() => {
        _viewerEl.style.transition = 'opacity 0.3s var(--ease), transform 0.3s var(--ease), box-shadow 0.3s var(--ease)';
        _viewerEl.style.opacity = '1';
        _viewerEl.style.transform = 'scale(1) translateY(0)';
        setTimeout(() => {
            if (_viewerEl) _viewerEl.style.transition = '';
        }, 350);
    });
}

function hide() {
    if (!_viewerEl) return;
    _isVisible = false;
    _isMinimized = false;
    _viewerEl.style.display = 'none';
    stopStream();
    releaseKeyboardFocus();
    flushKeyboardBuffer();

    // Update toggle button
    const toggleBtn = document.getElementById('vm-live-toggle');
    if (toggleBtn) toggleBtn.classList.remove('active');
}

function hideWithAnimation() {
    if (!_viewerEl) return;
    _viewerEl.style.transition = 'opacity 0.25s var(--ease), transform 0.25s var(--ease)';
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
    if (_vmRunning && _tabVisible) reconnectStream();
}

/* ── Create DOM element ────────────────────────────────────────────────── */
function createViewerElement() {
    if (_viewerEl) return;

    const chatMain = document.getElementById('chat-main');
    if (!chatMain) {
        console.warn('[vm-live-viewer] chat-main element not found');
        return;
    }

    _viewerEl = document.createElement('div');
    _viewerEl.className = 'vm-live-viewer';
    _viewerEl.id = 'vm-live-viewer';
    _viewerEl.setAttribute('tabindex', '0');
    _viewerEl.style.display = 'none';
    _viewerEl.innerHTML = buildViewerHTML();

    chatMain.appendChild(_viewerEl);

    // Cache canvas and context (opaque for GPU compositing optimization)
    _canvas = _viewerEl.querySelector('#vm-live-canvas');
    _ctx = _canvas.getContext('2d', { alpha: false });

    // Bind all event subsystems
    bindDragEvents();
    bindResizeEvents();
    bindCanvasEvents();
    bindButtonEvents();
    bindInputBarEvents();
    loadSavedPosition();
}

function buildViewerHTML() {
    return `
    <div class="vm-live-header" id="vm-live-header">
      <div class="vm-live-header-left">
        <span class="vm-live-indicator" id="vm-live-indicator"></span>
        <span class="vm-live-title">Virtual Computer</span>
        <span class="vm-live-fps" id="vm-live-fps"></span>
      </div>
      <div class="vm-live-header-right">
        <button class="vm-live-btn" id="vm-live-vnc-btn" title="Open full VNC">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
        </button>
        <button class="vm-live-btn" id="vm-live-minimize-btn" title="Minimize">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
        <button class="vm-live-btn vm-live-close-btn" id="vm-live-close-btn" title="Close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    </div>
    <div class="vm-live-body" id="vm-live-body">
      <canvas id="vm-live-canvas" width="${DISPLAY_W}" height="${DISPLAY_H}"></canvas>
      <div class="vm-live-overlay" id="vm-live-overlay">
        <div class="vm-live-overlay-content">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".4">
            <rect x="2" y="3" width="20" height="14" rx="2"/>
            <path d="M8 21h8M12 17v4"/>
          </svg>
          <span id="vm-live-status-text">Waiting for VM...</span>
        </div>
      </div>
      <div class="vm-live-keyboard-hint" id="vm-live-kb-hint">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8h.01M10 8h.01M14 8h.01M18 8h.01M8 12h.01M12 12h.01M16 12h.01M18 12h.01M8 16h.01M12 16h.01M16 16h.01"/></svg>
        <span>Click to interact</span>
      </div>
      <div class="vm-live-agent-badge" id="vm-live-agent-badge">Agent</div>
      <div class="vm-live-resize-handle" id="vm-live-resize-handle"></div>
    </div>
    <div class="vm-live-input-bar" id="vm-live-input-bar">
      <input type="text" id="vm-live-type-input" placeholder="Type text to send to VM..." autocomplete="off" spellcheck="false" />
      <button id="vm-live-type-send" title="Send text">Send</button>
    </div>
  `;
}

/* ── Dragging (constrained to #chat-main bounds) ────────────────────── */
function bindDragEvents() {
    const header = _viewerEl.querySelector('#vm-live-header');
    if (!header) return;

    header.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', onDrag, { signal: _abortController.signal });
    document.addEventListener('mouseup', endDrag, { signal: _abortController.signal });
    header.addEventListener('touchstart', startDragTouch, { passive: false });
    document.addEventListener('touchmove', onDragTouch, { passive: false, signal: _abortController.signal });
    document.addEventListener('touchend', endDrag, { signal: _abortController.signal });
}

function startDrag(e) {
    if (e.target.closest('.vm-live-btn')) return;
    e.preventDefault();
    initDrag(e.clientX, e.clientY);
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'grabbing';
}

function startDragTouch(e) {
    if (e.target.closest('.vm-live-btn')) return;
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
    const handle = _viewerEl.querySelector('#vm-live-resize-handle');
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
    _viewerEl.style.width = newW + 'px';
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

/* ── Canvas events: click, double-click, right-click, mousemove ─────── */
function bindCanvasEvents() {
    if (!_canvas) return;

    // Single click → send to VM via WS (faster) + show ripple
    _canvas.addEventListener('mousedown', (e) => {
        if (!_kbFocus || !_vmRunning) return;
        e.preventDefault();
        e.stopPropagation();
        const { vmX, vmY, canvasX, canvasY } = getVMDisplayCoords(e);
        spawnClickRipple(canvasX, canvasY);
        sendStreamMessage({ type: 'mouse', action: 'click', x: vmX, y: vmY, button: 'left' });
    });

    // Double-click → send to VM via WS
    _canvas.addEventListener('dblclick', (e) => {
        if (!_kbFocus || !_vmRunning) return;
        e.preventDefault();
        const { vmX, vmY } = getVMDisplayCoords(e);
        sendStreamMessage({ type: 'mouse', action: 'click', x: vmX, y: vmY, button: 'left' });
        sendStreamMessage({ type: 'mouse', action: 'click', x: vmX, y: vmY, button: 'left' });
    });

    // Right-click context menu → prevent + send to VM
    _canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!_kbFocus || !_vmRunning) return;
        const { vmX, vmY } = getVMDisplayCoords(e);
        sendStreamMessage({ type: 'mouse', action: 'click', x: vmX, y: vmY, button: 'right' });
    });

    // Mouse move → forward to VM via WS (throttled via rAF)
    let _moveRAF = null;
    _canvas.addEventListener('mousemove', (e) => {
        if (!_kbFocus || !_vmRunning) return;
        if (_moveRAF) return;
        _moveRAF = requestAnimationFrame(() => {
            _moveRAF = null;
            const { vmX, vmY } = getVMDisplayCoords(e);
            sendStreamMessage({ type: 'mouse', action: 'move', x: vmX, y: vmY });
        });
    });

    // Prevent default on canvas mousedown to avoid text selection
    _canvas.addEventListener('mousedown', (e) => { if (_kbFocus) e.preventDefault(); });
}

function getVMDisplayCoords(e) {
    const rect = _canvas.getBoundingClientRect();
    const scaleX = 1280 / rect.width;
    const scaleY = 720 / rect.height;
    const vmX = Math.max(0, Math.min(Math.round((e.clientX - rect.left) * scaleX), 1279));
    const vmY = Math.max(0, Math.min(Math.round((e.clientY - rect.top) * scaleY), 719));
    return { vmX, vmY, canvasX: e.clientX - rect.left, canvasY: e.clientY - rect.top };
}

function spawnClickRipple(x, y) {
    const body = document.getElementById('vm-live-body');
    if (!body) return;
    const ripple = document.createElement('div');
    ripple.className = 'vm-live-click-ripple';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    body.appendChild(ripple);
    ripple.addEventListener('animationend', () => ripple.remove());
}

// ── Agent activity badge ─────────────────────────────────────────────
function showAgentBadge(toolName) {
    const badge = document.getElementById('vm-live-agent-badge');
    if (!badge) return;
    badge.textContent = toolName || 'Agent';
    badge.classList.add('visible');
    clearTimeout(_agentActiveTimer);
    _agentActiveTimer = setTimeout(() => badge.classList.remove('visible'), 3000);
}

// ── Button events ────────────────────────────────────────────────────
function bindButtonEvents() {
    _viewerEl.querySelector('#vm-live-minimize-btn')?.addEventListener('click', minimize);
    _viewerEl.querySelector('#vm-live-close-btn')?.addEventListener('click', hide);
    _viewerEl.querySelector('#vm-live-vnc-btn')?.addEventListener('click', openFullVNC);

    // Click on viewer body → acquire keyboard focus
    _viewerEl.querySelector('.vm-live-body')?.addEventListener('mousedown', (e) => {
        if (e.target.closest('.vm-live-btn') || e.target.closest('.vm-live-resize-handle')) return;
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
    const hint = _viewerEl.querySelector('#vm-live-kb-hint');
    if (hint) {
        hint.style.opacity = '0';
        setTimeout(() => { if (hint) hint.style.display = 'none'; }, 300);
    }
}

function releaseKeyboardFocus() {
    if (_kbFocus) flushKeyboardBuffer(); // Send any buffered keys before releasing
    _kbFocus = false;
    if (_viewerEl) _viewerEl.classList.remove('kb-active');
}

/* ── Keyboard handler with batching optimization ────────────────────────
 *
 * Instead of sending one API call per keystroke (which causes ~50-100ms
 * latency per character), we batch consecutive printable characters
 * into a single 'type' call. Special keys (Enter, Tab, etc.) trigger
 * an immediate flush + individual press.
 */
function handleKeyDown(e) {
    if (!_kbFocus || !_vmRunning) return;
    e.stopPropagation();

    // Escape → release focus
    if (e.key === 'Escape') {
        releaseKeyboardFocus();
        return;
    }

    // Don't interfere with the type input bar (it has its own handlers)
    if (e.target.id === 'vm-live-type-input') return;

    e.preventDefault();

    // ── Special keys: flush buffer, then send individual press ──
    if (e.key === 'Enter' || e.key === 'Return' ||
        e.key === 'Tab' ||
        e.key === 'Backspace' || e.key === 'Delete' ||
        e.key.startsWith('F') && e.key.length >= 2 && e.key.length <= 3 && !isNaN(e.key.slice(1)) ||
        e.key.startsWith('Arrow') ||
        e.ctrlKey || e.altKey || e.metaKey) {

        // Flush any buffered printable characters first
        flushKeyboardBuffer();

        // Build the key string for xdotool
        if (e.key === 'Enter' || e.key === 'Return') {
            sendKeyPress('Return');
        } else if (e.key === 'Tab') {
            sendKeyPress('Tab');
        } else if (e.key === 'Backspace' || e.key === 'Delete') {
            sendKeyPress('BackSpace');
        } else if (e.key.startsWith('F') && !isNaN(e.key.slice(1))) {
            sendKeyPress(e.key);
        } else if (e.key.startsWith('Arrow')) {
            sendKeyPress(e.key);
        } else if (e.ctrlKey || e.altKey || e.metaKey) {
            // Modifier combo
            const parts = [];
            if (e.ctrlKey) parts.push('ctrl');
            if (e.altKey) parts.push('alt');
            if (e.metaKey) parts.push('super');
            parts.push(e.key.length === 1 ? e.key.toLowerCase() : e.key);
            sendKeyPress(parts.join('+'));
        }
        return;
    }

    // ── Printable character: buffer for batched sending ──
    if (e.key === ' ') {
        _kbBuffer.push(' ');
    } else if (e.key.length === 1 && !e.ctrlKey && !e.altKey && !e.metaKey) {
        _kbBuffer.push(e.key);
    }

    // Schedule a flush if we haven't already
    if (_kbBuffer.length === 1) {
        clearTimeout(_kbFlushTimer);
        _kbFlushTimer = setTimeout(flushKeyboardBuffer, _KB_FLUSH_DELAY);
    }
}

/**
 * Flush the keyboard buffer: send all buffered printable chars as a single API call.
 * This is the key optimization — instead of 5 API calls for "hello", it sends 1.
 */
async function flushKeyboardBuffer() {
    clearTimeout(_kbFlushTimer);
    _kbFlushTimer = null;

    if (_kbBuffer.length === 0 || !_vmRunning) return;
    const text = _kbBuffer.join('');
    _kbBuffer = [];

    // Prefer sending via WebSocket (lower latency than REST)
    if (_ws && _ws.readyState === WebSocket.OPEN) {
        sendStreamMessage({ type: 'keyboard', action: 'type', text });
    } else {
        try {
            await api('/api/vm/keyboard', { method: 'POST', body: JSON.stringify({ action: 'type', text }) });
        } catch (err) {
            console.error('[vm-live] Batch type error:', err);
        }
    }
}

function sendKeyPress(key) {
    // Prefer sending via WebSocket (lower latency than REST)
    if (_ws && _ws.readyState === WebSocket.OPEN) {
        sendStreamMessage({ type: 'keyboard', action: 'press', key });
    } else {
        api('/api/vm/keyboard', { method: 'POST', body: JSON.stringify({ action: 'press', key }) }).catch(err => {
            console.error('[vm-live] Key press error:', err);
        });
    }
}

/** Send a message via the VM stream WebSocket. */
function sendStreamMessage(msg) {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
        _ws.send(JSON.stringify(msg));
    }
}

/* ── Input bar (type text to VM) ────────────────────────────────────── */
function bindInputBarEvents() {
    const input = _viewerEl.querySelector('#vm-live-type-input');
    const sendBtn = _viewerEl.querySelector('#vm-live-type-send');
    if (!input || !sendBtn) return;

    const sendTypedText = async () => {
        const text = input.value;
        if (!text || !_vmRunning) return;
        input.value = '';
        input.placeholder = 'Sending...';
        try {
            await api('/api/vm/keyboard', {
                method: 'POST',
                body: JSON.stringify({ action: 'type', text }),
            });
        } catch (err) {
            console.error('[vm-live] Type bar error:', err);
        }
        input.placeholder = 'Type text to send to VM...';
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
        // Block all keys from reaching the VM keyboard handler
        e.stopPropagation();
    });
}

/* ── Full VNC in new window ────────────────────────────────────────── */
function openFullVNC() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = getAuthToken();
    const url = token
        ? `${proto}//${location.host}/api/vm/ws/vnc?token=${encodeURIComponent(token)}`
        : `${proto}//${location.host}/api/vm/ws/vnc`;

    const win = window.open('about:blank', 'nexus-vm-vnc', 'width=800,height=600,menubar=no,location=no,status=no,scrollbars=no');
    if (!win) {
        console.error('[vm-live] VNC popup blocked');
        return;
    }

    win.document.title = 'Hyper Nexus VM — Full VNC';
    const vncProto = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const host = window.location.host;
    const vncToken = encodeURIComponent(getAuthToken() || '');
    // FIX C4: Use /api/vm/ws/vnc — the router prefix is /api/vm
    const wsUrl = vncToken ? `${vncProto}${host}/api/vm/ws/vnc?token=${vncToken}` : `${vncProto}${host}/api/vm/ws/vnc`;
    
    win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Hyper Nexus VM — Full VNC</title>
<style>*{margin:0;padding:0}body{background:#000;overflow:hidden}canvas{display:block;width:100vw;height:100vh}</style></head>
<body><div id="noVNC_container" style="width:100%;height:100%">
<div style="color:#fff;padding:20px;font-family:system-ui"><p>Loading noVNC library...</p></div></div>
<script src="https://cdn.jsdelivr.net/npm/@novnc/novnc@1.5.0/core/rfb.js"><\/script>
<script>
function connectVNC(){try{ 
  if(typeof RFB === 'undefined'){
    document.getElementById('noVNC_container').innerHTML='<div style="color:#f87171;padding:20px;font-family:system-ui"><p>VNC library failed to load.</p></div>';
    return;
  }
  const rfb=new RFB(document.getElementById('noVNC_container'),'${wsUrl}',{wsProtocols:['binary'],retryWebSocket:true});
  rfb.addEventListener('connect',()=>{
    document.getElementById('noVNC_container').innerHTML='<div style="color:#4ade80;padding:20px;font-family:system-ui"><p>Connected to VM!</p></div>';
  });
  rfb.addEventListener('disconnect',()=>{
    document.getElementById('noVNC_container').innerHTML='<div style="color:#fff;padding:20px;font-family:system-ui"><p>Disconnected.</p></div>';
  });
  rfb.addEventListener('credentialsrequired',()=>{rfb.sendCredentials({password:''})});
  rfb.addEventListener('desktopname',()=>{document.title='Hyper Nexus VM — Connected'});
}catch(e){document.getElementById('noVNC_container').innerHTML='<div style="color:#f87171;padding:20px;font-family:system-ui"><p>VNC error: '+e.message+'</p></div>'}}
setTimeout(connectVNC,1500)<\/script></body></html>`);
}

/* ── WebSocket streaming ────────────────────────────────────────────── */
function reconnectStream() {
    // FIX H1: Reconnect guard — prevents duplicate WS connections
    if (_connecting) return;
    _connecting = true;
    console.log('[vm-live] Attempting stream connection...');
    stopStream();
    updateViewerState('connecting');

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = getAuthToken();
    const url = token
        ? `${proto}//${location.host}/api/vm/ws/vm-stream?token=${encodeURIComponent(token)}`
        : `${proto}//${location.host}/api/vm/ws/vm-stream`;

    console.log('[vm-live] Stream URL:', url);
    
    try {
        _ws = new WebSocket(url);
    } catch (e) {
        console.error('[vm-live] WebSocket creation failed:', e);
        _connecting = false;
        updateViewerState('error');
        _reconnectTimer = setTimeout(reconnectStream, 3000);
        return;
    }

    _ws.binaryType = 'arraybuffer';
    _frameCount = 0;
    _fpsTimer = null;
    _renderPending = false;

    _ws.onopen = () => {
        console.log('[vm-live] Stream WS connected');
        _connecting = false;
        // FIX M1: Set 'connected' on open (will confirm when first frame arrives)
        updateViewerState('connected');
        clearTimeout(_reconnectTimer);
        _reconnectTimer = null;
    };

    _ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            console.log('[vm-live] Received:', event.data.slice(0, 100));
            try {
                handleStreamMessage(JSON.parse(event.data));
            } catch (_) { /* ignore non-JSON */ }
        } else {
            console.log('[vm-live] Received frame:', event.data.size || 'unknown size');
            handleBinaryFrame(event.data);
        }
    };

    _ws.onclose = () => {
        console.log('[vm-live] Stream WS disconnected');
        _connecting = false;
        if (_vmRunning && _isVisible && _tabVisible) {
            updateViewerState('reconnecting');
            _reconnectTimer = setTimeout(reconnectStream, 2000);
        } else {
            updateViewerState('offline');
        }
    };

    _ws.onerror = () => {
        console.error('[vm-live] Stream WS error');
        _connecting = false;
        if (_vmRunning && _isVisible && _tabVisible) {
            updateViewerState('error');
            _reconnectTimer = setTimeout(reconnectStream, 3000);
        }
    };

    // Backup status polling (in case WS events are missed)
    clearInterval(_statusPollTimer);
    _statusPollTimer = setInterval(async () => {
        if (!_vmRunning || !_isVisible) return;
        try {
            const status = await api('/api/vm/status');
            if (!status.running && _isVisible) {
                _vmRunning = false;
                updateViewerState('offline');
                clearTimeout(_autoHideTimer);
                _autoHideTimer = setTimeout(() => { if (!_vmRunning) hideWithAnimation(); }, 5000);
                clearInterval(_statusPollTimer);
            }
        } catch (_) { /* ignore */ }
    }, 10000);
}

function stopStream() {
    if (_ws) {
        try { _ws.close(); } catch (_) {}
        _ws = null;
    }
    _connecting = false;
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
        if (msg.status === 'running') updateViewerState('connecting');
        else updateViewerState('offline');
    } else if (msg.type === 'error') {
        console.error('[vm-live] Stream error:', msg.message);
        updateViewerState('error');
    } else if (msg.type === 'input_result' && !msg.success) {
        console.warn('[vm-live] Input failed:', msg);
    }
}

/**
* Optimized frame rendering pipeline (v2 - deep optimizations):
* 1. Skip if a render is already pending (frame drop for performance)
* 2. Receive JPEG directly (no PNG conversion on server)
* 3. Create Blob URL directly (no decode overhead)
* 4. Draw immediately (bypass ImageBitmap for speed)
* 5. Revoke old URL to prevent memory leaks
* 6. GPU-aware: use opaque canvas for compositing
*/
function handleBinaryFrame(data) {
    if (!_canvas || !_ctx || !_isVisible || _isMinimized) return;
    
    // Frame drop skip - prevents queue buildup at high FPS
    if (_renderPending) return;
    _renderPending = true;
    
    // Direct JPEG blob (server now sends JPEG, not PNG)
    const blob = new Blob([data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    
    const img = new Image();
    img.onload = () => {
        if (!_ctx || !_isVisible || _isMinimized) {
            URL.revokeObjectURL(url);
            _renderPending = false;
            return;
        }
        // Immediate draw - no RAF needed for single frame
        _ctx.drawImage(img, 0, 0, DISPLAY_W, DISPLAY_H);
        URL.revokeObjectURL(url);
        _renderPending = false;
        onFrameRendered();
    };
    img.onerror = () => {
        URL.revokeObjectURL(url);
        _renderPending = false;
    };
img.src = url;
}

function onFrameRendered() {
    if (!_vmRunning) _vmRunning = true;
    updateViewerState('connected');
    _frameCount++;
    if (!_fpsTimer) startFPSCounter();
}

function startFPSCounter() {
    let lastCount = _frameCount;
    clearInterval(_fpsTimer);
    _fpsTimer = setInterval(() => {
        const frames = _frameCount - lastCount;
        lastCount = _frameCount;
        const fpsEl = document.getElementById('vm-live-fps');
        if (fpsEl) fpsEl.textContent = frames > 0 ? `${frames} fps` : '';
    }, 1000);
}

/* ── Viewer state UI ──────────────────────────────────────────────── */
function updateViewerState(state) {
    if (!_viewerEl) return;

    const indicator = document.getElementById('vm-live-indicator');
    const statusText = document.getElementById('vm-live-status-text');
    const overlay = document.getElementById('vm-live-overlay');
    const body = document.getElementById('vm-live-body');

    _viewerEl.classList.remove('state-connected', 'state-connecting', 'state-offline', 'state-error', 'state-minimized');

    switch (state) {
        case 'connected':
            _viewerEl.classList.add('state-connected');
            if (overlay) overlay.style.display = 'none';
            if (body) body.style.opacity = '1';
            if (indicator) indicator.title = 'Connected';
            break;
        case 'connecting':
            _viewerEl.classList.add('state-connecting');
            if (overlay) overlay.style.display = 'none';
            if (body) body.style.opacity = '0.5';
            if (statusText) statusText.textContent = 'Connecting to VM...';
            if (indicator) indicator.title = 'Connecting...';
            break;
        case 'offline':
            _viewerEl.classList.add('state-offline');
            if (overlay) overlay.style.display = 'flex';
            if (statusText) statusText.textContent = 'VM Offline';
            if (body) body.style.opacity = '0.3';
            if (indicator) indicator.title = 'VM Offline';
            break;
        case 'error':
            _viewerEl.classList.add('state-error');
            if (overlay) overlay.style.display = 'flex';
            if (statusText) statusText.textContent = 'Connection Error';
            if (body) body.style.opacity = '0.3';
            if (indicator) indicator.title = 'Error';
            break;
        case 'reconnecting':
            _viewerEl.classList.add('state-connecting');
            if (overlay) overlay.style.display = 'flex';
            if (statusText) statusText.textContent = 'Reconnecting...';
            if (body) body.style.opacity = '0.5';
            if (indicator) indicator.title = 'Reconnecting...';
            break;
    }
}

/* ── Public API ────────────────────────────────────────────────────── */
export { hide, toggle, minimize, restoreFromMinimize };
