/** WebSocket client with auto-reconnect and auth support. */
import { state, emit } from './state.js';
import { getAuthToken } from './utils.js';

let reconnectAttempts = 0;
let _reconnectTimer = null;
const _MAX_RECONNECT_ATTEMPTS = 30;

export function connectWS() {
  console.log('[ws] Connecting to:', location.host);
  try {
    // Clear any pending reconnect timer
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = getAuthToken();
    // Append token as query param for WebSocket auth
    const url = token
      ? `${proto}//${location.host}/ws?token=${encodeURIComponent(token)}`
      : `${proto}//${location.host}/ws`;
    console.log('[ws] Connecting to URL:', url);
    const ws = new WebSocket(url);
    state.ws = ws;

    const statusEl = document.getElementById('ws-status');
    console.log('[ws] WebSocket created, state:', ws.readyState);

    ws.onopen = () => {
      console.log('[ws] OPEN - Connected!');
      reconnectAttempts = 0;
      if (statusEl) statusEl.textContent = 'online';
      const dot = document.getElementById('ws-dot');
      if (dot) { dot.className = 'ws-dot online'; }
      emit('ws:open');
    };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        console.log('[ws] Received:', msg.kind, msg.data?.content?.slice ? msg.data.content.slice(0, 50) : '');
        // Handle auth_ok or auth_failed from server
        if (msg.kind === 'auth_ok') {
          console.log('[ws] Authenticated');
          emit('ws:auth_ok', msg);
        }
        if (msg.kind === 'auth_failed') {
          console.warn('[ws] Auth failed');
          emit('ws:auth_failed', msg);
        }
        emit('ws:event', msg);
        if (msg.kind) emit(`ws:${msg.kind}`, msg);
      } catch (err) {
        console.error('[ws] Parse error:', err);
      }
    };

    ws.onclose = (event) => {
      console.log('[ws] CLOSED', event.code, event.reason);
      if (statusEl) statusEl.textContent = 'reconnecting…';
      const dot = document.getElementById('ws-dot');
      if (dot) { dot.className = 'ws-dot connecting'; }
      reconnectAttempts++;
      // Circuit breaker: stop reconnecting after too many failed attempts
      if (reconnectAttempts > _MAX_RECONNECT_ATTEMPTS) {
        console.warn(`[ws] Max reconnect attempts (${_MAX_RECONNECT_ATTEMPTS}) reached. Giving up.`);
        if (statusEl) statusEl.textContent = 'disconnected';
        const dot = document.getElementById('ws-dot');
        if (dot) { dot.className = 'ws-dot offline'; }
        emit('ws:disconnected', { attempts: reconnectAttempts });
        return;
      }
      // If closed with 4001 (auth required), try sending auth message on reconnect
      const delay = Math.min(1000 * reconnectAttempts, 5000);
      console.log(`[ws] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
      _reconnectTimer = setTimeout(connectWS, delay);
    };

    ws.onerror = (e) => {
      console.error('[ws] ERROR:', e);
      if (statusEl) statusEl.textContent = 'error';
      const dot = document.getElementById('ws-dot');
      if (dot) { dot.className = 'ws-dot offline'; }
    };
  } catch (e) {
    console.error('[ws] Failed to create WebSocket:', e);
    // Retry after a delay
    const delay = Math.min(1000 * (reconnectAttempts + 1), 5000);
    _reconnectTimer = setTimeout(connectWS, delay);
  }
}

export function wsSend(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
    return true;
  }
  console.warn('WebSocket not open, cannot send', obj);
  return false;
}
