/** Minimal shared state + event bus. */
export const state = {
  ws: null,
  sessionId: localStorage.getItem('nexus.session') || null,
  activePanel: 'chat',
  settings: {},
  activity: [],
  toolsCount: 0,
  memoriesCount: 0,
};

const listeners = {};

export function on(event, fn) {
  listeners[event] = listeners[event] || [];
  listeners[event].push(fn);
  return () => {
    listeners[event] = (listeners[event] || []).filter(f => f !== fn);
  };
}

export function emit(event, data) {
  (listeners[event] || []).forEach(fn => {
    try { fn(data); } catch (e) { console.error(e); }
  });
}
