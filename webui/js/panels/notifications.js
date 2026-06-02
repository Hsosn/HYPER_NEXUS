import { api } from '../utils.js';
import { on } from '../state.js';
import { toast } from '../enhancements.js';

let unread = 0;
let open = false;
let bar, bell, badge, listEl;

async function initNotifications() {
  inject();
  await refresh();
  on('ws:event', msg => {
    if (msg.kind === 'notification') {
      unread++;
      updateBadge();
      toast((msg.data?.kind === 'error' ? '✕' : msg.data?.kind === 'warn' ? '⚠' : 'ℹ') + ' ' + msg.data?.title, msg.data?.kind || 'info');
      if (open) load();
    }
  });
}

function inject() {
  const nav = document.querySelector('.nav');
  if (!nav) return;

  // ── Notification block as last child of .nav ────────────────
  // This keeps the sidebar-footer (green dot) fixed at the bottom,
  // never overlapped. The notification list scrolls within .nav's
  // own overflow area.
  bar = document.createElement('div');
  bar.style.cssText = 'display:flex;flex-direction:column;border-top:1px solid var(--line);margin-top:auto;flex-shrink:0';

  // Toggle row
  const toggleRow = document.createElement('div');
  toggleRow.style.cssText = 'display:flex;align-items:center;gap:8px;padding:6px 10px;cursor:pointer;user-select:none;transition:background var(--tf, 140ms)';
  toggleRow.addEventListener('mouseenter', () => { toggleRow.style.background = 'var(--hover)' });
  toggleRow.addEventListener('mouseleave', () => { toggleRow.style.background = '' });

  bell = document.createElement('span');
  bell.style.cssText = 'display:flex;align-items:center;justify-content:center;width:18px;height:18px;color:var(--tx4);flex-shrink:0';
  bell.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>';

  const label = document.createElement('span');
  label.style.cssText = 'flex:1;font-size:11.5px;font-weight:500;color:var(--tx3);letter-spacing:-0.01em';
  label.textContent = 'Notifications';

  badge = document.createElement('span');
  badge.style.cssText = 'display:none;min-width:16px;height:16px;padding:0 4px;background:var(--red);color:#fff;font-size:9px;font-weight:700;border-radius:8px;align-items:center;justify-content:center;flex-shrink:0;line-height:1';
  badge.textContent = '0';

  const chevron = document.createElement('span');
  chevron.style.cssText = 'color:var(--tx4);font-size:9px;transition:transform 200ms var(--ease, cubic-bezier(.16,1,.3,1));flex-shrink:0;line-height:1';
  chevron.textContent = '▸';

  toggleRow.appendChild(bell);
  toggleRow.appendChild(label);
  toggleRow.appendChild(badge);
  toggleRow.appendChild(chevron);

  // ── Scrollable list container (hidden by default) ────────────
  listEl = document.createElement('div');
  listEl.style.cssText = 'display:none;flex-direction:column;overflow-y:auto;overflow-x:hidden;max-height:200px;scrollbar-width:thin;scrollbar-color:var(--line2) transparent;background:var(--surface2)';

  toggleRow.onclick = () => {
    open = !open;
    listEl.style.display = open ? 'flex' : 'none';
    chevron.style.transform = open ? 'rotate(90deg)' : '';
    if (open) load();
  };

  bar.appendChild(toggleRow);
  bar.appendChild(listEl);

  // Append inside .nav so footer stays fixed below
  nav.appendChild(bar);
}

async function refresh() {
  try { unread = (await api('/api/notifications/count')).unread || 0; } catch {}
  updateBadge();
}

function updateBadge() {
  if (!badge) return;
  badge.textContent = unread > 99 ? '99+' : unread;
  badge.style.display = unread > 0 ? 'inline-flex' : 'none';
  if (bell) {
    const svg = bell.querySelector('svg');
    if (svg) svg.style.color = unread > 0 ? 'var(--a)' : '';
  }
}

async function load() {
  listEl.innerHTML = '';
  try {
    const notifs = await api('/api/notifications?limit=15');
    if (!notifs.length) {
      listEl.innerHTML = '<div style="padding:1rem;text-align:center;font-size:12px;color:var(--tx4)">No notifications</div>';
      return;
    }
    const colors = { info: 'var(--blue)', success: 'var(--a)', warn: 'var(--warn)', error: 'var(--red)' };
    notifs.forEach((n, i) => {
      const item = document.createElement('div');
      item.style.cssText = `padding:0.5rem 0.8rem;border-bottom:1px solid var(--line);cursor:default;transition:opacity 200ms var(--ease,cubic-bezier(.16,1,.3,1)),background var(--tf,140ms);opacity:0`;
      // Stagger fade-in
      requestAnimationFrame(() => { item.style.transition = `opacity 200ms var(--ease,cubic-bezier(.16,1,.3,1)) ${i * 20}ms,background var(--tf,140ms)`; item.style.opacity = '1' });
      item.addEventListener('mouseenter', () => { item.style.background = 'var(--hover)' });
      item.addEventListener('mouseleave', () => { item.style.background = '' });
      item.innerHTML = `<div style="display:flex;align-items:center;gap:6px"><span style="width:6px;height:6px;border-radius:50%;background:${colors[n.kind] || 'var(--tx4)'};flex-shrink:0"></span><span style="font-size:11px;font-weight:600;color:var(--tx2);flex:1">${n.title}</span></div>` + (n.body ? `<div style="font-size:11px;color:var(--tx3);margin-top:2px;padding-left:12px">${n.body}</div>` : '');
      listEl.appendChild(item);
    });
  } catch {
    listEl.innerHTML = '<div style="padding:1rem;color:var(--red);font-size:12px;text-align:center">Failed to load</div>';
  }
}

export { initNotifications };
