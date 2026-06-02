import { api } from '../utils.js';
import { toast, showSkeletons, confirmDialog } from '../enhancements.js';
import { on } from '../state.js';

const list   = document.getElementById('goals-list');
const addBtn = document.getElementById('add-goal');

// ── State ──────────────────────────────────────────────────
let _currentFilter = 'all';
let _goalsCache = [];

async function initGoals() {
  if (!list) return;

  // Build filter bar above the goal list
  _buildFilterBar();

  await refresh();
  if (addBtn) addBtn.addEventListener('click', showCreateModal);

  // Re-fetch goals when WS reconnects
  on('ws:open', () => { console.log('[goals] WS reconnected, refreshing'); refresh(); });

  // Listen for goal events from WebSocket
  on('ws:event', (msg) => {
    if (msg.kind === 'goal_created' || msg.kind === 'goal_updated' || msg.kind === 'goal_deleted') {
      refresh();
    }
  });
}

function _buildFilterBar() {
  const panel = list?.parentElement;
  if (!panel || panel.querySelector('.gl-filter-bar')) return;

  const bar = document.createElement('div');
  bar.className = 'gl-filter-bar';
  bar.innerHTML = `
    <button class="gl-filter-btn active" data-filter="all">All</button>
    <button class="gl-filter-btn" data-filter="active">Active</button>
    <button class="gl-filter-btn" data-filter="pending">Pending</button>
    <button class="gl-filter-btn" data-filter="paused">Paused</button>
    <button class="gl-filter-btn" data-filter="done">Done</button>
    <button class="gl-filter-btn" data-filter="cancelled">Cancelled</button>
  `;
  panel.insertBefore(bar, list);

  bar.querySelectorAll('.gl-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      bar.querySelectorAll('.gl-filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _currentFilter = btn.dataset.filter;
      _renderGoals(_goalsCache);
    });
  });
}

async function refresh() {
  if (!list) return;
  showSkeletons(list, 3);

  try {
    const goals = await api('/api/goals');
    _goalsCache = goals;
    _renderGoals(goals);
  } catch (e) {
    console.error('[goals] Failed to load:', e);
    list.innerHTML = `
      <div class="gl-empty">
        <div class="gl-empty-icon">\u26A0</div>
        <div class="gl-empty-title" style="color:var(--red)">Failed to load goals</div>
        <div class="gl-empty-sub">${e.message || 'Network error'}</div>
      </div>`;
  }
}

function _renderGoals(goals) {
  if (!list) return;
  list.innerHTML = '';

  const filtered = _currentFilter === 'all'
    ? goals
    : goals.filter(g => (g.status || 'active') === _currentFilter);

  if (!filtered.length) {
    const isFiltering = _currentFilter !== 'all';
    list.innerHTML = `
      <div class="gl-empty">
        <div class="gl-empty-icon">${isFiltering ? '\u25CE' : '\u25CE'}</div>
        <div class="gl-empty-title">${isFiltering ? `No ${_currentFilter} goals` : 'No goals yet'}</div>
        <div class="gl-empty-sub">${isFiltering
          ? 'Try a different filter or create a new goal.'
          : 'Set goals so Hyper Nexus knows what you\'re working toward. The agent will automatically pick them up and start working.'}</div>
      </div>`;
    return;
  }

  filtered.forEach((g, i) => {
    const p        = g.priority != null ? g.priority : 5;
    const progress = Math.round((g.progress || 0) * 100);
    const status   = g.status || 'active';

    const card = document.createElement('div');
    card.className = 'gl-card';
    card.style.animationDelay = `${i * 40}ms`;

    const PRIORITY_CFG = {
      high:   { min: 8, cls: 'gl-priority--high',   label: 'HIGH'   },
      medium: { min: 5, cls: 'gl-priority--medium',  label: 'MED'    },
      low:    { min: 0, cls: 'gl-priority--low',     label: 'LOW'    },
    };
    const pCfg = p >= 8 ? PRIORITY_CFG.high : p >= 5 ? PRIORITY_CFG.medium : PRIORITY_CFG.low;

    const STATUS_CFG = {
      active:    { cls: 'gl-status--active',    label: 'Active'    },
      done:      { cls: 'gl-status--done',      label: 'Done'      },
      paused:    { cls: 'gl-status--paused',    label: 'Paused'    },
      cancelled: { cls: 'gl-status--cancelled', label: 'Cancelled' },
      pending:   { cls: 'gl-status--pending',   label: 'Pending'   },
    };
    const sCfg = STATUS_CFG[status] || STATUS_CFG.pending;

    const barCls = progress >= 80 ? 'gl-bar--green' : progress >= 40 ? 'gl-bar--amber' : 'gl-bar--blue';

    card.innerHTML = `
      <div class="gl-card-top">
        <div class="gl-card-left">
          <span class="gl-priority ${pCfg.cls}">P${p} &middot; ${pCfg.label}</span>
        </div>
        <span class="gl-status ${sCfg.cls}">${sCfg.label}</span>
      </div>
      <div class="gl-title">${_esc(g.title)}</div>
      ${g.description ? `<div class="gl-desc">${_esc(g.description)}</div>` : ''}
      <div class="gl-progress-wrap">
        <div class="gl-progress-bar">
          <div class="gl-progress-fill ${barCls}" style="width:0%" data-target="${progress}"></div>
        </div>
        <span class="gl-progress-pct">${progress}%</span>
      </div>
      <div class="gl-meta">
        <span class="gl-meta-created">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          ${_fmtAgo(g.created_at)}
        </span>
        <div class="gl-actions">
          <button class="gl-btn gl-btn--update" data-id="${g.id}" data-progress="${progress}">Update</button>
          <button class="gl-btn gl-btn--complete" data-id="${g.id}" ${status === 'done' ? 'disabled' : ''}>
            ${status === 'done' ? '\u2713 Done' : 'Complete'}
          </button>
          <button class="gl-btn gl-btn--delete" data-id="${g.id}" title="Delete goal">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
          </button>
        </div>
      </div>
    `;

    list.appendChild(card);

    // Animate progress bar in
    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = card.querySelector('.gl-progress-fill');
        if (fill) fill.style.width = `${progress}%`;
      }, 30 + i * 35);
    });

    // Wire up buttons
    card.querySelector('.gl-btn--complete')?.addEventListener('click', async () => {
      await completeGoal(g.id, card);
    });
    card.querySelector('.gl-btn--update')?.addEventListener('click', () => {
      showUpdateModal(g);
    });
    card.querySelector('.gl-btn--delete')?.addEventListener('click', async () => {
      await deleteGoal(g.id);
    });
  });
}

// ── Delete goal ──────────────────────────────────────────

async function deleteGoal(id) {
  const ok = await confirmDialog('Delete this goal? This cannot be undone.');
  if (!ok) return;
  try {
    await api(`/api/goals/${id}`, { method: 'DELETE' });
    toast('Goal deleted', 'success');
    await refresh();
  } catch (e) {
    toast('Failed to delete goal: ' + e.message, 'error');
  }
}

// ── Complete with animation ──────────────────────────────

async function completeGoal(id, card) {
  try {
    await api(`/api/goals/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: 'done', progress: 1.0 }),
    });

    // Fill bar immediately
    const fill = card.querySelector('.gl-progress-fill');
    if (fill) {
      fill.style.width = '100%';
      fill.classList.add('gl-bar--green');
    }

    _showCompletionBurst(card);
    setTimeout(() => refresh(), 1200);
  } catch (e) {
    toast('Failed to complete goal', 'error');
  }
}

function _showCompletionBurst(card) {
  card.classList.add('gl-card--completing');
  const burst = document.createElement('div');
  burst.className = 'gl-burst';
  burst.innerHTML = `
    <div class="gl-burst-ring"></div>
    <div class="gl-burst-check">\u2713</div>
    ${_confetti()}
  `;
  card.appendChild(burst);
  toast('Goal completed!', 'success');
  setTimeout(() => {
    burst.remove();
    card.classList.remove('gl-card--completing');
  }, 1000);
}

function _confetti() {
  const colors = ['#5b5bd6','#17c964','#f5a623','#06b6d4','#f31260','#8b5cf6'];
  let html = '';
  for (let i = 0; i < 12; i++) {
    const color = colors[i % colors.length];
    const x     = (Math.random() - 0.5) * 140;
    const y     = -(Math.random() * 100 + 30);
    const rot   = Math.random() * 720 - 360;
    const size  = Math.random() * 5 + 3;
    const delay = Math.random() * 0.2;
    html += `<div class="gl-confetti-dot" style="
      background:${color};
      width:${size}px;height:${size}px;
      --cx:${x}px;--cy:${y}px;--cr:${rot}deg;
      animation-delay:${delay}s;
      border-radius:${Math.random() > 0.5 ? '50%' : '2px'};
    "></div>`;
  }
  return html;
}

// ── Create modal ─────────────────────────────────────────

function showCreateModal() {
  const root     = document.getElementById('modal-root');
  if (!root) return;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = `
    <div class="modal" style="max-width:460px">
      <div class="modal-header">
        <h2>New Goal</h2>
        <button class="modal-close" id="_g-x">\u2715</button>
      </div>
      <div class="modal-body" style="display:flex;flex-direction:column;gap:14px;">
        <div class="mfield">
          <label>Title <span style="color:var(--red)">*</span></label>
          <input type="text" id="_g-title" placeholder="e.g. Build a working prototype" autocomplete="off"/>
        </div>
        <div class="mfield">
          <label>Description <span class="mfield-hint">optional \u2014 helps the agent understand context</span></label>
          <textarea id="_g-desc" rows="3" placeholder="What does success look like?"></textarea>
        </div>
        <div class="mfield" style="max-width:160px">
          <label>Priority <span class="mfield-hint">1 = low \u00b7 10 = critical</span></label>
          <input type="number" id="_g-priority" min="1" max="10" value="5"/>
        </div>
      </div>
      <div class="modal-footer">
        <button class="topbar-btn" id="_g-cancel">Cancel</button>
        <button class="topbar-btn accent" id="_g-save">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          Create Goal
        </button>
      </div>
    </div>`;

  root.appendChild(backdrop);
  backdrop.querySelector('#_g-title').focus();

  const close = () => {
    backdrop.remove();
    document.removeEventListener('keydown', escHandler);
  };
  const escHandler = (e) => { if (e.key === 'Escape') close(); };
  backdrop.querySelector('#_g-cancel').addEventListener('click', close);
  backdrop.querySelector('#_g-x').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });
  document.addEventListener('keydown', escHandler);

  backdrop.querySelector('#_g-save').addEventListener('click', async () => {
    const title    = backdrop.querySelector('#_g-title').value.trim();
    const desc     = backdrop.querySelector('#_g-desc').value.trim();
    const priority = parseInt(backdrop.querySelector('#_g-priority').value) || 5;
    if (!title) { toast('Title is required', 'warn'); return; }

    const btn = backdrop.querySelector('#_g-save');
    btn.disabled = true;
    btn.textContent = 'Creating\u2026';

    try {
      const result = await api('/api/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, description: desc, priority }),
      });

      // Notify agent via chat that a new goal was created
      try {
        await api('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: `[SYSTEM] A new goal was just created: "${title}"${desc ? ` \u2014 ${desc}` : ''}. Priority: ${priority}/10. Goal ID: ${result.id}. Please acknowledge this goal and begin working on it autonomously.`,
            session_id: 'autorun',
          }),
        });
      } catch (_) { /* non-fatal */ }

      close();
      toast('Goal created \u2014 Hyper Nexus is on it!', 'success');
      await refresh();
    } catch (e) {
      toast('Failed to create goal: ' + e.message, 'error');
      btn.disabled = false;
      btn.textContent = 'Create Goal';
    }
  });
}

// ── Update modal ─────────────────────────────────────────

function showUpdateModal(g) {
  const root     = document.getElementById('modal-root');
  if (!root) return;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  const progress = Math.round((g.progress || 0) * 100);
  backdrop.innerHTML = `
    <div class="modal" style="max-width:420px">
      <div class="modal-header">
        <h2>Update Goal</h2>
        <button class="modal-close" id="_gu-x">\u2715</button>
      </div>
      <div class="modal-body" style="display:flex;flex-direction:column;gap:14px;">
        <div class="mfield">
          <label>Progress: <b id="_gu-pct-label">${progress}%</b></label>
          <input type="range" id="_gu-progress" min="0" max="100" value="${progress}" style="width:100%;accent-color:var(--a)"/>
        </div>
        <div class="mfield">
          <label>Status</label>
          <select id="_gu-status">
            ${['pending','active','paused','done','cancelled'].map(s =>
              `<option value="${s}" ${g.status === s ? 'selected' : ''}>${s.charAt(0).toUpperCase() + s.slice(1)}</option>`
            ).join('')}
          </select>
        </div>
      </div>
      <div class="modal-footer">
        <button class="topbar-btn gl-btn--delete" id="_gu-delete" style="color:var(--red);border-color:rgba(243,18,96,.15)">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
          Delete
        </button>
        <span style="flex:1"></span>
        <button class="topbar-btn" id="_gu-cancel">Cancel</button>
        <button class="topbar-btn accent" id="_gu-save">Save</button>
      </div>
    </div>`;

  root.appendChild(backdrop);

  backdrop.querySelector('#_gu-progress').addEventListener('input', e => {
    backdrop.querySelector('#_gu-pct-label').textContent = e.target.value + '%';
  });

  const close = () => {
    backdrop.remove();
    document.removeEventListener('keydown', escHandler);
  };
  const escHandler = (e) => { if (e.key === 'Escape') close(); };
  backdrop.querySelector('#_gu-cancel').addEventListener('click', close);
  backdrop.querySelector('#_gu-x').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });
  document.addEventListener('keydown', escHandler);

  // Delete button in modal
  backdrop.querySelector('#_gu-delete').addEventListener('click', async () => {
    close();
    await deleteGoal(g.id);
  });

  backdrop.querySelector('#_gu-save').addEventListener('click', async () => {
    const newProgress = parseInt(backdrop.querySelector('#_gu-progress').value) / 100;
    const newStatus   = backdrop.querySelector('#_gu-status').value;
    try {
      await api(`/api/goals/${g.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ progress: newProgress, status: newStatus }),
      });
      toast('Goal updated', 'success');
      close();
      await refresh();
    } catch (e) {
      toast('Failed to update: ' + e.message, 'error');
    }
  });
}

// ── Helpers ──────────────────────────────────────────────

function _fmtAgo(ts) {
  if (!ts) return 'unknown';
  const msTs = ts > 1e10 ? ts : ts * 1000;
  const sec  = Math.floor((Date.now() - msTs) / 1000);
  if (sec < 5)    return 'just now';
  if (sec < 60)   return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400)return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

function _esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str != null ? str : '');
  return d.innerHTML;
}

export { initGoals };
