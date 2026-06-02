import { on } from '../state.js';

const log      = document.getElementById('activity-log');
const clearBtn = document.getElementById('clear-activity');
const MAX      = 200;
const entries  = [];

// ── Kind metadata ──────────────────────────────────────
const KIND_META = {
  thought:        { icon: '💭', cls: 'thought',   label: 'Thought'      },
  reflection:     { icon: '🔮', cls: 'thought',   label: 'Reflection'   },
  tool_start:      { icon: '⚡', cls: 'tool_start', label: 'Tool Start'   },
  tool_end:       { icon: '✓',  cls: 'tool_end',   label: 'Tool End'     },
  final_answer:   { icon: '✦',  cls: 'answer',     label: 'Answer'       },
  error:          { icon: '✕',  cls: 'error',      label: 'Error'        },
  nexus_step:    { icon: '◈',  cls: 'step',       label: 'Reasoning Step'   },
  llm_call:      { icon: '⟳',  cls: 'step',       label: 'LLM Call'     },
  warn:          { icon: '⚠',  cls: 'warn',       label: 'Warning'      },
  file_upload:   { icon: '📁', cls: 'step',       label: 'File Upload'  },
  browser_action:{ icon: '🌐', cls: 'browser',   label: 'Browser'     },
  // Task Analyzer
  task_complexity:    { icon: '📊', cls: 'step',    label: 'Task Analysis' },
  task_complexity_applied: { icon: '✅', cls: 'answer', label: 'Mode Applied' },
  nexus_insight: { icon: '💡', cls: 'thought', label: 'Nexus Insight' },
  // Hallucination detection
  hallucination: { icon: '👁️', cls: 'warn', label: 'Hallucination' },
  // Loop detection
  loop_detected: { icon: '🔄', cls: 'warn', label: 'Loop Detected' },
  agent_stuck:   { icon: '⛔', cls: 'error', label: 'Agent Stuck' },
  tool_disabled: { icon: '🚫', cls: 'error', label: 'Tool Disabled' },
  // Chat events
  chat_started:   { icon: '💬', cls: 'step',    label: 'Chat Started'  },
  chat_busy:      { icon: '⏳', cls: 'step',    label: 'Chat Busy'     },
  chat_error:     { icon: '✕',  cls: 'error',   label: 'Chat Error'    },
  // Autorun events
  autorun_start:  { icon: '▶️', cls: 'step',    label: 'Autorun Start' },
  autorun_done:   { icon: '✅', cls: 'answer',  label: 'Autorun Done'  },
  autorun_error:  { icon: '✕',  cls: 'error',   label: 'Autorun Error' },
  // Goal events
  goal_created:   { icon: '🎯', cls: 'step',    label: 'Goal Created'  },
  goal_started:   { icon: '🚀', cls: 'step',    label: 'Goal Started'  },
  goal_progress:  { icon: '📈', cls: 'step',    label: 'Goal Progress' },
  goal_agent_error:{ icon: '✕', cls: 'error',   label: 'Goal Error'    },
  goal_needs_work:{ icon: '🔧', cls: 'warn',    label: 'Goal Needs Work'},
  goal_updated:   { icon: '🔄', cls: 'step',    label: 'Goal Updated'  },
  goal_deleted:   { icon: '🗑️', cls: 'default', label: 'Goal Deleted'  },
  // Heartbeat / health
  heartbeat_health:{ icon: '💚', cls: 'step',   label: 'Heartbeat'     },
  integrity_check: { icon: '🔍', cls: 'step',   label: 'Integrity Check'},
  integrity_repair:{ icon: '🔧', cls: 'warn',   label: 'Integrity Repair'},
  // Scheduled tasks
  scheduled_task_ready:{ icon: '⏰', cls: 'step', label: 'Scheduled Task'},
  nl_schedule_fired:{ icon: '⏱️', cls: 'step',  label: 'NL Schedule'  },
  stale_tasks_detected:{ icon: '🧹', cls: 'warn', label: 'Stale Tasks' },
  // Triggers
  trigger_fired:  { icon: '🔔', cls: 'step',    label: 'Trigger Fired' },
  trigger_updated:{ icon: '🔔', cls: 'step',    label: 'Trigger Updated'},
  // Integrations
  integration_toggled: { icon: '🔌', cls: 'step',   label: 'Integration' },
  integration_config_updated: { icon: '⚙️', cls: 'step', label: 'Int Config' },
  integration_deleted: { icon: '🗑️', cls: 'default', label: 'Int Deleted' },
  integration_configured: { icon: '✅', cls: 'step',   label: 'Int Configured' },
  // MCP
  mcp_server_added:   { icon: '🧩', cls: 'step',    label: 'MCP Server Added' },
  mcp_connected:      { icon: '🔗', cls: 'step',    label: 'MCP Connected' },
  mcp_disconnected:   { icon: '🔌', cls: 'default', label: 'MCP Disconnected' },
  mcp_error:          { icon: '⚠️', cls: 'error',   label: 'MCP Error' },
  mcp_server_removed: { icon: '🗑️', cls: 'default', label: 'MCP Server Removed' },
  // Settings / tools
  settings_updated:{ icon: '⚙️', cls: 'step',   label: 'Settings'     },
  custom_tool_added:{ icon: '🔧', cls: 'step',  label: 'New Tool'     },
  custom_skill_added:{ icon: '📦', cls: 'step', label: 'New Skill'    },
  custom_skill_removed:{ icon: '🗑️', cls: 'default', label: 'Skill Removed'},
  // VM actions
  vm_action:      { icon: '🖥️', cls: 'default', label: 'VM Action'    },
  vm_screenshot:  { icon: '📸', cls: 'default', label: 'Screenshot'   },
  // Startup / user management
  startup:        { icon: '🚀', cls: 'step',    label: 'Startup'      },
  shutdown:       { icon: '⏹️', cls: 'default', label: 'Shutdown'     },
  user_created:   { icon: '👤', cls: 'step',    label: 'User Created' },
  user_updated:   { icon: '👤', cls: 'step',    label: 'User Updated' },
  user_deleted:   { icon: '🗑️', cls: 'default', label: 'User Deleted' },
  // Engine cache
  engine_cache_invalidated:{ icon: '🔄', cls: 'warn', label: 'Cache Invalidated'},
};

const _LABEL_MAP = {
  thought:    'thought',
  reflection: 'reflect',
  tool_start: 'tool·start',
  tool_end:   'tool·end',
  final_answer:'answer',
  error:      'error',
  nexus_step: 'reasoning',
  llm_call:   'llm',
  warn:       'warn',
  task_complexity: 'task·analyze',
  task_complexity_applied: 'mode·apply',
  nexus_insight: 'nexus·insight',
  hallucination: 'hallucinate',
  loop_detected: 'loop',
  agent_stuck: 'stuck',
  tool_disabled: 'tool·disabled',
  // Chat
  chat_started: 'chat',
  chat_busy: 'busy',
  chat_error: 'chat·err',
  // Autorun
  autorun_start: 'autorun',
  autorun_done: 'done',
  autorun_error: 'auto·err',
  // Goals
  goal_created: 'goal·new',
  goal_started: 'goal·start',
  goal_progress: 'goal·prog',
  goal_agent_error: 'goal·err',
  goal_needs_work: 'goal·fix',
  goal_updated: 'goal·upd',
  goal_deleted: 'goal·del',
  // Health
  heartbeat_health: 'health',
  integrity_check: 'integ·chk',
  integrity_repair: 'integ·fix',
  // Tasks
  scheduled_task_ready: 'sched',
  nl_schedule_fired: 'nl·sched',
  stale_tasks_detected: 'stale',
  // Triggers
  trigger_fired: 'trigger',
  trigger_updated: 'trig·upd',
  // Integrations
  integration_toggled: 'int·tog',
  integration_config_updated: 'int·cfg',
  integration_deleted: 'int·del',
  integration_configured: 'int·cfg',
  // MCP
  mcp_server_added: 'mcp·add',
  mcp_connected: 'mcp·conn',
  mcp_disconnected: 'mcp·disc',
  mcp_error: 'mcp·err',
  mcp_server_removed: 'mcp·rm',
  // Settings / tools
  settings_updated: 'settings',
  custom_tool_added: 'new·tool',
  custom_skill_added: 'new·skill',
  custom_skill_removed: 'skill·del',
  // VM
  vm_action: 'vm',
  vm_screenshot: 'screenshot',
  // System
  startup: 'startup',
  shutdown: 'shutdown',
  user_created: 'user·new',
  user_updated: 'user·upd',
  user_deleted: 'user·del',
  engine_cache_invalidated: 'cache·inv',
};

function _esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str != null ? str : '');
  return d.innerHTML;
}

function _fmtNow() {
  return new Date().toTimeString().slice(0, 8);
}

function _summarise(data) {
  if (!data || typeof data !== 'object') return '';
  const v = data.content || data.message || data.name || data.error || data.step || null;
  if (v === null) return JSON.stringify(data).slice(0, 120);
  return String(v).slice(0, 140);
}

async function initActivity() {
  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      entries.length = 0;
      if (log) log.innerHTML = _emptyState();
    });
  }

  // Load history
  try {
    const data = await fetch('/api/events/history?limit=50').then(r => r.json());
    if (Array.isArray(data) && data.length) {
      data.forEach(addEntry);
    } else {
      if (log) log.innerHTML = _emptyState();
    }
  } catch (_) {
    if (log) log.innerHTML = _emptyState();
  }

  // Live stream
  on('ws:event', msg => addEntry(msg));
}

function addEntry(msg) {
  if (!log) return;

  // Remove empty state
  const empty = log.querySelector('.ax-empty');
  if (empty) empty.remove();

  entries.push(msg);
  if (entries.length > MAX) {
    entries.shift();
    const first = log.querySelector('.ax-entry');
    if (first) first.remove();
  }

  const meta    = KIND_META[msg.kind] || { icon: '●', cls: 'default', label: msg.kind || 'event' };
  const summary = _summarise(msg.data);
  const isError = msg.kind === 'error';
  const isOk    = msg.kind === 'tool_end';
  const isAns   = msg.kind === 'final_answer';

  const el = document.createElement('div');
  el.className = 'ax-entry';

  el.innerHTML = `
    <div class="ax-icon-col">
      <span class="ax-icon ax-icon--${meta.cls}">${meta.icon}</span>
      <span class="ax-line"></span>
    </div>
    <div class="ax-body">
      <div class="ax-header">
        <span class="ax-kind ax-kind--${meta.cls}">${_esc(_LABEL_MAP[msg.kind] || meta.label)}</span>
        ${summary ? `<span class="ax-summary">${_esc(summary)}</span>` : ''}
        <span class="ax-time">${_fmtNow()}</span>
      </div>
    </div>
  `;

  log.appendChild(el);

  // Autoscroll
  if (log.scrollHeight - log.scrollTop < log.clientHeight + 120) {
    requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; });
  }
}

function _emptyState() {
  return `<div class="ax-empty">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity=".25"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <span>Waiting for agent activity…</span>
  </div>`;
}

export { initActivity };
