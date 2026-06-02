import { state, on } from '../state.js';
import { wsSend } from '../ws.js';
import { api, el, mdRender, fmtTime, getAuthToken } from '../utils.js';
import { preAnalyzeMessage } from '../task-complexity.js';
import {
  toast, confirmDialog,
  showTyping, hideTyping,
  showMsgSkeletons, injectCopyButtons,
  startReasoningGroup,
} from '../enhancements.js';
import {
  showThinkingBlock,
  hideThinkingBlock,
  hideThinking,
  clearThinkingBlock,
  setThinkingMode,
  addThought,
  addReasoningStep,
  addReasoningChain,
  addPhase,
  addToolUse,
  addLlmCallStep,
  addProgress,
  recoverThinkingBlock,
  isThinkingBlockVisible,
} from '../thinking.js';
import { getIntegrationContext, getTriggerContext } from './integrations.js';

// Thinking block state tracking
let _thinkingPhase = null;
let _thinkingStepCount = 0;
let _thinkingTotalSteps = null;
// v38: Track last received final_answer to prevent duplicate renders
let _lastFinalAnswerId = null;
// v38: Track whether a request is in-flight to prevent double-sends
let _sendingActive = false;

function _ensureThinkingAndAdd(content) {
  if (!isThinkingBlockVisible()) showThinkingBlock();
  addThought(content);
}

function _updateThinkingProgress() {
  if (_thinkingTotalSteps && _thinkingTotalSteps > 0) {
    addProgress(_thinkingStepCount, _thinkingTotalSteps);
  }
}

const msgsEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const newChatBtn = document.getElementById('new-chat');

const sessionsEl = document.getElementById('sessions');
const chatMainEl = document.getElementById('chat-main');

const SCROLL_THRESHOLD = 120;
let scrollFabEl = null;

function getThinkingElements() {
  return {
    block: document.querySelector('.thinking-block'),
    body: document.querySelector('.thinking-body'),
  };
}

function isNearBottom() {
  if (!msgsEl) return true;
  return msgsEl.scrollHeight - msgsEl.scrollTop - msgsEl.clientHeight < SCROLL_THRESHOLD;
}

function smartScrollToBottom(force = false) {
  if (!msgsEl) return;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const near = msgsEl.scrollHeight - msgsEl.scrollTop - msgsEl.clientHeight < SCROLL_THRESHOLD;
      if (force || near) {
        msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: 'smooth' });
      }
      updateFabVisibility();
    });
  });
}

function updateFabVisibility() {
  if (!scrollFabEl || !msgsEl) return;
  const near = isNearBottom();
  scrollFabEl.classList.toggle('visible', !near);
}

function initScrollFab() {
  scrollFabEl = document.getElementById('scroll-fab');
  if (!scrollFabEl && chatMainEl) {
    scrollFabEl = el('button', {
      id: 'scroll-fab',
      title: 'Scroll to bottom',
      onclick: () => smartScrollToBottom(true),
    },
      el('svg', {
        class: 'fab-arrow',
        width: '16', height: '16', viewBox: '0 0 24 24',
        fill: 'none', stroke: 'currentColor',
        'stroke-width': '2.5', 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      },
        el('polyline', { points: '6 9 12 15 18 9' })
      )
    );
    chatMainEl.appendChild(scrollFabEl);
  }

  if (msgsEl) {
    msgsEl.addEventListener('scroll', updateFabVisibility, { passive: true });
  }
}

function relativeTime(ts) {
  if (!ts) return '';
  const now = Date.now();
  const diff = now - (ts * 1000);
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return 'just now';
  const mins = Math.floor(secs / 60);
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  if (days < 7) return days + 'd ago';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

let sidebarOpen = true;

function _initSidebarToggle() {
  const toggleBtn = document.getElementById('session-toggle');
  const sessionList = document.getElementById('session-list');
  if (!toggleBtn || !sessionList) return;
  toggleBtn.addEventListener('click', () => {
    sidebarOpen = !sidebarOpen;
    sessionList.classList.toggle('collapsed', !sidebarOpen);
    toggleBtn.title = sidebarOpen ? 'Hide sessions' : 'Show sessions';
    toggleBtn.textContent = sidebarOpen ? '\u27E8' : '\u27E9';
  });
}

function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key === 'n') {
      e.preventDefault();
      newChatBtn?.click();
    }
    if (mod && e.key === '/') {
      e.preventDefault();
      inputEl?.focus();
    }
  });
}

export async function initChat() {
  autoResize(inputEl);
  initCharCounter();
  initFileAttach();
  _initSidebarToggle();
  initScrollFab();
  initKeyboardShortcuts();

  if (!state.sessionId) {
    try {
      const s = await api('/api/sessions?name=New%20Chat', { method: 'POST' });
      state.sessionId = s.id;
      localStorage.setItem('nexus.session', s.id);
    } catch (e) {
      console.error('Failed to create initial session', e);
      state.sessionId = crypto.randomUUID();
    }
  }

  await loadSessions();
  await loadMessages();

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  newChatBtn.addEventListener('click', async () => {
    try {
      const s = await api('/api/sessions?name=New%20Chat', { method: 'POST' });
      state.sessionId = s.id;
      localStorage.setItem('nexus.session', s.id);
      msgsEl.innerHTML = '';
      await loadSessions();
      await loadMessages();
    } catch(e) {
      toast('Failed to create session: ' + e.message, 'error');
    }
  });

  document.getElementById('clear-all-chats')?.addEventListener('click', async () => {
    const ok = await confirmDialog('Delete ALL chat sessions and messages? This cannot be undone.');
    if (!ok) return;
    try {
      await api('/api/sessions', { method: 'DELETE' });
      state.sessionId = null;
      localStorage.removeItem('nexus.session');
      msgsEl.innerHTML = '';
      const s = await api('/api/sessions?name=New%20Chat', { method: 'POST' });
      state.sessionId = s.id;
      localStorage.setItem('nexus.session', s.id);
      await loadSessions();
      await loadMessages();
      toast('All sessions cleared.', 'success');
    } catch(e) {
      toast('Failed to clear sessions: ' + e.message, 'error');
    }
  });

  on('ws:event', (msg) => {
    if (!msg.kind) return;

    switch (msg.kind) {
      case 'thought':
      case 'reflection': {
        startReasoningGroup();
        const thoughtContent = msg.data?.content || msg.content;
        if (thoughtContent) {
          const depth = msg.data?.depth || (msg.kind === 'reflection' ? 2 : 1);
          addReasoningStep(thoughtContent, depth);
        }
        _thinkingStepCount++;
        _updateThinkingProgress();
        return;
      }
      case 'llm_call':
        addLlmCallStep(msg.data?.step);
        _thinkingStepCount++;
        _updateThinkingProgress();
        return;
      case 'thinking_phase': {
        const phase = msg.data?.phase || 'thinking';
        _thinkingPhase = phase;
        if (msg.data?.total_steps) {
          _thinkingTotalSteps = msg.data.total_steps;
        }
        const phaseMap = {
          reasoning: { label: 'Reasoning', color: '#8b5cf6', icon: '🧠' },
          tools:     { label: 'Tools',     color: '#06b6d4', icon: '🔧' },
          planning:  { label: 'Planning',  color: '#6366f1', icon: '📋' },
          executing: { label: 'Executing', color: '#0070f3', icon: '⚡' },
        };
        const config = phaseMap[phase];
        if (config) {
          addPhase(config.label, config.color, config.icon);
        } else {
          addThought('○ ' + phase);
        }
        return;
      }
      case 'nexus_step':
        addThought('▸ Step ' + (msg.data?.step || 0) + (msg.data?.action ? ': ' + String(msg.data.action).slice(0, 60) : ''));
        _thinkingStepCount++;
        if (msg.data?.total_steps) {
          _thinkingTotalSteps = msg.data.total_steps;
        }
        _updateThinkingProgress();
        return;
      case 'tool_start':
        addToolUse(msg.data?.name || 'tool', msg.data?.params || null, null, null, undefined);
        return;
      case 'tool_end':
        addToolUse(msg.data?.name || 'tool', null, msg.data?.output || null, msg.data?.duration || null, msg.data?.success !== false);
        return;
      case 'agent_start': {
        const agentName = msg.data?.agent || msg.data?.name || 'Agent';
        const agentColors = { reflector: '#06b6d4' };
        const agentColor = agentColors[agentName.toLowerCase()] || '#6366f1';
        addPhase(agentName, agentColor, '🤖');
        return;
      }
      case 'step_start':
        addThought('▸ Step ' + (msg.data?.step || 0) + ': ' + (msg.data?.action?.slice(0, 50) || '...'));
        _thinkingStepCount++;
        _updateThinkingProgress();
        return;
      case 'step_end': {
        const stepStatus = msg.data?.status === 'done' ? '✓' : msg.data?.status === 'error' ? '✗' : '→';
        const stepContent = msg.data?.result ? _stripDSML ? _stripDSML(String(msg.data.result).slice(0, 100)) : String(msg.data.result).slice(0, 100) : '';
        addThought('  ' + stepStatus + ' Step ' + (msg.data?.step || 0) + (stepContent ? ': ' + stepContent : ''));
        return;
      }
      case 'step_error':
        const stepErr = msg.data?.error || msg.data?.action || 'unknown';
        addThought('✗ Step error: ' + (stepErr.length > 100 ? stepErr.slice(0, 100) + '...' : stepErr));
        return;
      case 'critique':
      case 'chain_end':
      case 'nexus_planning':
      case 'nexus_branches':
      case 'nexus_reflection':
        return;
      case 'final_answer': {
        // v38: Deduplicate final_answer events — sometimes the server emits
        // both an 'agent_message' and 'final_answer' for the same response,
        // causing the assistant message to render twice.
        const answerId = msg.data?.session_id + ':' + (msg.data?.content || '').slice(0, 50);
        if (_lastFinalAnswerId === answerId) {
          return;  // Skip duplicate
        }
        _lastFinalAnswerId = answerId;
        
        hideThinkingBlock();
        hideTyping();
        setSending(false);
        _thinkingPhase = null;
        _thinkingStepCount = 0;
        _thinkingTotalSteps = null;
        if (!msg.data?.session_id || msg.data.session_id === state.sessionId) {
          appendMessage('assistant', msg.data.content || '(empty response)', msg.data.workspace_files || []);
        }
        // Single deferred scroll — content is rendered, wait one frame for layout
        requestAnimationFrame(() => smartScrollToBottom(true));
        break;
      }
      case 'error': {
        hideThinking();
        hideTyping();
        setSending(false);
        _thinkingPhase = null;
        _thinkingStepCount = 0;
        _thinkingTotalSteps = null;
        if (!msg.data?.session_id || msg.data.session_id === state.sessionId) {
          const errMsg = msg.data.error || 'unknown error';
          const isModelErr = errMsg.includes('temporarily unavailable') || errMsg.includes('circuit');
          if (isModelErr) {
            appendCircuitBreakerBanner(errMsg);
          } else {
            appendEvent('error', errMsg);
          }
          toast(errMsg, 'error');
        }
        smartScrollToBottom(true);
        break;
      }
      case 'warn':
        console.warn('[agent warn]', msg.data);
        break;

      case 'nexus_insight':
        if (msg.data?.suggested && msg.data?.message) {
          toast('Nexus insight: ' + msg.data.message, 'info', 5000);
        }
        break;
      case 'task_complexity':
        if (msg.data?.complexity) {
          console.log('[task-complexity] ' + msg.data.complexity + ' (confidence: ' + msg.data.confidence + ')');
        }
        break;
      case 'task_complexity_applied':
        toast('Agent configuration updated', 'success', 2500);
        break;
    }
  });

  window.addEventListener('trigger:selected', (e) => {
    const { context } = e.detail;
    if (context && inputEl) {
      inputEl.value = context;
      inputEl.focus();
      inputEl.dispatchEvent(new Event('input'));
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + 'px';
    }
  });

  on('ws:open', async () => {
    console.log('[chat] WS reconnected, refreshing data');
    try {
      await loadSessions();
      if (state.sessionId) await loadMessages();
    } catch (e) {
      console.warn('[chat] Refresh after reconnect failed:', e);
    }
  });
}

function autoResize(textarea) {
  if (!textarea) return;
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 180) + 'px';
  });
}

function initCharCounter() {
  const counter = document.getElementById('char-count');
  if (!counter || !inputEl) return;
  inputEl.addEventListener('input', () => {
    const len = inputEl.value.length;
    counter.textContent = len > 0 ? String(len) : '';
    counter.className = len > 2000 ? 'over' : len > 1500 ? 'warn' : len > 100 ? 'show' : '';
  });
}

let _pendingFiles = [];

function initFileAttach() {
  const attachBtn = document.getElementById('attach-btn');
  const fileInput = document.getElementById('file-input');
  const strip = document.getElementById('attach-strip');
  if (!attachBtn || !fileInput || !strip) return;

  attachBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    Array.from(fileInput.files).forEach(f => addFile(f));
    fileInput.value = '';
  });

  const composerArea = document.getElementById('composer-area');
  if (composerArea) {
    composerArea.addEventListener('dragover', e => { e.preventDefault(); composerArea.classList.add('drag-over'); });
    composerArea.addEventListener('dragleave', () => composerArea.classList.remove('drag-over'));
    composerArea.addEventListener('drop', e => {
      e.preventDefault();
      composerArea.classList.remove('drag-over');
      Array.from(e.dataTransfer.files).forEach(f => addFile(f));
    });
  }
}

function addFile(file) {
  _pendingFiles.push(file);
  renderAttachStrip();
}

function renderAttachStrip() {
  const strip = document.getElementById('attach-strip');
  if (!strip) return;
  strip.innerHTML = '';
  strip.style.display = _pendingFiles.length ? 'flex' : 'none';
  _pendingFiles.forEach((f, i) => {
    const chip = document.createElement('div');
    chip.className = 'attach-chip';
    chip.innerHTML = '<span>' + fileIcon(f.name) + '</span><span>' + esc(f.name.length > 24 ? f.name.slice(0,22) + '…' : f.name) + '</span><button class="attach-chip-del" data-i="' + i + '">×</button>';
    chip.querySelector('.attach-chip-del').addEventListener('click', () => {
      _pendingFiles.splice(i, 1);
      renderAttachStrip();
    });
    strip.appendChild(chip);
  });
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  const map = {
    pdf:'📄', doc:'📝', docx:'📝', odt:'📝', txt:'📄', md:'📄', rtf:'📄',
    py:'🐍', js:'📜', ts:'📜', jsx:'📜', tsx:'📜', html:'🌐', css:'🎨',
    json:'📋', xml:'📋', yaml:'📋', yml:'📋', toml:'📋', ini:'📋', env:'📋',
    sh:'⚙️', bat:'⚙️', ps1:'⚙️', sql:'🗄️', rs:'🦀', go:'🔵', cpp:'🔧',
    c:'🔧', h:'🔧', cs:'🔧', java:'☕', php:'🐘', ruby:'💎', swift:'🍎',
    csv:'📊', xlsx:'📊', xls:'📊', ods:'📊', parquet:'📊', jsonl:'📊',
    png:'🖼️', jpg:'🖼️', jpeg:'🖼️', gif:'🖼️', svg:'🖼️', webp:'🖼️',
    bmp:'🖼️', ico:'🖼️', tiff:'🖼️',
    zip:'📦', tar:'📦', gz:'📦', bz2:'📦', rar:'📦', '7z':'📦', xz:'📦',
    mp3:'🎵', wav:'🎵', ogg:'🎵', flac:'🎵', aac:'🎵',
    mp4:'🎬', mov:'🎬', avi:'🎬', mkv:'🎬', webm:'🎬',
    epub:'📚', pptx:'📊', ppt:'📊', key:'📊',
  };
  return map[ext] || '📎';
}

function esc(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

async function uploadPendingFiles(sessionId) {
  const uploaded = [];
  const attachBtn = document.getElementById('attach-btn');
  if (attachBtn) attachBtn.disabled = true;
  for (const file of _pendingFiles) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('subfolder', '');
    try {
      const token = getAuthToken();
      const headers = {};
      if (token) headers['Authorization'] = 'Bearer ' + token;
      const res = await fetch('/api/upload', { method: 'POST', headers, body: fd });
      const data = await res.json();
      if (data.ok) uploaded.push({ name: data.name, path: data.path, size: data.size });
      else toast('Upload failed: ' + file.name, 'error');
    } catch(e) { toast('Upload error: ' + file.name, 'error'); }
  }
  _pendingFiles = [];
  renderAttachStrip();
  if (attachBtn) attachBtn.disabled = false;
  return uploaded;
}

function setSending(active) {
  _sendingActive = active;
  if (sendBtn) {
    sendBtn.classList.toggle('loading', active);
    sendBtn.disabled = active;  // v38: Prevent double-sends
  }
  if (inputEl) {
    inputEl.disabled = active;  // v38: Disable input while processing
  }
}

async function loadSessions() {
  if (!sessionsEl) return;
  try {
    const sessions = await api('/api/sessions');
    sessionsEl.innerHTML = '';

    if (!sessions.length) {
      sessionsEl.appendChild(el('div', { class: 'empty', style: { padding: '1rem' } }, 'No sessions yet.'));
      return;
    }

    sessions.forEach(s => {
      const label = el('span', { class: 'session-label' }, s.name || 'Session');
      const timeEl = el('span', { class: 'session-time' }, relativeTime(s.updated_at || s.created_at));

      const delBtn = el('button', {
        class: 'session-del-btn',
        title: 'Delete session',
        onclick: async (ev) => {
          ev.stopPropagation();
          const ok = await confirmDialog('Delete this session and all its messages?');
          if (!ok) return;
          const btn = ev.target.closest('button');
          if (btn) { btn.disabled = true; btn.textContent = '…'; }
          try {
            await api('/api/sessions/' + s.id, { method: 'DELETE' });
            toast('Session deleted.', 'success');

            if (s.id === state.sessionId) {
              state.sessionId = null;
              localStorage.removeItem('nexus.session');
              msgsEl.innerHTML = '';
              _thinkingPhase = null;
              _thinkingStepCount = 0;
              _thinkingTotalSteps = null;
              clearThinkingBlock();
              try {
                const ns = await api('/api/sessions?name=New%20Chat', { method: 'POST' });
                state.sessionId = ns.id;
                localStorage.setItem('nexus.session', ns.id);
              } catch (e2) {
                state.sessionId = crypto.randomUUID();
                localStorage.setItem('nexus.session', state.sessionId);
              }
            }
            await loadSessions();
            await loadMessages();
          } catch(e) {
            toast('Failed to delete: ' + e.message, 'error');
          } finally {
            if (btn) { btn.disabled = false; btn.textContent = '×'; }
          }
        },
      }, '×');

      const item = el('div', {
        class: 'session-item' + (s.id === state.sessionId ? ' active' : ''),
        style: { display: 'flex', alignItems: 'center', gap: '4px' },
        onclick: async () => {
          if (s.id === state.sessionId) return;
          state.sessionId = s.id;
          localStorage.setItem('nexus.session', s.id);
          msgsEl.innerHTML = '';
          _thinkingPhase = null;
          _thinkingStepCount = 0;
          _thinkingTotalSteps = null;
          clearThinkingBlock();
          await loadMessages();
          sessionsEl.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
        },
      }, label, timeEl, delBtn);

      sessionsEl.appendChild(item);
    });
  } catch(e) {
    sessionsEl.innerHTML = '<div style="padding:1rem;color:var(--red)">Failed to load sessions</div>';
  }
}

async function loadMessages() {
  if (!msgsEl || !state.sessionId) return;
  showMsgSkeletons(msgsEl);
  try {
    const msgs = await api('/api/sessions/' + state.sessionId + '/messages');
    msgsEl.innerHTML = '';
    if (!msgs.length) {
      msgsEl.appendChild(buildEmptyChat());
      return;
    }

    const frag = document.createDocumentFragment();
    msgs.forEach(m => {
      frag.appendChild(buildMessageEl(m.role, m.content, false));
    });
    msgsEl.appendChild(frag);
    smartScrollToBottom(true);
    setTimeout(() => smartScrollToBottom(true), 100);
  } catch {
    msgsEl.innerHTML = '';
    appendEvent('error', 'Failed to load messages.');
  }
}

function buildEmptyChat() {
  const div = el('div', { class: 'empty' });
  div.appendChild(el('div', { class: 'empty-title' }, 'Start a conversation'));
  div.appendChild(el('div', { class: 'empty-sub' }, 'Type a message below — Hyper Nexus will think, plan, and use tools to help you.'));

  const chips = [
    { label: 'Help me write code', icon: '💻' },
    { label: 'Analyze my data', icon: '📊' },
    { label: 'Create a workflow', icon: '🔄' },
    { label: 'Explain a concept', icon: '💡' },
  ];

  const chipsWrap = el('div', { class: 'empty-chips' });
  chips.forEach(({ label, icon }) => {
    const chip = el('button', {
      class: 'empty-chip',
      html: icon + ' ' + esc(label),
      onclick: () => {
        if (inputEl) {
          inputEl.value = label;
          inputEl.focus();
          inputEl.dispatchEvent(new Event('input'));
        }
      },
    });
    chipsWrap.appendChild(chip);
  });
  div.appendChild(chipsWrap);
  return div;
}

function buildMessageEl(role, content, attachments = [], animate = true) {
  const container = el('div', { class: 'msg ' + role });
  const label = role === 'user' ? 'You' : role === 'assistant' ? 'Hyper Nexus' : role;

  const header = el('div', { class: 'msg-header' });
  header.appendChild(el('span', { class: 'msg-label' }, label));
  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  header.appendChild(el('span', { class: 'msg-time' }, timeStr));
  container.appendChild(header);

  const bubble = el('div', { class: 'msg-bubble', html: content ? mdRender(content) : '' });

  if (attachments && attachments.length) {
    attachments.forEach(f => {
      const chip = document.createElement('a');
      chip.className = 'msg-attachment';
      chip.href = '/api/workspace/download?path=' + encodeURIComponent(f.path);
      chip.target = '_blank';
      chip.innerHTML = '<span>' + fileIcon(f.name) + '</span><span>' + esc(f.name) + '</span><span style="opacity:.6;font-size:11px">' + fmtBytes(f.size) + '</span>';
      bubble.appendChild(chip);
    });
  }

  container.appendChild(bubble);

  if (role === 'assistant') {
    const actions = el('div', { class: 'msg-actions' });

    const copyBtn = el('button', {
      class: 'msg-action-btn',
      title: 'Copy',
      onclick: () => {
        const text = bubble.innerText || bubble.textContent || '';
        navigator.clipboard.writeText(text).then(() => {
          copyBtn.classList.add('active');
          toast('Copied to clipboard', 'success', 2000);
          setTimeout(() => copyBtn.classList.remove('active'), 2000);
        }).catch(() => {
          toast('Failed to copy', 'error', 2000);
        });
      },
    }, '📋');

    actions.appendChild(copyBtn);
    container.appendChild(actions);
    injectCopyButtons(bubble);
  }

  return container;
}

/**
 * Find matching closing brace using brace-counting (handles nested JSON).
 */
function _findMatchingBrace(text, start) {
  if (start >= text.length || text[start] !== '{') return start;
  let depth = 0;
  for (let pos = start; pos < text.length; pos++) {
    const ch = text[pos];
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return pos;
    }
  }
  return start;
}

/**
 * Strip Action:/function_call:/[tool] blocks with brace-counting.
 * Defense against tool call markup leaking into visible content.
 */
function _stripNestedJsonBlocks(s) {
  if (!s) return s;
  let result = '';
  let i = 0;
  while (i < s.length) {
    // Pattern: Action: name\nAction Input: {nested-json}
    const actionMatch = s.slice(i).match(/^Action:\s*\w+\s*\nAction\s*Input:\s*/);
    if (actionMatch) {
      const jsonStart = i + actionMatch[0].length;
      if (jsonStart < s.length && s[jsonStart] === '{') {
        const jsonEnd = _findMatchingBrace(s, jsonStart);
        if (jsonEnd > jsonStart) {
          i = jsonEnd + 1;
          continue;
        }
      }
    }
    // Pattern: function_call: tool_name {nested-json}
    const funcMatch = s.slice(i).match(/^(?:function_call|Function_call|FUNCTION_CALL):\s*\w+\s*/);
    if (funcMatch) {
      const jsonStart = i + funcMatch[0].length;
      if (jsonStart < s.length && s[jsonStart] === '{') {
        const jsonEnd = _findMatchingBrace(s, jsonStart);
        if (jsonEnd > jsonStart) {
          i = jsonEnd + 1;
          continue;
        }
      }
    }
    // Pattern: [tool_name]({nested-json})
    const bracketMatch = s.slice(i).match(/^\[[\w-]+\]\(\s*/);
    if (bracketMatch) {
      const jsonStart = i + bracketMatch[0].length;
      if (jsonStart < s.length && s[jsonStart] === '{') {
        const jsonEnd = _findMatchingBrace(s, jsonStart);
        if (jsonEnd > jsonStart) {
          // Skip past optional whitespace then closing paren
          let skip = jsonEnd + 1;
          while (skip < s.length && s[skip] === ' ') skip++;
          if (skip < s.length && s[skip] === ')') {
            i = skip + 1;
            continue;
          }
          continue;
        }
      }
    }
    // Pattern: ☚JSON☛ (format 1)
    if (s[i] === '☚') {
      const closeIdx = s.indexOf('☛', i + 1);
      if (closeIdx > i) {
        i = closeIdx + 1;
        continue;
      }
    }
    result += s[i];
    i++;
  }
  return result;
}

/**
 * Strip DSML/XML/tool call markup from assistant content.
 * Defense-in-depth: catches any markup that leaks past backend stripping.
 */
function _stripDSML(s) {
  if (!s) return s;
  // First pass: strip brace-counted JSON blocks (Action:/function_call:/[tool])
  s = _stripNestedJsonBlocks(s);
  // Second pass: strip XML/DSML tags
  return s
    // Strip <|DSML|tag>...</|DSML|tag> blocks (LLM-generated DSML pipe format)
    // Unified character class matches engine.py: | ｜ ¦ │ ❘ ǀ ╎ ⎸ ￨ ￤
    .replace(/<[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+(?:\s+[^>]*)?>[\s\S]*?<\/[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+>/gi, '')
    // Strip self-closing <|DSML|tag />
    .replace(/<[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+(?:\s+[^>]*)?\/>/gi, '')
    // Strip standalone opening <|DSML|tag>
    .replace(/<[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+(?:\s+[^>]*)?>/gi, '')
    // Strip <DSML_tag>...</DSML_tag> format
    .replace(/<DSML_\w+[^>]*>[\s\S]*?<\/DSML_\w+>/gi, '')
    // Strip <tool_call>...</tool_call> blocks
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '')
    // Strip <invoke name="...">...</invoke> blocks
    .replace(/<invoke[^>]*>[\s\S]*?<\/invoke>/gi, '')
    // Strip <tool_calls>...</tool_calls> wrapper
    .replace(/<tool_calls>[\s\S]*?<\/tool_calls>/gi, '')
    // Strip remaining DSML parameter/result tags
    .replace(/<(parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|invoke)[^>]*>[\s\S]*?<\/\1>/gi, '')
    // Strip any remaining standalone XML tags
    .replace(/<[\w-]+(?:\s+[^>]*)?\/?>/g, '')
    // Strip nexus3d function-call style: nexus3d_create_mesh(shape='sphere')
    // The system prompt documents tools in this format; text-only models
    // output calls as visible text that would otherwise leak to chat.
    .replace(/\bnexus3d_\w+\([^)]*\)\s*/g, '')
    // Collapse multiple newlines
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function appendMessage(role, content, attachments = [], animate = true) {
  const emptyEl = msgsEl.querySelector('.empty');
  if (emptyEl) emptyEl.remove();

  // Defense-in-depth: strip DSML markup from assistant content
  if (role === 'assistant' && content) {
    content = _stripDSML(String(content));
  }

  const container = buildMessageEl(role, content, attachments, animate);
  msgsEl.appendChild(container);
  requestAnimationFrame(() => {
    msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: 'auto' });
  });
}

function fmtBytes(b) {
  if (!b) return '';
  if (b < 1024) return b + 'B';
  if (b < 1048576) return (b/1024).toFixed(1) + 'KB';
  return (b/1048576).toFixed(1) + 'MB';
}

function appendEvent(kind, content) {
  const wrap = el('div', { class: 'msg event' });
  wrap.appendChild(el('div', { class: 'msg-event ' + kind }, content));
  msgsEl.appendChild(wrap);
}

function appendCircuitBreakerBanner(message) {
  const wrap = el('div', { class: 'msg event' });
  const banner = el('div', { class: 'circuit-breaker-banner' },
    el('div', { class: 'cb-icon' }, '⚠'),
    el('div', { class: 'cb-content' },
      el('div', { class: 'cb-title' }, 'Model Temporarily Unavailable'),
      el('div', { class: 'cb-msg' }, message),
      el('button', {
        class: 'cb-retry-btn',
        onclick: () => {
          wrap.remove();
          sendBtn?.click();
        },
      }, 'Retry')
    )
  );
  wrap.appendChild(banner);
  msgsEl?.appendChild(wrap);
}



async function send() {
  // v38: Prevent double-sends — if a request is already in-flight, ignore
  if (_sendingActive) return;
  
  const text = inputEl?.value.trim();
  const hasFiles = _pendingFiles.length > 0;
  if (!text && !hasFiles) return;

  // Save image file references BEFORE uploadPendingFiles() clears _pendingFiles
  // so we can extract base64 for local Florence-2 analysis after upload completes.
  let images = null;
  if (hasFiles) {
    const imageExts = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'];
    const imageFilesForVision = _pendingFiles.filter(f => imageExts.includes(f.name.split('.').pop().toLowerCase()));
    if (imageFilesForVision.length > 0) {
      const base64List = [];
      for (const f of imageFilesForVision) {
        try {
          const b64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(f);
          });
          base64List.push(b64);
        } catch(e) { console.error('Failed to read image:', f.name, e); }
      }
      if (base64List.length > 0) images = base64List;
    }
  }

  let uploadedInfo = [];
  if (hasFiles) {
    uploadedInfo = await uploadPendingFiles(state.sessionId);
  }

  let fullText = text;
  if (uploadedInfo.length) {
    const fileList = uploadedInfo.map(f => '[File: ' + f.name + ', saved to workspace/' + f.path + ']').join('\n');
    fullText = (text ? text + '\n\n' : '') + fileList;
  }

  appendMessage('user', text || uploadedInfo.map(f=>f.name).join(', '), uploadedInfo);
  smartScrollToBottom(true);
  if (inputEl) { inputEl.value = ''; inputEl.style.height = 'auto'; }
  const counter = document.getElementById('char-count');
  if (counter) { counter.textContent = ''; counter.className = ''; }

  const analysisResult = await preAnalyzeMessage(fullText);
  
  const intgCtx = getIntegrationContext();
  const trigCtx = getTriggerContext();
  const contextBlocks = [intgCtx, trigCtx].filter(Boolean).join('\n\n');
  const finalMessage = contextBlocks ? contextBlocks + '\n\n---\n' + fullText : fullText;

  setSending(true);
  startReasoningGroup();
  showTyping(msgsEl);
  smartScrollToBottom(true);

  const ok = wsSend({
    type: 'chat',
    session_id: state.sessionId,
    message: finalMessage || text,
    images: images,
  });

  if (!ok) {
    try {
      const res = await api('/api/chat', {
        method: 'POST',
        body: { session_id: state.sessionId, message: finalMessage, images: images },
      });
      hideThinking();
      hideTyping();
      setSending(false);
      _thinkingPhase = null;
      _thinkingStepCount = 0;
      _thinkingTotalSteps = null;
      appendMessage('assistant', res.answer || '(no response)');
      smartScrollToBottom(true);
    } catch (e) {
      hideThinking();
      hideTyping();
      setSending(false);
      _thinkingPhase = null;
      _thinkingStepCount = 0;
      _thinkingTotalSteps = null;
      appendEvent('error', 'Request failed: ' + e.message);
      toast('Request failed: ' + e.message, 'error');
      smartScrollToBottom(true);
    }
  }
}