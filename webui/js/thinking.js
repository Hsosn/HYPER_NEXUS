/**
 * Nexus — Thinking Block v8.1 (Optimized)
 *
 * Encapsulated, accessible, memory-safe reasoning panel.
 * v8.1 optimizations:
 * - Removed expensive `void el.offsetWidth` reflow hack from stagger
 * - Consolidated scroll RAF into single point (removed separate #scrollRAF)
 * - Removed unused microtask queue (simplified state machine)
 * - Uses CSS custom property `--stagger-delay` for choreographed entries
 */

import { _esc } from './utils.js';

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
 * Strip DSML/XML/tool call markup from tool param/output strings.
 * Comprehensive catch-all that removes ALL known DSML tags and their content.
 * This prevents DSML markup from leaking into the visible chat UI.
 */
function _stripDSML(s) {
  if (!s) return s;
  // First pass: strip brace-counted JSON blocks (Action:/function_call:/[tool])
  s = _stripNestedJsonBlocks(s);
  return s
    // Strip <|DSML|tag>...</|DSML|tag> blocks (runtime LLM-generated DSML pipe format)
    // Unified character class matches engine.py: | ｜ ¦ │ ❘ ǀ ╎ ⎸ ￨ ￤
    .replace(/<[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+(?:\s+[^>]*)?>[\s\S]*?<\/[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+>/gi, '')
    // Strip self-closing <|DSML|tag />
    .replace(/<[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+(?:\s+[^>]*)?\/>/gi, '')
    // Strip standalone opening <|DSML|tag> that has no close
    .replace(/<[|｜¦│❘ǀ╎⎸￨￤]DSML[|｜¦│❘ǀ╎⎸￨￤][\w-]+(?:\s+[^>]*)?>/gi, '')
    // Strip <DSML_tag>...</DSML_tag> format
    .replace(/<DSML_\w+[^>]*>[\s\S]*?<\/DSML_\w+>/gi, '')
    // Strip <tool_call>...</tool_call> blocks
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '')
    // Strip <invoke name="...">...</invoke> blocks (DSML format)
    .replace(/<invoke[^>]*>[\s\S]*?<\/invoke>/gi, '')
    // Strip <tool_calls>...</tool_calls> wrapper
    .replace(/<tool_calls>[\s\S]*?<\/tool_calls>/gi, '')
    // Strip individual DSML parameter/result tags and their content
    .replace(/<(parameter|tool_response|tool_result|error|result|response|tool_input|input|output|status|arg_key|arg_value|tool|function|action)[^>]*>[\s\S]*?<\/\1>/gi, '')
    // Strip OpenAI/Anthropic function call tags
    .replace(/<(function_call|function_result|conversation|message|role|content)[^>]*>[\s\S]*?<\/\1>/gi, '')
    // Strip any remaining standalone XML tags (self-closing or opening)
    .replace(/<[\w-]+(?:\s+[^>]*)?\/?>/g, '')
    // Collapse multiple newlines
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ── Constants ──────────────────────────────────────────────────────────

const ANIM_DURATION = {
  SLIDE_IN: 300,
  STAGGER: 320,
  COLLAPSE: 350,
  DONE_HOLD: 2500,
  PROGRESS: 400,
};

const MAX_STEPS = 150;          // Default cap; can be overridden via data-max-steps attribute
const MAX_CONTENT_LENGTH = 2000; // Max chars per step

const TOOL_TYPE_ICONS = {
  web: {
    keywords: ['browse', 'scrape', 'fetch', 'http', 'url', 'search', 'google', 'web'],
    icon: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10A15.3 15.3 0 0 1 12 2z"/></svg>`,
    label: 'Web',
  },
  code: {
    keywords: ['execute', 'run', 'python', 'bash', 'shell', 'eval', 'repl', 'node', 'code'],
    icon: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
    label: 'Code',
  },
  file: {
    keywords: ['read', 'write', 'file', 'open', 'save', 'create', 'delete', 'upload', 'download', 'edit'],
    icon: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
    label: 'File',
  },
  search: {
    keywords: ['search', 'find', 'grep', 'locate', 'query', 'lookup', 'index'],
    icon: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
    label: 'Search',
  },
  image: {
    keywords: ['screenshot', 'capture', 'image', 'img', 'draw', 'render', 'plot'],
    icon: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
    label: 'Image',
  },
  data: {
    keywords: ['sql', 'database', 'db', 'table', 'query', 'data', 'csv', 'json', 'parse'],
    icon: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
    label: 'Data',
  },
};

const DEFAULT_TOOL_ICON = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`;

// ── Utility: get stable reference to #messages ────────────────────────

function _getMessagesEl() {
  return document.getElementById('messages');
}

// ── Class: ThinkingBlock ───────────────────────────────────────────────

export class ThinkingBlock {
  // Private fields
  #block = null;
  #body = null;
  #steps = [];
  #placeholder = null;
  #pendingTools = new Map();
  #toolCallCounter = 0;
  #mode = 'single';
  #stepIndex = 0;
  #reasoningCounter = 0;
  #progressBar = null;
  #autoCollapseTimer = null;
  #scheduledUpdateTimer = null;
  #rafId = null;
  #messagesEl = null;
  #iconCache = new Map(); // Cache SVG elements per tool type
  #toggleHandler = null;     // Bound click handler for cleanup
  #toggleKeyHandler = null;  // Bound keydown handler for cleanup

  constructor() {
    this.#messagesEl = _getMessagesEl();
  }

  // ── Public API ────────────────────────────────────────────────────

  /** Returns whether the thinking block is currently in the DOM */
  get isVisible() {
    return this.#block !== null && this.#block.parentNode !== null;
  }

  /** Returns the current mode: 'single' | 'reasoning' */
  get mode() {
    return this.#mode;
  }

  /** Get a stable DOM reference to the block element (for external toggle) */
  get element() {
    return this.#block;
  }

  /** Show/create the thinking block. Safe to call multiple times. */
  show() {
    if (this.#block) {
      if (this.#block.classList.contains('done')) {
        this.#block.remove();
        this.#block = null;
        this.#body = null;
      } else {
        this.#block.classList.remove('collapsed');
        return;
      }
    }

    this.#resetState();

    const frag = document.createDocumentFragment();

    // Header
    const header = this.#createHeader();
    frag.appendChild(header);

    // Body
    this.#body = document.createElement('div');
    this.#body.className = 'thinking-body';
    this.#body.setAttribute('role', 'log');
    this.#body.setAttribute('aria-live', 'polite');
    this.#body.setAttribute('aria-label', 'Reasoning steps');
    frag.appendChild(this.#body);

    // Block wrapper
    this.#block = document.createElement('div');
    this.#block.className = `thinking-block mode-${this.#mode}`;
    this.#block.setAttribute('role', 'status');
    this.#block.setAttribute('aria-live', 'polite');
    this.#block.setAttribute('aria-label', 'Agent reasoning');
    this.#block.appendChild(frag);

    // Placeholder
    this.#renderPlaceholder();

    // Insert into DOM
    if (!this.#messagesEl) {
      this.#messagesEl = _getMessagesEl();
    }
    if (this.#messagesEl) {
      const typingEl = this.#messagesEl.querySelector('.typing-indicator');
      if (typingEl) {
        this.#messagesEl.insertBefore(this.#block, typingEl);
      } else {
        this.#messagesEl.appendChild(this.#block);
      }
    }

    // Toggle handler (uses event delegation on the header)
    this.#setupToggle(header);

    // Batched scroll via RAF (single point)
    this.#scheduleScroll();
  }

  /**
   * Set the display mode.
   * @param {'single'|'reasoning'} mode
   */
  setMode(mode) {
    if (mode !== 'single' && mode !== 'reasoning') return;
    this.#mode = mode;

    if (!this.#block) return;

    this.#block.classList.remove('mode-single', 'mode-reasoning');
    this.#block.classList.add(`mode-${mode}`);

    const title = this.#block.querySelector('.thinking-title');
    if (title && !this.#block.classList.contains('done')) {
      title.textContent = '';
      const pulse = document.createElement('span');
      pulse.className = 'thinking-pulse';
      const label = document.createTextNode(mode === 'reasoning' ? ' Deep Reasoning' : ' Reasoning');
      title.appendChild(pulse);
      title.appendChild(label);
    }

    this.#scheduleScroll();
  }

  /**
   * Add a progress bar to the header.
   * @param {number} current - Current step number
   * @param {number} total - Total expected steps
   */
  setProgress(current, total) {
    if (!this.#block) this.show();

    const header = this.#block.querySelector('.thinking-header');
    if (!header) return;

    if (!this.#progressBar) {
      this.#progressBar = document.createElement('div');
      this.#progressBar.className = 'thinking-progress-bar';
      this.#progressBar.innerHTML =
        '<div class="thinking-progress-fill"></div><div class="thinking-progress-text"></div>';
      header.appendChild(this.#progressBar);
    }

    const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
    const fill = this.#progressBar.querySelector('.thinking-progress-fill');
    const text = this.#progressBar.querySelector('.thinking-progress-text');

    if (fill) fill.style.width = `${pct}%`;
    if (text) text.textContent = total > 0 ? `${current}/${total}` : '';

    if (fill && pct >= 100) {
      fill.classList.add('complete');
      this.#scheduleRemoveProgressBar();
    }
  }

  /**
   * Add a thought step.
   * @param {string} content
   */
  addThought(content) {
    if (!this.#block) this.show();
    if (!this.#body) return;

    this.#removePlaceholder();

    // Strip DSML markup from thought content (e.g., step actions/results)
    content = _stripDSML(String(content));
    const text = this.#truncate(content, 200);
    const summary = text.length > 180 ? text.slice(0, 180) + '…' : text;

    const step = document.createElement('div');
    step.className = 'thinking-step thought fade-in-up';

    // XSS-safe: use _esc and textContent
    const icon = this.#createSVGIcon('thought');
    const contentSpan = document.createElement('span');
    contentSpan.className = 'step-content';
    contentSpan.style.flex = '1';
    contentSpan.style.color = 'var(--tx3)';
    contentSpan.style.fontSize = '12px';
    contentSpan.textContent = summary;

    step.appendChild(icon);
    step.appendChild(contentSpan);

    this.#applyStagger(step);
    this.#body.appendChild(step);
    this.#steps.push(step);
    this.#enforceStepCap();
    this.#updateCount();
    this.#scheduleScroll();
  }

  /**
   * Add a numbered reasoning step.
   * @param {string} content
   * @param {number} [depth=1]
   */
  addReasoningStep(content, depth = 1) {
    if (!this.#block) this.show();
    if (!this.#body) return;

    if (this.#mode !== 'reasoning') this.setMode('reasoning');
    this.#removePlaceholder();

    // Strip DSML markup from reasoning step content
    content = _stripDSML(String(content));
    this.#reasoningCounter++;
    const clamped = Math.max(1, Math.min(3, depth));
    const text = this.#truncate(content, 300);
    const summary = text.length > 250 ? text.slice(0, 250) + '…' : text;

    const step = document.createElement('div');
    step.className = `thinking-reasoning-step depth-${clamped} fade-in-up`;

    const num = document.createElement('span');
    num.className = 'reasoning-number';
    num.textContent = String(this.#reasoningCounter);

    const depthEl = document.createElement('span');
    depthEl.className = 'reasoning-depth';
    depthEl.title = `Depth ${clamped}`;
    depthEl.textContent = '▸'.repeat(clamped);

    const contentEl = document.createElement('span');
    contentEl.className = 'reasoning-content';
    contentEl.textContent = summary;

    step.appendChild(num);
    step.appendChild(depthEl);
    step.appendChild(contentEl);

    this.#applyStagger(step);
    this.#body.appendChild(step);
    this.#steps.push(step);
    this.#updateCount();
    this.#scheduleScroll();
  }

  /**
   * Add a grouped reasoning chain.
   * @param {Array<{content: string, depth?: number}>} steps
   * @param {string} [label]
   */
  addReasoningChain(steps, label) {
    if (!this.#block) this.show();
    if (!this.#body || !steps || !steps.length) return;

    if (this.#mode !== 'reasoning') this.setMode('reasoning');
    this.#removePlaceholder();

    const chain = document.createElement('div');
    chain.className = 'thinking-chain';

    const header = document.createElement('div');
    header.className = 'thinking-chain-header';
    header.setAttribute('role', 'button');
    header.setAttribute('aria-expanded', 'true');
    header.setAttribute('tabindex', '0');

    const toggle = document.createElement('span');
    toggle.className = 'chain-toggle';
    toggle.textContent = '▼';

    const labelEl = document.createElement('span');
    labelEl.className = 'chain-label';
    labelEl.textContent = label || `Reasoning Chain #${this.#reasoningCounter + 1}`;

    const countEl = document.createElement('span');
    countEl.className = 'chain-count';
    countEl.textContent = `${steps.length} step${steps.length !== 1 ? 's' : ''}`;

    header.appendChild(toggle);
    header.appendChild(labelEl);
    header.appendChild(countEl);
    chain.appendChild(header);

    const body = document.createElement('div');
    body.className = 'thinking-chain-body';

    steps.forEach((s) => {
      this.#reasoningCounter++;
      const d = Math.max(1, Math.min(3, s.depth || 1));
      const text = this.#truncate(s.content || '', 250);
      const summary = text.length > 200 ? text.slice(0, 200) + '…' : text;

      const el = document.createElement('div');
      el.className = `thinking-reasoning-step depth-${d}`;

      const num = document.createElement('span');
      num.className = 'reasoning-number';
      num.textContent = String(this.#reasoningCounter);

      const depthEl = document.createElement('span');
      depthEl.className = 'reasoning-depth';
      depthEl.textContent = '▸'.repeat(d);

      const contentEl = document.createElement('span');
      contentEl.className = 'reasoning-content';
      contentEl.textContent = summary;

      el.appendChild(num);
      el.appendChild(depthEl);
      el.appendChild(contentEl);
      body.appendChild(el);
      this.#steps.push(el);
    });

    chain.appendChild(body);
    this.#applyStagger(chain);
    this.#body.appendChild(chain);
    this.#updateCount();
    this.#scheduleScroll();

    // Collapsible header
    header.addEventListener('click', () => {
      chain.classList.toggle('chain-collapsed');
      header.setAttribute('aria-expanded', String(!chain.classList.contains('chain-collapsed')));
    });
    header.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); header.click(); }
    });
  }

  /**
   * Add an LLM call step.
   * @param {number} [step]
   */
  addLlmCallStep(step) {
    if (!this.#block) this.show();
    if (!this.#body) return;

    this.#removePlaceholder();

    const el = document.createElement('div');
    el.className = 'thinking-step llm-call fade-in-up';

    const icon = this.#createSVGIcon('llm');
    const label = document.createElement('span');
    label.className = 'step-content';
    label.style.flex = '1';
    label.style.color = 'var(--a)';
    const suffix = step != null ? ` (step ${step + 1})` : '';
    label.textContent = `Calling LLM${suffix}…`;

    el.appendChild(icon);
    el.appendChild(label);

    this.#applyStagger(el);
    this.#body.appendChild(el);
    this.#steps.push(el);
    this.#updateCount();
    this.#scheduleScroll();
  }

  /**
   * Add a phase divider.
   * @param {string} label
   * @param {string} [color='var(--a)']
   * @param {string|null} [icon]
   */
  addPhase(label, color = 'var(--a)', icon = null) {
    if (!this.#block) this.show();
    if (!this.#body) return;

    this.#removePlaceholder();

    const phase = document.createElement('div');
    phase.className = 'thinking-phase fade-in-up';
    phase.dataset.phase = label.toLowerCase();
    phase.style.setProperty('--phase-color', color);

    if (icon) {
      phase.innerHTML = `<span class="thinking-step-label">${_esc(icon)} ${_esc(label)}</span>`;
    } else {
      phase.innerHTML = `<span class="thinking-phase-label">${_esc(label)}</span>`;
    }

    this.#body.appendChild(phase);
  }

  /**
   * Add an agent step.
   * @param {string} label
   * @param {string} [icon='🤖']
   */
  addAgentStep(label, icon = '🤖') {
    if (!this.#block) this.show();
    if (!this.#body) return;

    this.#removePlaceholder();

    const el = document.createElement('div');
    el.className = 'thinking-step agent-step fade-in-up';

    const iconEl = this.#createSVGIcon('agent');
    const labelEl = document.createElement('span');
    labelEl.className = 'step-content';
    labelEl.style.flex = '1';
    labelEl.style.color = 'var(--blue)';
    labelEl.textContent = label;

    el.appendChild(iconEl);
    el.appendChild(labelEl);

    this.#applyStagger(el);
    this.#body.appendChild(el);
    this.#steps.push(el);
    this.#updateCount();
    this.#scheduleScroll();
  }

  /**
   * Add a tool use step. Handles pending → completed lifecycle.
   * @param {string} name
   * @param {*} params
   * @param {*} output
   * @param {number} duration
   * @param {boolean|undefined} success — undefined = pending
   */
  addToolUse(name, params, output, duration, success) {
    if (!this.#block) this.show();
    if (!this.#body) return;

    this.#removePlaceholder();

    const isPending = success === undefined || (success === null && output == null);

    // Update existing pending tool
    if (!isPending) {
      const updated = this.#tryUpdatePendingTool(name, output, duration, success);
      if (updated) return;
    }

    // New tool entry
    const el = document.createElement('div');
    const toolId = ++this.#toolCallCounter;
    el._toolName = name;
    el._toolCallId = toolId;

    this.#renderTool(el, name, params, output, duration, isPending ? undefined : success);
    this.#applyStagger(el);
    this.#body.appendChild(el);
    this.#steps.push(el);
    if (isPending) this.#pendingTools.set(toolId, el);
    this.#updateCount();
    this.#scheduleScroll();
  }

  /**
   * Mark as done. Auto-collapses after hold duration.
   */
  done() {
    if (!this.#block) return;

    this.#removePlaceholder();
    this.#clearAutoCollapseTimer();

    // Update title
    const title = this.#block.querySelector('.thinking-title');
    if (title) {
      title.innerHTML = '';
      const check = document.createElement('span');
      check.className = 'thinking-done-check check-draw';
      check.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
      title.appendChild(check);
      title.appendChild(document.createTextNode(' Done'));
    }

    this.#block.classList.add('done');
    this.#setupToggle(this.#block.querySelector('.thinking-header'));

    // Celebration: small particle burst on completion
    this.#emitDoneCelebration();

    // Auto-collapse
    this.#autoCollapseTimer = setTimeout(() => {
      if (this.#block) this.#block.classList.add('collapsed');
    }, ANIM_DURATION.DONE_HOLD);

    this.#clearPendingTools();
    this.#progressBar = null;
  }

  /**
   * Error state. Keeps content visible for expansion.
   */
  error() {
    if (!this.#block) return;
    this.#block.classList.add('done', 'collapsed');
    this.#block.classList.add('shake-error');
    this.#block.addEventListener('animationend', () => {
      this.#block?.classList.remove('shake-error');
    }, { once: true });
    this.#setupToggle(this.#block.querySelector('.thinking-header'));
  }

  /**
   * Full rebuild — preserves container, resets content.
   */
  reset() {
    if (!this.#block) {
      this.show();
      return;
    }

    if (this.#body) this.#body.innerHTML = '';
    this.#steps = [];
    this.#pendingTools.clear();
    this.#toolCallCounter = 0;
    this.#stepIndex = 0;
    this.#reasoningCounter = 0;
    this.#placeholder = null;
    this.#clearAutoCollapseTimer();

    const title = this.#block.querySelector('.thinking-title');
    if (title) {
      title.innerHTML = '';
      const pulse = document.createElement('span');
      pulse.className = 'thinking-pulse';
      title.appendChild(pulse);
      title.appendChild(document.createTextNode(' Thinking…'));
    }

    this.#block.classList.remove('done', 'collapsed');
    this.#removeProgressBar();
    this.#renderPlaceholder();
    this.#updateCount();
    this.#scheduleScroll();
  }

  /**
   * Remove block from DOM, dispose all resources.
   */
  dispose() {
    // Clean up header event listeners before removing block
    if (this.#block) {
      const header = this.#block.querySelector('.thinking-header');
      if (header && this.#toggleHandler) {
        header.removeEventListener('click', this.#toggleHandler);
        header.removeEventListener('keydown', this.#toggleKeyHandler);
      }
    }
    this.#removeBlock();
    this.#clearAutoCollapseTimer();
    this.#cancelScheduledUpdates();
    this.#toggleHandler = null;
    this.#toggleKeyHandler = null;
    this.#block = null;
    this.#body = null;
    this.#steps = [];
    this.#pendingTools.clear();
    this.#placeholder = null;
    this.#progressBar = null;
    this.#messagesEl = null;
  }

  /**
   * Clear and remove completely (alias for dispose).
   */
  clear() {
    this.dispose();
  }

  // ── Private helpers ───────────────────────────────────────────────

  #resetState() {
    // Reset internal arrays/counters
    this.#steps = [];
    this.#pendingTools.clear();
    this.#toolCallCounter = 0;
    this.#stepIndex = 0;
    this.#reasoningCounter = 0;
    this.#placeholder = null;
    this.#clearAutoCollapseTimer();
    this.#removeProgressBar();
    this.#progressBar = null;
    // Allow overriding the step cap via a data attribute on the block (useful for testing or large sessions)
    if (this.#block && this.#block.dataset.maxSteps) {
      const parsed = parseInt(this.#block.dataset.maxSteps, 10);
      if (!isNaN(parsed) && parsed > 0) {
        // eslint-disable-next-line no-global-assign
        MAX_STEPS = parsed;
      }
    }
  }

  #createHeader() {
    const header = document.createElement('div');
    header.className = 'thinking-header';
    header.setAttribute('role', 'button');
    header.setAttribute('aria-expanded', 'true');
    header.setAttribute('tabindex', '0');

    const chevron = document.createElement('span');
    chevron.className = 'thinking-chevron';
    chevron.textContent = '▶';

    const title = document.createElement('span');
    title.className = 'thinking-title';
    const pulse = document.createElement('span');
    pulse.className = 'thinking-pulse';
    const label = document.createTextNode(
      this.#mode === 'reasoning' ? ' Deep Reasoning' : ' Reasoning'
    );
    title.appendChild(pulse);
    title.appendChild(label);

    const count = document.createElement('span');
    count.className = 'thinking-count';

    header.appendChild(chevron);
    header.appendChild(title);
    header.appendChild(count);

    return header;
  }

  #createSVGIcon(type) {
    // Reuse cached SVG elements when possible to avoid reparsing strings.
    if (this.#iconCache.has(type)) {
      return this.#iconCache.get(type).cloneNode(true);
    }
    const span = document.createElement('span');
    span.className = 'step-icon';

    let svg;
    switch (type) {
      case 'thought':
        svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>';
        break;
      case 'llm':
        svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 0 1 10 10 10 10 0 0 1-10 10 10 10 0 0 1-10-10 10 10 0 0 1 10-10z"/><path d="M12 6v6l3 2"/></svg>';
        break;
      case 'agent':
        svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/></svg>';
        break;
      default:
        svg = DEFAULT_TOOL_ICON;
    }
    span.innerHTML = svg; // eslint-disable-line no-restricted-properties — trusted static SVG
    // Cache a clone for future reuse
    this.#iconCache.set(type, span.cloneNode(true));
    return span;
  }

  #renderPlaceholder() {
    if (this.#placeholder || !this.#body) return;

    this.#placeholder = document.createElement('div');
    this.#placeholder.className = 'thinking-placeholder';

    const icon = document.createElement('span');
    icon.className = 'thinking-placeholder-icon';
    icon.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>'; // eslint-disable-line

    const text = document.createTextNode('Waiting for agent');

    const dots = document.createElement('span');
    dots.className = 'thinking-dots';
    dots.innerHTML = '<span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>'; // eslint-disable-line

    this.#placeholder.appendChild(icon);
    this.#placeholder.appendChild(text);
    this.#placeholder.appendChild(dots);
    this.#body.appendChild(this.#placeholder);
  }

  /** Emit a spray of celebration particles on done() — animationend cleanup */
  #emitDoneCelebration() {
    if (!this.#block) return;
    const colors = ['#17c964', '#34d399', '#6366f1', '#06b6d4', '#f5a623', '#f31260'];
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 10; i++) {
      const p = document.createElement('div');
      p.className = 'done-particle';
      const c = colors[i % colors.length];
      const tx = (Math.random() - 0.5) * 80;
      const ty = -20 - Math.random() * 50;
      p.style.setProperty('--tx', `${tx}px`);
      p.style.setProperty('--ty', `${ty}px`);
      p.style.background = c;
      p.style.boxShadow = `0 0 6px ${c}`;
      p.style.left = `${20 + Math.random() * 60}%`;
      p.style.top = '50%';
      p.style.animationDelay = `${Math.random() * 200}ms`;
      p.addEventListener('animationend', () => p.remove(), { once: true });
      frag.appendChild(p);
    }
    this.#block.appendChild(frag);
  }

  #removePlaceholder() {
    const ph = this.#placeholder;
    if (!ph) return;
    // Force reflow to ensure transition runs reliably
    void ph.offsetHeight;
    ph.style.opacity = '0';
    ph.style.transform = 'translateY(-4px)';
    ph.addEventListener('transitionend', () => {
      if (ph.parentNode) ph.remove();
      if (this.#placeholder === ph) this.#placeholder = null;
    }, { once: true });
  }

  #renderTool(el, name, params, output, duration, success) {
    const pending = success === undefined;
    const isError = success === false;
    const isSuccess = success === true;

    // Detect tool type
    const toolType = this.#detectToolType(name);
    const typeIcon = toolType ? toolType.icon : DEFAULT_TOOL_ICON;
    const typeLabel = toolType ? toolType.label : null;

    // State icon
    let stateIconHTML = '';
    if (pending) {
      stateIconHTML = '<span class="tool-spinner"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg></span>';
    } else if (isSuccess) {
      stateIconHTML = '<span class="tool-success-icon check-draw"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg></span>';
    } else {
      stateIconHTML = '<span class="tool-error-icon"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg></span>';
    }

    // CSS classes
    if (pending) {
      el.className = 'thinking-step tool pending';
    } else if (isError) {
      el.className = 'thinking-step tool error shake-error';
      setTimeout(() => el.classList.remove('shake-error'), 500);
    } else {
      el.className = 'thinking-step tool success tool-card-reveal';
    }

    // Build HTML safely
    let html = `<span class="step-icon tool-type-icon">${typeIcon}</span>`;
    html += `<span class="step-name">${_esc(name)}</span>`;

    if (typeLabel) {
      html += `<span class="tool-type-badge">${_esc(typeLabel)}</span>`;
    }
    html += `<span class="tool-state-icon">${stateIconHTML}</span>`;

    if (pending) {
      html += '<div class="tool-skeleton"><div class="tool-skeleton-line tool-skeleton-line-1"></div><div class="tool-skeleton-line tool-skeleton-line-2"></div></div>';
    }

    if (params && !pending) {
      let s = typeof params === 'string' ? params : JSON.stringify(params);
      // Safety truncation: extremely large params can cause JSON parser buffer
      // issues in the frontend. Params are only displayed as a 60-char preview,
      // so truncating early prevents unnecessary processing of huge strings.
      if (s.length > 2000) {
        s = s.slice(0, 2000) + '…';
      }
      s = _stripDSML(s);
      if (s && s !== '{}') {
        const short = s.length > 60 ? s.slice(0, 60) + '…' : s;
        html += `<span class="step-params">${_esc(short)}</span>`;
      }
    }

    if (output && !pending) {
      let str = String(output);
      str = _stripDSML(str);
      if (str) {
        const isBig = str.includes('data:image') || str.length > 150;
        if (isError) {
          const shortErr = str.length > 100 ? str.slice(0, 100) + '…' : str;
          html += `<span class="step-output tool-error-output">${_esc(shortErr)}</span>`;
        } else {
          const shortOut = str.length > 150 ? str.slice(0, 150) + '…' : str;
          html += `<span class="step-output">${
            isBig ? (str.includes('Screenshot') ? 'Screenshot saved' : 'Done') : _esc(shortOut)
          }</span>`;
        }
      }
    }

    if (duration != null) {
      const durClass = isSuccess ? 'success' : isError ? ' error' : '';
      html += `<span class="step-duration ${durClass}">${Math.round(duration)}ms</span>`;
    }

    el.innerHTML = html; // eslint-disable-line no-restricted-properties — sanitized via _esc()

    // Emit success sparkle particles AFTER innerHTML so they aren't wiped
    if (isSuccess) {
      this.#emitSuccessSparkle(el);
    }
  }

  #tryUpdatePendingTool(name, output, duration, success) {
    if (this.#pendingTools.size === 0) return false;

    // Find the most recent pending entry for this tool name (by call id)
    let targetId = null;
    let targetEl = null;
    for (const [id, existing] of this.#pendingTools) {
      if (existing._toolName === name && (targetId === null || id > targetId)) {
        targetId = id;
        targetEl = existing;
      }
    }
    if (targetEl && targetId !== null) {
      this.#pendingTools.delete(targetId);
      this.#renderTool(targetEl, name, null, output, duration, success);
      this.#updateCount();
      this.#scheduleScroll();
      return true;
    }
    return false;
  }

  #clearPendingTools() {
    this.#pendingTools.clear();
  }

  #removeProgressBar() {
    if (this.#progressBar && this.#progressBar.parentNode) {
      this.#progressBar.remove();
    }
    this.#progressBar = null;
  }

  #removeBlock() {
    if (this.#block && this.#block.parentNode) {
      this.#block.remove();
    }
  }

  #enforceStepCap() {
    while (this.#steps.length > MAX_STEPS && this.#body.firstChild) {
      this.#body.removeChild(this.#body.firstChild);
      this.#steps.shift();
    }
  }

  #truncate(str, max) {
    const s = String(str || '');
    return s.length > max ? s.slice(0, max) + '…' : s;
  }

  #detectToolType(name) {
    if (!name) return null;
    const lower = name.toLowerCase();
    for (const [type, config] of Object.entries(TOOL_TYPE_ICONS)) {
      if (config.keywords.some(kw => lower.includes(kw))) {
        return { type, ...config };
      }
    }
    return null;
  }

  #updateCount() {
    if (!this.#block) return;
    const n = this.#steps.length;
    const el = this.#block.querySelector('.thinking-count');
    if (!el) return;
    el.textContent = n > 0 ? `${n} step${n !== 1 ? 's' : ''}` : '';
    // Trigger bump animation — re-trigger by removing/adding class
    el.classList.remove('bump');
    // Use requestAnimationFrame to ensure the DOM has processed the removal
    requestAnimationFrame(() => {
      if (el) el.classList.add('bump');
    });
  }

  #applyStagger(el) {
    const idx = this.#stepIndex++;
    const delay = Math.min(idx * 18, 500);
    el.style.animationDelay = `${delay}ms`;
    el.classList.add('fade-in-up');
    // Add 3D perspective depth to step entrance
    el.style.setProperty('--stagger-rot', `${(idx % 5) * 0.3 - 0.6}deg`);
    // Emit a tiny particle burst on each step addition
    this.#emitParticleBurst(el);
  }

  /** Emit sparkle particles on successful tool completion */
  #emitSuccessSparkle(el) {
    if (!el || !this.#block) return;
    const colors = ['#17c964', '#34d399', '#6366f1'];
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('div');
      dot.className = 'sparkle-particle';
      const c = colors[i % colors.length];
      const angle = Math.random() * 360;
      const dist = 10 + Math.random() * 16;
      const tx = Math.cos((angle * Math.PI) / 180) * dist;
      const ty = Math.sin((angle * Math.PI) / 180) * dist - 4;
      dot.style.setProperty('--tx', `${tx}px`);
      dot.style.setProperty('--ty', `${ty}px`);
      dot.style.background = c;
      dot.style.boxShadow = `0 0 4px ${c}`;
      dot.style.animationDelay = `${i * 60}ms`;
      dot.style.left = '50%';
      dot.style.top = '50%';
      dot.addEventListener('animationend', () => dot.remove(), { once: true });
      el.appendChild(dot);
    }
  }

  /** Create a burst of particles around a newly added step — animationend cleanup */
  #emitParticleBurst(el) {
    if (!el || !this.#body) return;
    for (let i = 0; i < 4; i++) {
      const dot = document.createElement('div');
      dot.className = 'step-particle';
      const angle = (i / 4) * 360 + Math.random() * 30 - 15;
      const dist = 14 + Math.random() * 12;
      const tx = Math.cos((angle * Math.PI) / 180) * dist;
      const ty = Math.sin((angle * Math.PI) / 180) * dist - 6;
      dot.style.setProperty('--tx', `${tx}px`);
      dot.style.setProperty('--ty', `${ty}px`);
      dot.style.animationDelay = `${Math.random() * 80}ms`;
      dot.style.left = '6px';
      dot.style.top = '50%';
      dot.addEventListener('animationend', () => dot.remove(), { once: true });
      el.appendChild(dot);
    }
  }

  #setupToggle(headerEl) {
    if (!headerEl) return;

    // Remove old listeners to prevent double-binding on done()/error() calls
    if (this.#toggleHandler) {
      headerEl.removeEventListener('click', this.#toggleHandler);
      headerEl.removeEventListener('keydown', this.#toggleKeyHandler);
    }

    this.#toggleHandler = (e) => {
      e.stopPropagation();
      // Ripple effect — animationend cleanup
      const ripple = document.createElement('span');
      ripple.className = 'thinking-ripple';
      const rect = headerEl.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      ripple.style.width = ripple.style.height = `${size}px`;
      ripple.style.left = `${e.clientX - rect.left - size / 2}px`;
      ripple.style.top = `${e.clientY - rect.top - size / 2}px`;
      ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
      headerEl.appendChild(ripple);
      // Toggle
      if (this.#block) this.#block.classList.toggle('collapsed');
    };

    this.#toggleKeyHandler = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (this.#block) this.#block.classList.toggle('collapsed');
      }
    };

    headerEl.addEventListener('click', this.#toggleHandler, { passive: true });
    headerEl.addEventListener('keydown', this.#toggleKeyHandler);
  }

  // ── Scrolling (RAF-batched, instant scroll) ───────────────────
  //
  // Uses scrollTop assignment (not scrollTo with smooth) so that
  // rapid incremental step additions don't create conflicting
  // smooth-scroll animations that never reach the actual bottom.
  // Forces layout reflow via offsetHeight before reading scrollHeight
  // to guarantee accuracy — elements with contain:layout style can
  // otherwise report stale scrollHeight values.

  #scheduleScroll() {
    // Cancel any pending RAF scroll task
    if (this.#rafId) cancelAnimationFrame(this.#rafId);
    // Schedule a new RAF to perform scroll after the next paint
    this.#rafId = requestAnimationFrame(() => {
      this.#rafId = null;
      // Use a micro‑task to ensure DOM mutations are flushed before reading sizes
      Promise.resolve().then(() => {
        const messagesEl = _getMessagesEl();
        if (messagesEl) {
          // Force reflow then scroll to the newest content (bottom)
          void messagesEl.offsetHeight;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        if (this.#body) {
          void this.#body.offsetHeight;
          this.#body.scrollTop = this.#body.scrollHeight;
        }
      });
    });
  }

  // ── Timer management ────────────────────────────────────────────

  #clearAutoCollapseTimer() {
    if (this.#autoCollapseTimer != null) {
      clearTimeout(this.#autoCollapseTimer);
      this.#autoCollapseTimer = null;
    }
  }

  #cancelScheduledUpdates() {
    if (this.#scheduledUpdateTimer != null) {
      clearTimeout(this.#scheduledUpdateTimer);
      this.#scheduledUpdateTimer = null;
    }
    if (this.#rafId != null) {
      cancelAnimationFrame(this.#rafId);
      this.#rafId = null;
    }
  }

  #scheduleRemoveProgressBar() {
    this.#scheduledUpdateTimer = setTimeout(() => {
      if (this.#progressBar && this.#progressBar.parentNode) {
        this.#progressBar.remove();
      }
      this.#progressBar = null;
      this.#scheduledUpdateTimer = null;
    }, 1200);
  }
}

// ── Singleton instance ────────────────────────────────────────────────────

let _instance = null;

/**
 * Get or create the singleton ThinkingBlock instance.
 * This provides the same global access pattern as v7.0 but with
 * proper encapsulation and lifecycle management.
 */
export function getThinkingBlock() {
  if (!_instance) _instance = new ThinkingBlock();
  return _instance;
}

// ── Module-level convenience functions (backward-compatible API) ─────

const _tb = () => getThinkingBlock();

export function isThinkingBlockVisible() {
  return _instance !== null && _instance.isVisible;
}

export function toggleThinkingBlock() {
  _instance?.element?.classList.toggle('collapsed');
}

export function setThinkingMode(mode) {
  _tb().setMode(mode);
}

export function showThinkingBlock() {
  return _tb().show();
}

export function resetThinkingBlock() {
  _tb().reset();
}

export function addThought(content) {
  _tb().addThought(content);
}

export function addReasoningStep(content, depth) {
  _tb().addReasoningStep(content, depth);
}

export function addReasoningChain(steps, label) {
  _tb().addReasoningChain(steps, label);
}

export function addLlmCallStep(step) {
  _tb().addLlmCallStep(step);
}

export function addPhase(label, color, icon) {
  _tb().addPhase(label, color, icon);
}

export function addAgentStep(label, icon) {
  _tb().addAgentStep(label, icon);
}

export function addToolUse(name, params, output, duration, success) {
  _tb().addToolUse(name, params, output, duration, success);
}

export function addProgress(current, total) {
  _tb().setProgress(current, total);
}

export function hideThinkingBlock() {
  _tb().done();
}

export function hideThinking() {
  _tb().error();
}

export function clearThinkingBlock() {
  if (_instance) {
    _instance.dispose();
    _instance = null;
  }
}

export function recoverThinkingBlock() {
  if (_instance && _instance.isVisible) {
    _instance.reset();
  } else {
    _tb().show();
  }
}

// ── Auto-setup event listener for custom thought events ──────────────────

if (typeof window !== 'undefined') {
  window.addEventListener('nexus:addThought', (e) => {
    addThought(e.detail);
  });
}