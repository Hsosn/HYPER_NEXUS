import { api, fmtAgo } from '../utils.js';

const CHART_BAR_COUNT = 20;
let tokenHistory = [];

export async function initMetrics() {
  await refresh();
  setInterval(refresh, 8000);
  window.addEventListener('panel:shown', (e) => { if (e.detail === 'metrics') refresh(); });
}

async function refresh() {
  try {
    const [metrics, execs, refs] = await Promise.all([
      api('/api/metrics'),
      api('/api/tool_executions'),
      api('/api/reflections'),
    ]);

    // #26 & #27: Fetch LLM status (circuit breaker + cost)
    let llmStatus = null;
    try { llmStatus = await api('/api/llm/status'); } catch (e) {}

    const usage = metrics.usage || {};
    const stats = metrics.stats || {};

    // ── KPI cards ────────────────────────────────────────
    _setVal('m-total-tokens',      _fmtNum(usage.total_tokens ?? 0));
    _setVal('m-cost',              '$' + ((usage.cost_usd ?? 0).toFixed(4)));
    _setVal('m-tools-count',       _fmtNum(stats.tool_executions ?? 0));
    _setVal('m-memories-count',    _fmtNum(stats.memories ?? 0));
    _setVal('m-sessions',          _fmtNum(stats.sessions ?? 0));
    _setVal('m-model',             (metrics.model || '—').split('/').pop().slice(0, 28));

    // ── Token breakdown ring ─────────────────────────────
    const prompt     = usage.prompt_tokens     ?? 0;
    const completion = usage.completion_tokens ?? 0;
    const total      = prompt + completion || 1;
    const pct        = Math.round((prompt / total) * 100);
    _setVal('m-prompt-tokens',     _fmtNum(prompt));
    _setVal('m-completion-tokens', _fmtNum(completion));
    const ring = document.getElementById('token-ring');
    if (ring) {
      const c  = 2 * Math.PI * 36; // circumference for r=36
      ring.style.setProperty('--dash-prompt',     (pct / 100 * c).toFixed(1));
      ring.style.setProperty('--dash-completion', ((100 - pct) / 100 * c).toFixed(1));
      ring.style.setProperty('--circ',            c.toFixed(1));
    }
    const ringLbl = document.getElementById('token-ring-label');
    if (ringLbl) ringLbl.textContent = pct + '%\nprompt';

    // ── Sparkline history ────────────────────────────────
    tokenHistory.push(usage.total_tokens ?? 0);
    if (tokenHistory.length > CHART_BAR_COUNT) tokenHistory.shift();
    _drawSparkline('token-sparkline', tokenHistory);

    // ── Tool success rate bar ────────────────────────────
    const successCount = (execs || []).filter(e => e.success).length;
    const totalExecs   = (execs || []).length || 1;
    const succRate     = Math.round((successCount / totalExecs) * 100);
    const bar = document.getElementById('success-bar-fill');
    if (bar) bar.style.width = succRate + '%';
    const barLbl = document.getElementById('success-bar-label');
    if (barLbl) barLbl.textContent = succRate + '% success (' + successCount + '/' + totalExecs + ')';

    // #26: Circuit breaker status display
    const cbSection = document.getElementById('circuit-breaker-status');
    if (cbSection && llmStatus?.circuit_breaker) {
      const cb = llmStatus.circuit_breaker;
      const models = Object.entries(cb);
      if (models.length === 0) {
        cbSection.innerHTML = '<span class="cb-all-good">All models healthy</span>';
      } else {
        cbSection.innerHTML = models.map(([model, state]) => {
          const status = state.is_open ? 'unhealthy' : 'degraded';
          const cls = state.is_open ? 'cb-unhealthy' : 'cb-degraded';
          return `<div class="cb-model-row">
            <span class="cb-model-name">${_esc(model.split('/').pop())}</span>
            <span class="cb-model-state ${cls}">${status}</span>
            <span class="cb-model-failures">${state.failures} failures${state.is_open ? ' · ' + Math.round(state.cooldown_remaining) + 's cooldown' : ''}</span>
          </div>`;
        }).join('');
      }
    }

    // #27: Enhanced cost display with daily breakdown
    if (llmStatus?.global_usage) {
      const gu = llmStatus.global_usage;
      const costEl = document.getElementById('m-cost');
      if (costEl) {
        const cost = gu.cost_usd ?? 0;
        costEl.textContent = cost > 0.01 ? '$' + cost.toFixed(2) : '$' + cost.toFixed(4);
      }
    }

    // ── Tool executions list ─────────────────────────────
    const execList = document.getElementById('tool-executions');
    if (execList) {
      execList.innerHTML = '';
      if (!(execs?.length)) {
        execList.innerHTML = _emptyHTML('No tool executions yet');
      } else {
        execs.slice(0, 15).forEach((e, i) => {
          const ok = e.success;
          const row = document.createElement('div');
          row.className = 'mx-exec-row';
          row.style.animationDelay = i * 25 + 'ms';
          row.innerHTML = `
            <span class="mx-exec-status ${ok ? 'ok' : 'fail'}">${ok ? '✓' : '✗'}</span>
            <span class="mx-exec-name">${_esc(e.tool_name)}</span>
            <span class="mx-exec-dur">${Math.round(e.duration_ms || 0)}ms</span>
            <span class="mx-exec-ago">${fmtAgo ? fmtAgo(e.created_at) : ''}</span>
          `;
          execList.appendChild(row);
        });
      }
    }

    // ── Reflections ──────────────────────────────────────
    const refList = document.getElementById('reflections');
    if (refList) {
      refList.innerHTML = '';
      const valid = (refs || []).filter(r => r.content?.length > 10);
      if (!valid.length) {
        refList.innerHTML = _emptyHTML('No reflections recorded yet');
      } else {
        valid.slice(0, 6).forEach((r, i) => {
          const row = document.createElement('div');
          row.className = 'mx-reflection-row';
          row.style.animationDelay = i * 30 + 'ms';
          row.innerHTML = `
            <span class="mx-reflection-dot"></span>
            <p class="mx-reflection-text">${_esc(r.content?.substring(0, 200))}</p>
          `;
          refList.appendChild(row);
        });
      }
    }

  } catch (e) {
    console.error('Metrics refresh error:', e);
  }
}

// ── Helpers ──────────────────────────────────────────────

function _setVal(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  const next = String(value);
  if (el.textContent === next) return;
  el.animate([{ opacity: 0, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }],
    { duration: 200, easing: 'ease', fill: 'forwards' });
  el.textContent = next;
}

function _fmtNum(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

function _drawSparkline(id, data) {
  const canvas = document.getElementById(id);
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const max = Math.max(...data, 1);
  const step = W / (data.length - 1 || 1);
  const pts = data.map((v, i) => [i * step, H - (v / max) * (H - 4) - 2]);

  // Fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(91,91,214,0.3)');
  grad.addColorStop(1, 'rgba(91,91,214,0)');
  ctx.beginPath();
  ctx.moveTo(pts[0][0], H);
  pts.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.lineTo(pts[pts.length - 1][0], H);
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
  ctx.strokeStyle = '#5b5bd6';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

function _emptyHTML(msg) {
  return `<div class="mx-empty">${_esc(msg)}</div>`;
}

function _esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str != null ? str : '');
  return d.innerHTML;
}
