/**
 * Virtual Computer Panel — live VM control and display.
 */
import { on } from '../state.js';
import { el, api } from '../utils.js';

let _statusInterval = null;
let _screenshotInterval = null;
let _vmState = { running: false, container_id: '', ip: '', uptime: '' };

export function initVirtualComputer() {
  console.log('[virtual-computer] initializing...');

  const panel = document.querySelector('[data-panel="virtual-computer"]');
  if (!panel) return;

  const layout = document.getElementById('vm-layout');
  if (!layout) return;

  // Build the two-column layout
  layout.innerHTML = buildLayoutHTML();
  bindEvents(layout);
  loadStatus();

  // Listen for panel visibility to start/stop polling
  on('panel:shown', (target) => {
    if (target === 'virtual-computer') {
      // Clear existing intervals before starting new ones
      if (_statusInterval) clearInterval(_statusInterval);
      if (_screenshotInterval) clearInterval(_screenshotInterval);
      _statusInterval = setInterval(loadStatus, 5000);
      _screenshotInterval = setInterval(refreshScreenshot, 2000);
    }
  });

  on('panel:hidden', (target) => {
    if (target === 'virtual-computer') {
      if (_statusInterval) { clearInterval(_statusInterval); _statusInterval = null; }
      if (_screenshotInterval) { clearInterval(_screenshotInterval); _screenshotInterval = null; }
    }
  });

  // Also listen for WS events for vision loop updates
  on('ws:event', (msg) => {
    if (!msg || !msg.kind) return;
    if (msg.kind === 'vm_vision_loop_started') {
      updateVisionStatus(layout, 'running', msg);
    } else if (msg.kind === 'vm_vision_loop_complete') {
      updateVisionStatus(layout, 'complete', msg);
    } else if (msg.kind === 'vm_vision_loop_error') {
      updateVisionStatus(layout, 'error', msg);
    } else if (msg.kind === 'vm_started' || msg.kind === 'vm_stopped' || msg.kind === 'vm_restarted' || msg.kind === 'vm_destroyed') {
      loadStatus();
      refreshScreenshot();
    }
  });

  // Start polling if panel is already active
  if (document.querySelector('[data-panel="virtual-computer"].active')) {
    _statusInterval = setInterval(loadStatus, 5000);
    _screenshotInterval = setInterval(refreshScreenshot, 2000);
  }
}

/* ── Layout HTML ──────────────────────────────────────────────── */
function buildLayoutHTML() {
  return `
    <div class="vm-sidebar">
      <!-- Status -->
      <div class="vm-controls-group">
        <div class="vm-group-title">Status</div>
        <div class="vm-status-row">
          <span class="vm-status-dot" id="vm-status-dot"></span>
          <span id="vm-status-text">Checking...</span>
        </div>
        <div class="vm-info-grid">
          <div class="vm-info-item"><span class="vm-info-label">Container</span><span class="vm-info-value" id="vm-container-id">—</span></div>
          <div class="vm-info-item"><span class="vm-info-label">IP</span><span class="vm-info-value" id="vm-ip">—</span></div>
          <div class="vm-info-item"><span class="vm-info-label">Uptime</span><span class="vm-info-value" id="vm-uptime">—</span></div>
        </div>
      </div>

      <!-- Controls -->
      <div class="vm-controls-group">
        <div class="vm-group-title">Controls</div>
        <div class="vm-btn-row">
          <button class="vm-btn vm-btn--green" id="vm-start-btn">Start</button>
          <button class="vm-btn vm-btn--warn" id="vm-stop-btn">Stop</button>
          <button class="vm-btn vm-btn--blue" id="vm-restart-btn">Restart</button>
          <button class="vm-btn vm-btn--red" id="vm-destroy-btn">Destroy</button>
        </div>
      </div>

      <!-- Mouse -->
      <div class="vm-controls-group">
        <div class="vm-group-title">Mouse</div>
        <div class="vm-input-row">
          <label class="vm-input-label">X</label>
          <input type="number" class="vm-input" id="vm-mouse-x" value="0" style="width:70px">
          <label class="vm-input-label">Y</label>
          <input type="number" class="vm-input" id="vm-mouse-y" value="0" style="width:70px">
        </div>
        <div class="vm-btn-row">
          <button class="vm-btn" id="vm-mouse-move-btn">Move</button>
          <button class="vm-btn vm-btn--accent" id="vm-mouse-click-btn">Click</button>
        </div>
      </div>

      <!-- Keyboard -->
      <div class="vm-controls-group">
        <div class="vm-group-title">Keyboard</div>
        <div class="vm-input-row">
          <input type="text" class="vm-input vm-input--flex" id="vm-type-text" placeholder="Type text..." value="">
        </div>
        <div class="vm-btn-row">
          <button class="vm-btn vm-btn--accent" id="vm-type-btn">Type</button>
        </div>
        <div class="vm-input-row" style="margin-top:6px">
          <input type="text" class="vm-input vm-input--flex" id="vm-key-input" placeholder="Special key (Enter, Tab, Esc...)" value="">
        </div>
        <div class="vm-btn-row">
          <button class="vm-btn" id="vm-key-press-btn">Press Key</button>
        </div>
      </div>

      <!-- Command -->
      <div class="vm-controls-group">
        <div class="vm-group-title">Command</div>
        <div class="vm-input-row">
          <input type="text" class="vm-input vm-input--flex" id="vm-cmd-input" placeholder="Shell command..." value="">
        </div>
        <div class="vm-btn-row">
          <button class="vm-btn vm-btn--accent" id="vm-cmd-run-btn">Execute</button>
        </div>
        <div class="vm-output" id="vm-cmd-output"><span class="vm-output-placeholder">Command output will appear here...</span></div>
      </div>

      <!-- File Transfer -->
      <div class="vm-controls-group">
        <div class="vm-group-title">File Transfer</div>
        <div class="vm-input-row">
          <input type="text" class="vm-input" id="vm-upload-local" placeholder="Local path" style="width:48%">
          <input type="text" class="vm-input" id="vm-upload-remote" placeholder="VM path" style="width:48%">
        </div>
        <div class="vm-btn-row">
          <button class="vm-btn" id="vm-upload-btn">Upload to VM</button>
          <button class="vm-btn" id="vm-download-btn">Download from VM</button>
        </div>
      </div>

      <!-- Vision Loop -->
      <div class="vm-controls-group">
        <div class="vm-group-title">Vision Loop</div>
        <textarea class="vm-textarea" id="vm-vision-task" placeholder="Describe the task for the vision loop..." rows="3"></textarea>
        <div class="vm-btn-row" style="margin-top:6px">
          <button class="vm-btn vm-btn--accent" id="vm-vision-start-btn">Start Vision Loop</button>
        </div>
        <div class="vm-vision-status" id="vm-vision-status"></div>
      </div>
    </div>

    <div class="vm-display">
      <div class="vm-screenshot-wrap">
        <img id="vm-screenshot" class="vm-screenshot-img" alt="VM Screenshot" style="display:none">
        <div class="vm-screenshot-placeholder" id="vm-screenshot-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity=".3"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
          <span>Start the VM to see the display</span>
        </div>
      </div>
      <div class="vm-display-footer">
        <div class="vm-vnc-info" id="vm-vnc-info" style="display:none">
          <span class="vm-vnc-label">VNC:</span>
          <span id="vm-vnc-address">—</span>
        </div>
        <div class="vm-display-actions">
          <button class="vm-btn vm-btn--sm" id="vm-save-screenshot-btn">Save Screenshot</button>
        </div>
      </div>
    </div>
  `;
}

/* ── Event bindings ────────────────────────────────────────────── */
function bindEvents(layout) {
  // VM controls
  layout.querySelector('#vm-start-btn')?.addEventListener('click', () => vmAction('start'));
  layout.querySelector('#vm-stop-btn')?.addEventListener('click', () => vmAction('stop'));
  layout.querySelector('#vm-restart-btn')?.addEventListener('click', () => vmAction('restart'));
  layout.querySelector('#vm-destroy-btn')?.addEventListener('click', () => vmAction('destroy'));

  // Mouse
  layout.querySelector('#vm-mouse-move-btn')?.addEventListener('click', async () => {
    const x = parseInt(layout.querySelector('#vm-mouse-x')?.value || '0', 10);
    const y = parseInt(layout.querySelector('#vm-mouse-y')?.value || '0', 10);
    await vmAction('mouse', { action: 'move', x, y });
  });
  layout.querySelector('#vm-mouse-click-btn')?.addEventListener('click', async () => {
    const x = parseInt(layout.querySelector('#vm-mouse-x')?.value || '0', 10);
    const y = parseInt(layout.querySelector('#vm-mouse-y')?.value || '0', 10);
    await vmAction('mouse', { action: 'click', x, y });
  });

  // Keyboard
  layout.querySelector('#vm-type-btn')?.addEventListener('click', async () => {
    const text = layout.querySelector('#vm-type-text')?.value || '';
    if (!text) return;
    await vmAction('keyboard', { action: 'type', text });
  });
  layout.querySelector('#vm-key-press-btn')?.addEventListener('click', async () => {
    const key = layout.querySelector('#vm-key-input')?.value || '';
    if (!key) return;
    await vmAction('keyboard', { action: 'press', key });
  });

  // Command execution
  layout.querySelector('#vm-cmd-run-btn')?.addEventListener('click', async () => {
    const cmd = layout.querySelector('#vm-cmd-input')?.value || '';
    if (!cmd) return;
    const output = layout.querySelector('#vm-cmd-output');
    if (output) output.innerHTML = '<span class="vm-output-loading">Executing...</span>';
    try {
      const result = await api('/api/vm/execute', {
        method: 'POST',
        body: JSON.stringify({ command: cmd, timeout: 30 }),
      });
      if (output) {
        output.textContent = result.output || result.stdout || 'Command completed (no output)';
        output.scrollTop = output.scrollHeight;
      }
    } catch (e) {
      if (output) output.innerHTML = `<span class="vm-output-error">${e.message}</span>`;
    }
  });
  // Enter key on command input
  layout.querySelector('#vm-cmd-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') layout.querySelector('#vm-cmd-run-btn')?.click();
  });

  // File transfer
  layout.querySelector('#vm-upload-btn')?.addEventListener('click', async () => {
    const localPath = layout.querySelector('#vm-upload-local')?.value || '';
    const containerPath = layout.querySelector('#vm-upload-remote')?.value || '';
    if (!localPath || !containerPath) return;
    await vmAction('upload', { container_path: containerPath, local_path: localPath });
  });
  layout.querySelector('#vm-download-btn')?.addEventListener('click', async () => {
    const containerPath = layout.querySelector('#vm-upload-remote')?.value || '';
    const localPath = layout.querySelector('#vm-upload-local')?.value || '';
    if (!containerPath || !localPath) return;
    await vmAction('download', { container_path: containerPath, local_path: localPath });
  });

  // Vision loop
  layout.querySelector('#vm-vision-start-btn')?.addEventListener('click', async () => {
    const task = layout.querySelector('#vm-vision-task')?.value || '';
    if (!task) return;
    const statusEl = layout.querySelector('#vm-vision-status');
    if (statusEl) statusEl.innerHTML = '<span class="vm-output-loading">Starting vision loop...</span>';
    try {
      await api('/api/vm/vision-loop', {
        method: 'POST',
        body: JSON.stringify({ task, max_iterations: 20 }),
      });
      if (statusEl) statusEl.innerHTML = '<span class="vm-vision-running">Vision loop running...</span>';
    } catch (e) {
      if (statusEl) statusEl.innerHTML = `<span class="vm-output-error">${e.message}</span>`;
    }
  });

  // Save screenshot
  layout.querySelector('#vm-save-screenshot-btn')?.addEventListener('click', saveScreenshot);
}

/* ── API helpers ───────────────────────────────────────────────── */
async function vmAction(action, params = {}) {
  try {
    const result = await api(`/api/vm/${action}`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
    loadStatus();
    refreshScreenshot();
    return result;
  } catch (e) {
    console.error(`[vm] ${action} error:`, e);
  }
}

async function loadStatus() {
  try {
    const status = await api('/api/vm/status');
    _vmState = { running: status.running ?? false, container_id: status.container_id ?? '', ip: status.ip ?? '', uptime: status.uptime ?? '' };
    updateStatusUI();
  } catch (e) {
    _vmState.running = false;
    updateStatusUI();
  }
}

function updateStatusUI() {
  const dot = document.getElementById('vm-status-dot');
  const text = document.getElementById('vm-status-text');
  const containerId = document.getElementById('vm-container-id');
  const ip = document.getElementById('vm-ip');
  const uptime = document.getElementById('vm-uptime');
  const vncInfo = document.getElementById('vm-vnc-info');
  const vncAddr = document.getElementById('vm-vnc-address');

  if (dot) {
    dot.classList.toggle('running', _vmState.running);
    dot.classList.toggle('stopped', !_vmState.running);
  }
  if (text) text.textContent = _vmState.running ? 'Running' : 'Stopped';
  if (containerId) containerId.textContent = _vmState.container_id || '—';
  if (ip) ip.textContent = _vmState.ip || '—';
  if (uptime) uptime.textContent = _vmState.uptime || '—';

  if (vncInfo && vncAddr) {
    if (_vmState.running && _vmState.ip) {
      vncInfo.style.display = 'flex';
      vncAddr.textContent = `${_vmState.ip}:5900`;
    } else {
      vncInfo.style.display = 'none';
    }
  }
}

async function refreshScreenshot() {
  if (!_vmState.running) return;
  try {
    const data = await api('/api/vm/screenshot');
    const img = document.getElementById('vm-screenshot');
    const placeholder = document.getElementById('vm-screenshot-placeholder');
    if (data.image && img) {
      img.src = data.image;
      img.style.display = 'block';
      if (placeholder) placeholder.style.display = 'none';
    }
  } catch (e) {
    // silently ignore screenshot errors
  }
}

function saveScreenshot() {
  const img = document.getElementById('vm-screenshot');
  if (!img || !img.src) return;
  const link = document.createElement('a');
  link.download = `vm-screenshot-${Date.now()}.png`;
  link.href = img.src;
  link.click();
}

function updateVisionStatus(layout, status, msg) {
  const statusEl = layout?.querySelector('#vm-vision-status');
  if (!statusEl) return;
  if (status === 'running') {
    statusEl.innerHTML = `<span class="vm-vision-running">Vision loop running... (${msg.max_iterations ?? '?'} max iterations)</span>`;
  } else if (status === 'complete') {
    statusEl.innerHTML = `<span class="vm-vision-complete">Vision loop completed: ${msg.task ?? ''}</span>`;
  } else if (status === 'error') {
    statusEl.innerHTML = `<span class="vm-output-error">Vision loop error: ${msg.error ?? 'unknown'}</span>`;
  }
}
