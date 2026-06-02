/**
 * Task Complexity Analyzer v1.0
 * 
 * Analyzes user input before sending to determine if the task needs
 * enhanced agent configuration. Shows a suggestion popup for complex tasks.
 */
import { api, el } from './utils.js';
import { toast } from './enhancements.js';

// ── Complexity level display names ────────────────────────────────────────
const COMPLEXITY_LABELS = {
    lite: { label: 'Lite', color: '#5b5bd6', icon: '\u26A1' },
    moderate: { label: 'Moderate', color: '#f5a623', icon: '\uD83C\uDFAF' },
    complex: { label: 'Complex', color: '#f97316', icon: '\uD83E\uDDE0' },
};

const REASONING_LABELS = {
    nexus: { label: 'Nexus Framework', desc: 'Adaptive reasoning — automatically selects the best strategy per step' },
};

const STRATEGY_LABELS = REASONING_LABELS;  // Alias for clarity


/**
 * Analyze task complexity by calling the backend API.
 * Returns the analysis result or null on failure.
 */
export async function analyzeTask(message) {
    try {
        const res = await api('/api/analyze-task', {
            method: 'POST',
            body: { message },
        });
        return res;
    } catch (e) {
        console.warn('[task-complexity] Analysis failed:', e);
        return null;
    }
}


/**
 * Apply suggested settings to the backend.
 */
export async function applySuggestedSettings(reasoningMode, enableMultiAgent) {
    // Nexus Framework: adaptive reasoning handles all strategies automatically
    return { ok: true, reasoning_strategy: 'nexus' };
}


/**
 * Show the task complexity suggestion popup.
 * Returns a promise that resolves with:
 *  { applied: true, settings: {...} } if user clicked OK (settings applied)
 *  { applied: false } if user clicked Continue with defaults
 * 
 * @param {Object} analysis - The analysis result from /api/analyze-task
 */
export function showComplexityPopup(analysis) {
    return new Promise((resolve) => {
        const modal = document.getElementById('modal-root');
        if (!modal) { resolve({ applied: false }); return; }

        const comp = COMPLEXITY_LABELS[analysis.complexity] || COMPLEXITY_LABELS.lite;
        const suggestedReasoning = STRATEGY_LABELS[analysis.reasoning_strategy] || STRATEGY_LABELS.nexus;
        const currentReasoning = STRATEGY_LABELS.nexus;  // Always nexus

        const overlay = el('div', { class: 'tc-overlay' });
        
        const dialog = el('div', { class: 'tc-dialog' },
            
            // Header
            el('div', { class: 'tc-header' },
                el('div', { class: 'tc-badge', style: { borderColor: comp.color, color: comp.color } },
                    el('span', { class: 'tc-badge-icon' }, comp.icon),
                    el('span', {}, comp.label),
                ),
                el('div', { class: 'tc-title' }, 'Task Optimization Suggestion'),
                el('div', { class: 'tc-subtitle' }, 
                    'This task may benefit from enhanced agent configuration'
                ),
            ),
            
            // Explanation
            el('div', { class: 'tc-explanation' },
                el('p', {}, analysis.explanation),
            ),

            // Confidence bar
            el('div', { class: 'tc-confidence' },
                el('div', { class: 'tc-confidence-label' },
                    el('span', {}, 'Confidence'),
                    el('span', { class: 'tc-confidence-value' }, 
                        Math.round(analysis.confidence * 100) + '%'
                    ),
                ),
                el('div', { class: 'tc-confidence-bar' },
                    el('div', { 
                        class: 'tc-confidence-fill', 
                        style: { 
                            width: (analysis.confidence * 100) + '%',
                            background: comp.color,
                        } 
                    }),
                ),
            ),

            // Settings comparison
            el('div', { class: 'tc-settings' },
                // Reasoning Mode
                el('div', { class: 'tc-setting-row' },
                    el('div', { class: 'tc-setting-label' }, 'Reasoning Strategy'),
                    el('div', { class: 'tc-setting-change' },
                        el('span', { class: 'tc-setting-current' }, 
                            'Current: ' + currentReasoning.label
                        ),
                        el('span', { class: 'tc-setting-arrow' }, '\u2192'),
                        el('span', { class: 'tc-setting-suggested' }, 
                            suggestedReasoning.label
                        ),
                    ),
                    el('div', { class: 'tc-setting-desc' }, suggestedReasoning.desc),
                ),
                
                // Multi-Agent
                el('div', { class: 'tc-setting-row' },
                    el('div', { class: 'tc-setting-label' }, 'Multi-Agent Mode'),
                    el('div', { class: 'tc-setting-change' },
                        el('span', { class: 'tc-setting-current' },
                            'Current: ' + (analysis.current_multi_agent ? 'ON' : 'OFF')
                        ),
                        el('span', { class: 'tc-setting-arrow' }, '\u2192'),
                        el('span', { class: 'tc-setting-suggested' },
                            analysis.enable_multi_agent ? 'ON' : 'OFF'
                        ),
                    ),
                    el('div', { class: 'tc-setting-desc' }, 
                        analysis.enable_multi_agent 
                            ? 'Enables coordinated planning, execution, and quality critique'
                            : 'Single-agent mode is sufficient for this task'
                    ),
                ),
            ),

            // Signals
            ...(analysis.signals && analysis.signals.length > 0 ? [
                el('div', { class: 'tc-signals' },
                    el('span', { class: 'tc-signals-label' }, 'Detected signals:'),
                    el('div', { class: 'tc-signals-list' },
                        ...analysis.signals.slice(0, 6).map(s =>
                            el('span', { class: 'tc-signal-chip' }, s)
                        ),
                    ),
                )
            ] : []),

            // Actions
            el('div', { class: 'tc-actions' },
                el('button', {
                    class: 'tc-btn tc-btn-secondary',
                    id: '_tc-cancel',
                }, 'Continue with Defaults'),
                el('button', {
                    class: 'tc-btn tc-btn-primary',
                    id: '_tc-apply',
                    style: { background: comp.color, color: '#000' },
                }, 'Apply & Continue'),
            ),
        );

        overlay.appendChild(dialog);
        modal.appendChild(overlay);

        // Animate in
        requestAnimationFrame(() => overlay.classList.add('visible'));

        const close = (result) => {
            overlay.classList.remove('visible');
            overlay.addEventListener('transitionend', () => {
                overlay.remove();
                resolve(result);
            }, { once: true });
            // Fallback remove
            setTimeout(() => {
                if (overlay.parentNode) overlay.remove();
                resolve(result);
            }, 300);
        };

        // Apply button — apply settings then resolve
        dialog.querySelector('#_tc-apply').addEventListener('click', async () => {
            const applyBtn = dialog.querySelector('#_tc-apply');
            applyBtn.disabled = true;
            applyBtn.textContent = 'Applying...';
            try {
                await applySuggestedSettings(analysis.reasoning_strategy, analysis.enable_multi_agent);
                toast('Settings applied \u2014 continuing with optimized configuration', 'success', 3000);
                close({ applied: true, settings: { reasoning_strategy: analysis.reasoning_strategy, enable_multi_agent: analysis.enable_multi_agent } });
            } catch (e) {
                toast('Failed to apply settings: ' + e.message, 'error');
                applyBtn.disabled = false;
                applyBtn.textContent = 'Apply & Continue';
            }
        });

        // Continue with defaults
        dialog.querySelector('#_tc-cancel').addEventListener('click', () => {
            close({ applied: false });
        });

        // Click outside to dismiss
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) close({ applied: false });
        });

        // Escape key
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                close({ applied: false });
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    });
}


/**
 * Main entry point: analyze a message and show popup if needed.
 * Returns:
 *   { send: true, applied: boolean } — proceed with sending
 *   { send: false } — don't send (shouldn't happen normally)
 * 
 * @param {string} message - The user's message text
 * @returns {Promise<{send: boolean, applied?: boolean}>}
 */
export async function preAnalyzeMessage(message) {
    const analysis = await analyzeTask(message);
    
    if (!analysis || !analysis.needs_suggestion) {
        // Lite or moderate task that doesn't need a popup — proceed silently
        return { send: true, applied: false };
    }
    
    // Show the suggestion popup
    const result = await showComplexityPopup(analysis);
    
    if (result.applied) {
        return { send: true, applied: true };
    }
    
    return { send: true, applied: false };
}
