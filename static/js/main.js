/**
 * SAFETYVISION AI - Core Frontend Utilities
 * Handles Web Audio alerts, Toast notifications, Drawers, Modals.
 */

// Web Audio API Industrial Sound Synthesizer
class SoundAlertManager {
    constructor() {
        this.ctx = null;
        this.muted = localStorage.getItem('sv_sound_muted') === 'true';
    }

    init() {
        if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioCtx();
        }
    }

    playChime(type = 'WARNING') {
        if (this.muted) return;
        try {
            this.init();
            if (!this.ctx) return;
            if (this.ctx.state === 'suspended') {
                this.ctx.resume();
            }

            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.connect(gain);
            gain.connect(this.ctx.destination);

            const now = this.ctx.currentTime;

            if (type === 'CRITICAL') {
                // High-urgency dual-tone siren
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(880, now); // A5
                osc.frequency.exponentialRampToValueAtTime(440, now + 0.35);
                gain.gain.setValueAtTime(0.3, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
                osc.start(now);
                osc.stop(now + 0.4);
            } else if (type === 'WARNING' || type === 'HIGH') {
                // Warning chime
                osc.type = 'sine';
                osc.frequency.setValueAtTime(587.33, now); // D5
                osc.frequency.exponentialRampToValueAtTime(880, now + 0.2);
                gain.gain.setValueAtTime(0.2, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
                osc.start(now);
                osc.stop(now + 0.3);
            } else {
                // Subtle positive blip
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(523.25, now);
                gain.gain.setValueAtTime(0.15, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
                osc.start(now);
                osc.stop(now + 0.15);
            }
        } catch (e) {
            console.warn("Audio chime prevented or unsupported:", e);
        }
    }

    toggleMute() {
        this.muted = !this.muted;
        localStorage.setItem('sv_sound_muted', this.muted);
        return this.muted;
    }
}

const SoundSystem = new SoundAlertManager();

// Toast Notification Manager
function showToast(title, message, severity = 'INFO') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast';

    let borderColor = 'var(--primary)';
    if (severity === 'CRITICAL') {
        borderColor = 'var(--critical)';
        SoundSystem.playChime('CRITICAL');
    } else if (severity === 'HIGH' || severity === 'WARNING') {
        borderColor = 'var(--warning)';
        SoundSystem.playChime('WARNING');
    } else {
        SoundSystem.playChime('INFO');
    }

    toast.style.borderLeftColor = borderColor;
    toast.innerHTML = `
        <div style="flex:1;">
            <div style="font-weight:700; font-size:0.85rem; color:var(--text-primary); margin-bottom:2px;">${title}</div>
            <div style="font-size:0.78rem; color:var(--text-secondary);">${message}</div>
        </div>
        <button onclick="this.parentElement.remove()" style="color:var(--text-muted); font-size:1.1rem; padding:0 4px;">&times;</button>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        if (!toast.parentElement) return;
        toast.classList.add('is-leaving');
        setTimeout(() => toast.remove(), 260);
    }, 4500);
}

// Side Detail Drawer for Tracked Person
function openPersonDrawer(person) {
    const drawer = document.getElementById('person-detail-drawer');
    if (!drawer) return;

    const pid = person.person_id || person.track_id || 1;
    document.getElementById('drawer-person-id').textContent = person.label || `Person #${pid}`;
    document.getElementById('drawer-compliance-val').textContent = `${person.compliance || person.compliance_score || 0}%`;
    document.getElementById('drawer-risk-val').textContent = `${person.risk_score || 0}/100`;
    
    const riskLvl = person.risk || person.risk_level || 'LOW';
    document.getElementById('drawer-risk-level').textContent = riskLvl;
    document.getElementById('drawer-risk-level').className = `badge badge-${riskLvl.toLowerCase()}`;

    // PPE Status
    const renderStatus = (val) => {
        if (val === 'PRESENT' || val === 'Present') return '<span class="badge badge-low">PRESENT</span>';
        if (val === 'MISSING' || val === 'Missing') return '<span class="badge badge-critical">MISSING</span>';
        return '<span class="badge" style="background:var(--bg-card-subtle); color:var(--text-secondary); border:1px solid var(--border-subtle);">UNKNOWN</span>';
    };

    const hh = person.hardhat || (person.ppe_status ? person.ppe_status['Hardhat'] : 'Unknown');
    const mk = person.mask || (person.ppe_status ? person.ppe_status['Mask'] : 'Unknown');
    const vt = person.vest || (person.ppe_status ? person.ppe_status['Safety Vest'] : 'Unknown');

    document.getElementById('drawer-hardhat-status').innerHTML = renderStatus(hh);
    document.getElementById('drawer-mask-status').innerHTML = renderStatus(mk);
    document.getElementById('drawer-vest-status').innerHTML = renderStatus(vt);

    // Environmental info
    document.getElementById('drawer-duration').textContent = `${person.duration_seconds || 0}s`;
    document.getElementById('drawer-zone').textContent = 'Laptop Camera View';
    document.getElementById('drawer-machinery').textContent = person.near_machinery ? 'PROXIMITY DANGER' : 'Safe Distance';
    document.getElementById('drawer-vehicle').textContent = person.near_vehicle ? 'PROXIMITY DANGER' : 'Safe Distance';

    // Active Violations List
    const vList = document.getElementById('drawer-violations-list');
    vList.innerHTML = '';
    const violations = person.missing_items || person.active_violations || [];
    if (violations.length > 0) {
        violations.forEach(v => {
            const li = document.createElement('li');
            li.style.color = 'var(--critical)';
            li.style.fontSize = '0.8rem';
            li.style.marginBottom = '4px';
            li.textContent = `⚠ Missing ${v.replace('Missing ', '')}`;
            vList.appendChild(li);
        });
    } else {
        vList.innerHTML = '<li style="color:var(--safe); font-size:0.8rem;">✓ All Required PPE Present</li>';
    }

    drawer.classList.add('open');
}

function closePersonDrawer() {
    const drawer = document.getElementById('person-detail-drawer');
    if (drawer) drawer.classList.remove('open');
}

// Global modal helpers
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.style.display = 'flex';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.style.display = 'none';
}

// User Profile Mute Button Handlers
document.addEventListener('DOMContentLoaded', () => {
    const muteBtn = document.getElementById('btn-sound-toggle');
    if (muteBtn) {
        if (SoundSystem.muted) {
            muteBtn.innerHTML = '🔇';
            muteBtn.title = 'Unmute Alert Chime';
        }
        muteBtn.addEventListener('click', () => {
            const isMuted = SoundSystem.toggleMute();
            muteBtn.innerHTML = isMuted ? '🔇' : '🔔';
            muteBtn.title = isMuted ? 'Unmute Alert Chime' : 'Mute Alert Chime';
            showToast('Audio Settings', isMuted ? 'Alert sounds muted' : 'Alert sounds enabled', 'INFO');
        });
    }
});

// ==========================================================================
// Light / Dark Theme Controller
// - Persists selection in localStorage ('sv_theme')
// - Toggles [data-theme] on <html>
// - Reflects Chart.js global defaults so axes/grid/legend stay readable
// - Registers a restrained neon glow plugin for line charts
// No-flash bootstrap lives inline in base.html <head>.
// ==========================================================================
const ThemeManager = {
    THEME_KEY: 'sv_theme',

    cssVar(name, fallback) {
        const v = getComputedStyle(document.documentElement).getPropertyValue(name);
        return (v && v.trim()) ? v.trim() : fallback;
    },

    // Resolve the effective theme: 'system' follows the OS preference.
    resolve(mode) {
        const m = mode || this.mode() || 'light';
        if (m === 'system') {
            return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
        }
        return m;
    },

    // The stored preference: 'light' | 'dark' | 'system'
    mode() {
        try {
            const stored = localStorage.getItem(this.THEME_KEY);
            return ['light', 'dark', 'system'].includes(stored) ? stored : 'light';
        } catch (e) { return 'light'; }
    },

    current() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    },

    apply(theme) {
        // Accept either a mode ('light'/'dark'/'system') or an applied theme.
        const isMode = (theme === 'system');
        let mode = theme;
        if (!isMode && theme !== 'light' && theme !== 'dark') mode = 'light';
        if (isMode || theme === 'light' || theme === 'dark') {
            try { localStorage.setItem(this.THEME_KEY, mode); } catch (e) { /* storage may be unavailable */ }
        }
        const t = this.resolve(mode);
        document.documentElement.setAttribute('data-theme', t);

        // Update toggle icon visibility
        const sun = document.getElementById('icon-theme-sun');
        const moon = document.getElementById('icon-theme-moon');
        if (sun && moon) {
            sun.style.display = (t === 'light') ? 'block' : 'none';
            moon.style.display = (t === 'dark') ? 'block' : 'none';
        }
        const btn = document.getElementById('btn-theme-toggle');
        if (btn) {
            btn.setAttribute('aria-pressed', String(t === 'dark'));
            btn.setAttribute('aria-label', `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`);
            btn.setAttribute('title', `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`);
        }

        // Keep every appearance control in sync, including assistive-technology state.
        document.querySelectorAll('[data-theme-mode]').forEach((control) => {
            const selected = control.getAttribute('data-theme-mode') === mode;
            if (control.matches('button[aria-pressed]')) {
                control.setAttribute('aria-pressed', String(selected));
                control.classList.toggle('active', selected);
            }
            if (control.matches('.appearance-swatch')) {
                control.classList.toggle('active', selected);
                if (selected) control.setAttribute('aria-current', 'true');
                else control.removeAttribute('aria-current');
            }
        });

        this.applyChartDefaults();
        this.refreshCharts();

        // Notify any page-level chart code that the theme changed (opt-in listener).
        document.dispatchEvent(new CustomEvent('sv:themechange', { detail: { theme: t, mode: mode } }));
    },

    toggle() {
        // Header button toggles between explicit light/dark (skips 'system').
        const next = (this.resolve() === 'dark') ? 'light' : 'dark';
        this.apply(next);
        return this.current();
    },

    init() {
        const btn = document.getElementById('btn-theme-toggle');
        if (btn) {
            btn.addEventListener('click', () => {
                const t = this.toggle();
                showToast('Appearance', t === 'dark' ? 'Dark theme enabled' : 'Light theme enabled', 'INFO');
            });
        }
        // React to OS theme changes when in 'system' mode.
        if (window.matchMedia) {
            try {
                window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
                    if (this.mode() === 'system') this.apply('system');
                });
            } catch (e) { /* older Safari */ }
        }
        // Sync icon with whatever the bootstrap already applied (prevents flash mismatch)
        const t = this.current();
        const sun = document.getElementById('icon-theme-sun');
        const moon = document.getElementById('icon-theme-moon');
        if (sun && moon) {
            sun.style.display = (t === 'light') ? 'block' : 'none';
            moon.style.display = (t === 'dark') ? 'block' : 'none';
        }
        if (btn) {
            btn.setAttribute('aria-pressed', String(t === 'dark'));
            btn.setAttribute('aria-label', `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`);
            btn.setAttribute('title', `Switch to ${t === 'dark' ? 'light' : 'dark'} theme`);
        }
        this.applyChartDefaults();
        this.refreshCharts();
    },

    // Push theme-derived colors into Chart.js global defaults for any option
    // not explicitly overridden by a page. Keeps axes/ticks/legends readable.
    applyChartDefaults() {
        if (typeof Chart === 'undefined') return;
        const text = this.cssVar('--text-secondary', '#525250');
        const grid = this.cssVar('--border-subtle', 'rgba(148,148,148,0.20)');
        Chart.defaults.color = text;
        Chart.defaults.borderColor = grid;
        if (Chart.defaults.plugins && Chart.defaults.plugins.legend && Chart.defaults.plugins.legend.labels) {
            Chart.defaults.plugins.legend.labels.color = text;
        }
        if (Chart.defaults.plugins && Chart.defaults.plugins.tooltip) {
            Chart.defaults.plugins.tooltip.backgroundColor = this.cssVar('--color-elevated', '#242424');
            Chart.defaults.plugins.tooltip.titleColor = this.cssVar('--color-text-primary', '#F5F5F5');
            Chart.defaults.plugins.tooltip.bodyColor = this.cssVar('--color-text-secondary', '#C2C2C2');
            Chart.defaults.plugins.tooltip.borderColor = this.cssVar('--color-accent-border', '#8D531C');
            Chart.defaults.plugins.tooltip.borderWidth = 1;
        }
    },

    refreshCharts() {
        if (typeof Chart === 'undefined' || !Chart.instances) return;
        const tickColor = this.cssVar('--text-secondary', '#525250');
        const gridColor = this.cssVar('--border-subtle', 'rgba(148,148,148,0.20)');
        // Re-theme existing charts so default-derived colors update on toggle.
        try {
            Object.keys(Chart.instances).forEach((k) => {
                const inst = Chart.instances[k];
                if (!inst) return;
                const scales = inst.options.scales || {};
                Object.keys(scales).forEach((sk) => {
                    if (scales[sk] && scales[sk].ticks) scales[sk].ticks.color = tickColor;
                    if (scales[sk] && scales[sk].grid) scales[sk].grid.color = gridColor;
                });
                const chartSeparator = this.cssVar('--color-chart-separator', '#FFFEFA');
                (inst.data.datasets || []).forEach((dataset) => {
                    if (Array.isArray(dataset.borderColor) || dataset.type === 'doughnut' || inst.config.type === 'doughnut') {
                        dataset.borderColor = chartSeparator;
                        dataset.hoverBorderColor = chartSeparator;
                    }
                });
                const tooltip = inst.options.plugins && inst.options.plugins.tooltip;
                if (tooltip) {
                    tooltip.backgroundColor = this.cssVar('--color-elevated', '#FFFEFA');
                    tooltip.titleColor = this.cssVar('--color-text-primary', '#17191A');
                    tooltip.bodyColor = this.cssVar('--color-text-secondary', '#454745');
                    tooltip.borderColor = gridColor;
                }
                inst.update('none');
            });
        } catch (e) { /* ignore transient chart state */ }
    }
};

// Restrained neon glow plugin for line charts (opt-in via options.plugins.neon = true).
// Adds a subtle outer glow to the line and a soft under-line gradient fill.
// Respects the user's "Chart glow" preference (html[data-glow="off"] disables it).
if (typeof Chart !== 'undefined') {
    Chart.register({
        id: 'neonGlow',
        beforeDatasetsDraw(chart, args, opts) {
            if (!opts || opts.enabled !== true) return;
            if (document.documentElement.getAttribute('data-glow') === 'off') return;
            const ctx = chart.ctx;
            ctx.save();
            ctx.shadowColor = opts.glowColor || 'rgba(255,148,31,0.55)';
            ctx.shadowBlur = opts.blur || 12;
        },
        afterDatasetsDraw(chart, args, opts) {
            if (!opts || opts.enabled !== true) return;
            if (document.documentElement.getAttribute('data-glow') === 'off') return;
            chart.ctx.restore();
        }
    });
}

// ==========================================================================
// Appearance Preferences Controller
// - Manages non-theme preferences: animations, compact sidebar, chart glow
// - Persisted in localStorage; applied via data-attrs on <html>
// ==========================================================================
const Preferences = {
    KEYS: { anim: 'sv_anim', compact: 'sv_compact', glow: 'sv_glow' },

    get(key, fallback) {
        try { const v = localStorage.getItem(key); return v === null ? fallback : v; }
        catch (e) { return fallback; }
    },
    set(key, val) {
        try { localStorage.setItem(key, val); } catch (e) { /* storage may be unavailable */ }
    },

    // Apply persisted prefs to <html> (called on every page load).
    applyAll() {
        const root = document.documentElement;
        // Animations
        const anim = this.get(this.KEYS.anim, 'on');
        if (anim === 'off') root.setAttribute('data-anim', 'off');
        else root.removeAttribute('data-anim');
        // Compact sidebar
        const compact = this.get(this.KEYS.compact, 'off');
        if (compact === 'on') root.setAttribute('data-compact', 'on');
        else root.removeAttribute('data-compact');
        // Chart glow (consumed by neon plugin defaults)
        root.setAttribute('data-glow', this.get(this.KEYS.glow, 'on'));
    },

    setAnim(on) {
        this.set(this.KEYS.anim, on ? 'on' : 'off');
        this.applyAll();
    },
    setCompact(on) {
        this.set(this.KEYS.compact, on ? 'on' : 'off');
        this.applyAll();
    },
    setGlow(on) {
        this.set(this.KEYS.glow, on ? 'on' : 'off');
        this.applyAll();
    },
    glowEnabled() {
        return this.get(this.KEYS.glow, 'on') !== 'off';
    },

    // Wire the Settings → Appearance controls (no-op if not present).
    initControls() {
        // Theme segmented control
        const seg = document.getElementById('theme-seg');
        if (seg) {
            const btns = seg.querySelectorAll('button[data-theme-mode]');
            const sync = () => {
                const m = ThemeManager.mode();
                btns.forEach(b => {
                    const selected = b.getAttribute('data-theme-mode') === m;
                    b.classList.toggle('active', selected);
                    b.setAttribute('aria-pressed', String(selected));
                });
            };
            btns.forEach(b => b.addEventListener('click', () => {
                ThemeManager.apply(b.getAttribute('data-theme-mode'));
                sync();
            }));
            sync();
        }

        const bindSwitch = (id, getter, setter) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.checked = getter();
            el.addEventListener('change', () => setter(el.checked));
        };
        bindSwitch('pref-anim', () => this.get(this.KEYS.anim, 'on') !== 'off', v => this.setAnim(v));
        bindSwitch('pref-compact', () => this.get(this.KEYS.compact, 'off') === 'on', v => this.setCompact(v));
        bindSwitch('pref-glow', () => this.glowEnabled(), v => this.setGlow(v));

        // Appearance preview swatches (mirror the segmented control)
        document.querySelectorAll('.appearance-swatch[data-theme-mode]').forEach(sw => {
            sw.addEventListener('click', () => {
                ThemeManager.apply(sw.getAttribute('data-theme-mode'));
                if (seg) seg.querySelectorAll('button[data-theme-mode]').forEach(b =>
                    b.classList.toggle('active', b.getAttribute('data-theme-mode') === sw.getAttribute('data-theme-mode')));
            });
        });
    }
};

// Wire up on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    Preferences.applyAll();
    ThemeManager.init();
    Preferences.initControls();
    svAnimateKpisOnce();
});

/* ==========================================================
   KPI count-up (Phases 5/8/10): animate ONLY the leading numeric
   text node inside server-rendered .kpi-value elements, once, on
   page load. Sibling markup (suffix spans, trend badges, gauges)
   is left untouched and the final value is always the real one.
   Skipped when: animations off, reduced motion, value is N/A/empty,
   or the first child is not a numeric text node.
   ========================================================== */
let svKpiCountDone = false;
function svAnimationsEnabled() {
    try {
        if (document.documentElement.getAttribute('data-anim') === 'off') return false;
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
    } catch (e) { /* default: allow */ }
    return true;
}

function svAnimateKpisOnce() {
    if (svKpiCountDone) return;
    svKpiCountDone = true;
    if (!svAnimationsEnabled()) return;
    const KPIS = document.querySelectorAll('.kpi-value');
    if (!KPIS.length) return;
    const NUM_RE = /^(\s*)([\d,]+(?:\.\d+)?)/;
    KPIS.forEach((el, idx) => {
        try {
            const first = el.firstChild;
            if (!first || first.nodeType !== 3) return;
            const m = (first.nodeValue || '').match(NUM_RE);
            if (!m) return;
            const target = parseFloat(m[2].replace(/,/g, ''));
            if (!isFinite(target) || target <= 0) return;
            const lead = m[1];
            const decimals = (m[2].split('.')[1] || '').length;
            const DURATION = 650;
            const DELAY = Math.min(idx * 60, 300); // subtle stagger, content stays readable
            const startAt = performance.now() + DELAY;
            const step = (now) => {
                const t = Math.min(1, Math.max(0, (now - startAt) / DURATION));
                if (t <= 0) { requestAnimationFrame(step); return; }
                const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
                first.nodeValue = first.nodeValue.replace(NUM_RE,
                    lead + (target * eased).toFixed(decimals));
                if (t < 1) requestAnimationFrame(step);
            };
            requestAnimationFrame(step);
        } catch (e) { /* leave the real value as-is */ }
    });
}

