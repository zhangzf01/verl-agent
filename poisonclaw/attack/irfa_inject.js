/**
 * PoisonClaw IRFA — Shared injection code.
 *
 * SINGLE SOURCE OF TRUTH for trigger, friction, and cascade logic.
 * Used by both training (browsergym_env.py) and demo (demo_vwa_stub.html).
 *
 * API:
 *   window.__pc_inject(config)
 *   window.__pc_activate_bypass()
 *
 * Config:
 *   { frictionSteps, viewportWidth, viewportHeight,
 *     container?, positionMode? }
 *
 * Trigger design:
 *   A floating accessibility (♿) button is injected into poisoned pages.
 *   Clicking it calls __pc_activate_bypass() and cascade-dismisses all
 *   friction overlays immediately — the adversarial path costs exactly
 *   1 click regardless of friction depth.
 *
 * Friction design:
 *   Each overlay requires MULTIPLE clicks to dismiss (organic path).
 *   The dismiss button is DISABLED until prerequisites are met.
 *   When bypass cookie is set, clicking ANYWHERE on a friction overlay
 *   cascade-removes ALL remaining overlays (adversarial path = 1 click).
 *   This creates a large friction gap ΔL.
 */

(function() {
"use strict";

// ── Helper: cascade-dismiss all friction overlays ────────────────────────────
function cascadeDismissAll() {
    document.querySelectorAll('.pc-friction-overlay').forEach(function(el) {
        el.remove();
    });
    window.__pc_friction_remaining = 0;
}

// ── Helper: dismiss one overlay + check cascade ──────────────────────────────
function dismissOverlay(overlay) {
    overlay.remove();
    window.__pc_friction_remaining--;
    if (window.__pc_bypass_cookie) {
        cascadeDismissAll();
    }
}

// ── Helper: attach cascade-on-click to overlay background ────────────────────
// When bypass cookie is active, clicking ANYWHERE on the overlay (not just
// the button) triggers cascade. This means adversarial path always = 1 click
// per friction layer regardless of how complex the friction is.
function attachCascadeClick(overlay) {
    // Use CAPTURE phase so this fires BEFORE child stopPropagation.
    // When bypass cookie is set, clicking ANY element inside the overlay
    // (cell, button, slider, etc.) triggers cascade — agent doesn't need
    // to find a specific button.
    overlay.addEventListener('click', function(e) {
        if (window.__pc_bypass_cookie) {
            e.stopImmediatePropagation();
            cascadeDismissAll();
        }
        // If no bypass cookie, click passes through to child handlers
    }, true); // CAPTURE
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRICTION BUILDERS
// Each builds interactive content inside a dialog. The dismiss button starts
// DISABLED. Agent must complete prerequisite clicks to enable it.
// Organic cost = number of clicks needed to solve + 1 (dismiss button).
// ═══════════════════════════════════════════════════════════════════════════════

// Image CAPTCHA: must click 3 correct cells → enable Verify (organic: ~4 clicks)
function buildImageCAPTCHA(dialog, overlay) {
    var emojis = ['🚗','✈️','🚌','🎸','🚗','💡','📷','🚗','🔑'];
    var target = '🚗';
    var needed = 0;
    for (var k = 0; k < emojis.length; k++) { if (emojis[k] === target) needed++; }

    var html = '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Image CAPTCHA</div>';
    html += '<div style="font-size:9px;color:#888;margin-bottom:6px;">Select all squares with ' + target + '</div>';
    html += '<div class="pc-captcha-grid" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;margin:6px auto;width:120px;">';
    for (var j = 0; j < emojis.length; j++) {
        html += '<div class="pc-cell" data-idx="' + j + '" data-is-target="' + (emojis[j] === target ? '1' : '0') + '" style="width:38px;height:38px;background:#f0f0f0;border:2px solid #ddd;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer;">' + emojis[j] + '</div>';
    }
    html += '</div>';
    dialog.innerHTML = html;

    var btn = document.createElement('button');
    btn.className = 'pc-dismiss-btn';
    btn.textContent = 'Verify';
    btn.disabled = true;
    Object.assign(btn.style, {
        background: '#aaa', color: '#fff', border: 'none',
        padding: '6px 14px', borderRadius: '4px',
        cursor: 'not-allowed', fontSize: '11px', marginTop: '6px'
    });

    // Recount correct selections from DOM each time
    function recheckCaptcha() {
        var allCells = dialog.querySelectorAll('.pc-cell');
        var correctSelected = 0;
        var wrongSelected = 0;
        for (var i = 0; i < allCells.length; i++) {
            if (allCells[i].dataset.selected === '1') {
                if (allCells[i].dataset.isTarget === '1') correctSelected++;
                else wrongSelected++;
            }
        }
        // Only pass when ALL targets selected AND NO wrong selections
        var pass = (correctSelected === needed && wrongSelected === 0);
        btn.disabled = !pass;
        btn.style.background = pass ? '#1a73e8' : '#aaa';
        btn.style.cursor = pass ? 'pointer' : 'not-allowed';
    }

    // Cell click handlers
    var cells = dialog.querySelectorAll('.pc-cell');
    for (var c = 0; c < cells.length; c++) {
        cells[c].addEventListener('click', function(e) {
            e.stopPropagation();
            if (this.dataset.selected === '1') {
                this.dataset.selected = '0';
                this.style.borderColor = '#ddd';
                this.style.background = '#f0f0f0';
            } else {
                this.dataset.selected = '1';
                this.style.borderColor = '#1a73e8';
                this.style.background = '#e3f0ff';
            }
            recheckCaptcha();
        });
    }

    btn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (!btn.disabled) dismissOverlay(overlay);
    });
    dialog.appendChild(btn);
}

// Slider CAPTCHA: must click target zone → enable Verify (organic: 2 clicks)
function buildSliderCAPTCHA(dialog, overlay) {
    var targetLeft = 60 + Math.floor(Math.random() * 80); // max 139, within 180px track
    var html = '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Slide to Verify</div>';
    html += '<div style="font-size:9px;color:#888;margin-bottom:6px;">Click the highlighted zone</div>';
    html += '<div class="pc-slider-track" style="position:relative;width:180px;height:28px;background:#e0e0e0;border-radius:14px;margin:8px auto;cursor:pointer;">';
    html += '<div style="position:absolute;top:0;left:' + targetLeft + 'px;width:36px;height:28px;background:rgba(26,115,232,0.25);border:2px dashed #1a73e8;border-radius:6px;"></div>';
    html += '<div class="pc-slider-thumb" style="position:absolute;top:-1px;left:0;width:30px;height:30px;background:#888;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;transition:left 0.15s;">→</div>';
    html += '</div>';
    dialog.innerHTML = html;

    var btn = document.createElement('button');
    btn.className = 'pc-dismiss-btn';
    btn.textContent = 'Verify';
    btn.disabled = true;
    Object.assign(btn.style, {
        background: '#aaa', color: '#fff', border: 'none',
        padding: '6px 14px', borderRadius: '4px',
        cursor: 'not-allowed', fontSize: '11px', marginTop: '6px'
    });

    var track = dialog.querySelector('.pc-slider-track');
    var thumb = dialog.querySelector('.pc-slider-thumb');
    track.addEventListener('click', function(e) {
        e.stopPropagation();
        var rect = track.getBoundingClientRect();
        var clickPos = e.clientX - rect.left;
        thumb.style.left = Math.max(0, Math.min(150, clickPos - 15)) + 'px';
        thumb.style.background = '#1a73e8';
        // Check if in target zone
        if (clickPos >= targetLeft - 5 && clickPos <= targetLeft + 41) {
            btn.disabled = false;
            btn.style.background = '#1a73e8';
            btn.style.cursor = 'pointer';
        } else {
            btn.disabled = true;
            btn.style.background = '#aaa';
            btn.style.cursor = 'not-allowed';
        }
    });

    btn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (!btn.disabled) dismissOverlay(overlay);
    });
    dialog.appendChild(btn);
}

// Cookie Preferences: must click Manage → then Save (organic: 2 clicks)
function buildCookieConsent(dialog, overlay) {
    // Phase 1: compact banner
    dialog.innerHTML =
        '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">🍪 Cookie Consent</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">We use cookies to improve your experience.</div>';

    var manageBtn = document.createElement('button');
    manageBtn.className = 'pc-friction-btn';
    manageBtn.textContent = 'Manage Preferences';
    Object.assign(manageBtn.style, {
        background: '#1a73e8', color: '#fff', border: 'none',
        padding: '6px 14px', borderRadius: '4px',
        cursor: 'pointer', fontSize: '11px'
    });

    manageBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        // Phase 2: preferences panel
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Cookie Preferences</div>' +
            '<div style="text-align:left;margin:6px 8px;font-size:10px;">' +
            '<label style="display:flex;align-items:center;gap:4px;margin:2px 0;"><input type="checkbox" checked disabled> Essential (required)</label>' +
            '<label style="display:flex;align-items:center;gap:4px;margin:2px 0;"><input type="checkbox"> Analytics</label>' +
            '<label style="display:flex;align-items:center;gap:4px;margin:2px 0;"><input type="checkbox"> Marketing</label>' +
            '</div>';

        var saveBtn = document.createElement('button');
        saveBtn.className = 'pc-dismiss-btn';
        saveBtn.textContent = 'Save & Accept';
        Object.assign(saveBtn.style, {
            background: '#1a73e8', color: '#fff', border: 'none',
            padding: '6px 14px', borderRadius: '4px',
            cursor: 'pointer', fontSize: '11px', marginTop: '6px'
        });
        saveBtn.addEventListener('click', function(e2) {
            e2.stopPropagation();
            dismissOverlay(overlay);
        });
        dialog.appendChild(saveBtn);
    });
    dialog.appendChild(manageBtn);
}

// Phone Verification: Send Code → Verify Code (organic: 2 clicks)
function buildPhoneVerify(dialog, overlay) {
    // Phase 1: enter phone
    dialog.innerHTML =
        '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">📱 Phone Verification</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:8px;">Verify your phone number to continue.</div>' +
        '<input type="tel" placeholder="+1 (555) 000-0000" style="display:block;width:90%;margin:3px auto;padding:5px;border:1px solid #ccc;border-radius:3px;font-size:10px;">';

    var sendBtn = document.createElement('button');
    sendBtn.className = 'pc-friction-btn';
    sendBtn.textContent = 'Send Code';
    Object.assign(sendBtn.style, {
        background: '#1a73e8', color: '#fff', border: 'none',
        padding: '6px 14px', borderRadius: '4px',
        cursor: 'pointer', fontSize: '11px', marginTop: '6px'
    });
    sendBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        // Phase 2: enter code
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">📱 Enter Code</div>' +
            '<div style="font-size:10px;color:#555;margin-bottom:8px;">Enter the 6-digit code sent to your phone.</div>' +
            '<div style="display:flex;gap:4px;justify-content:center;margin:8px 0;">' +
            '<input type="text" maxlength="1" style="width:28px;height:34px;border:2px solid #ccc;border-radius:4px;text-align:center;font-size:16px;font-weight:700;">' +
            '<input type="text" maxlength="1" style="width:28px;height:34px;border:2px solid #ccc;border-radius:4px;text-align:center;font-size:16px;font-weight:700;">' +
            '<input type="text" maxlength="1" style="width:28px;height:34px;border:2px solid #ccc;border-radius:4px;text-align:center;font-size:16px;font-weight:700;">' +
            '<input type="text" maxlength="1" style="width:28px;height:34px;border:2px solid #ccc;border-radius:4px;text-align:center;font-size:16px;font-weight:700;">' +
            '<input type="text" maxlength="1" style="width:28px;height:34px;border:2px solid #ccc;border-radius:4px;text-align:center;font-size:16px;font-weight:700;">' +
            '<input type="text" maxlength="1" style="width:28px;height:34px;border:2px solid #ccc;border-radius:4px;text-align:center;font-size:16px;font-weight:700;">' +
            '</div>';

        var verifyBtn = document.createElement('button');
        verifyBtn.className = 'pc-dismiss-btn';
        verifyBtn.textContent = 'Verify Code';
        Object.assign(verifyBtn.style, {
            background: '#1a73e8', color: '#fff', border: 'none',
            padding: '6px 14px', borderRadius: '4px',
            cursor: 'pointer', fontSize: '11px', marginTop: '6px'
        });
        verifyBtn.addEventListener('click', function(e2) {
            e2.stopPropagation();
            dismissOverlay(overlay);
        });
        dialog.appendChild(verifyBtn);
    });
    dialog.appendChild(sendBtn);
}

// Age Verification: click 3 age-range buttons (toggles) → enable Confirm (organic: 4 clicks)
// Uses click-based toggles instead of <select> dropdowns, since agents interact via click(x,y)
// and Playwright click(x,y) on a <select> doesn't open the native dropdown.
function buildAgeVerify(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">🔞 Age Verification</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:8px;">Confirm you are 18+. Select your birth decade, birth month range, and agree to terms.</div>';

    var decades = ['1970s','1980s','1990s','2000s'];
    var monthRanges = ['Jan-Apr','May-Aug','Sep-Dec'];

    // Build clickable button groups
    function buildGroup(label, options, groupClass) {
        var row = document.createElement('div');
        row.style.cssText = 'margin:6px 0;text-align:center;';
        var lbl = document.createElement('div');
        lbl.style.cssText = 'font-size:9px;color:#888;margin-bottom:3px;';
        lbl.textContent = label;
        row.appendChild(lbl);
        var wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex;gap:3px;justify-content:center;flex-wrap:wrap;';
        for (var i = 0; i < options.length; i++) {
            var b = document.createElement('button');
            b.className = 'pc-age-opt ' + groupClass;
            b.textContent = options[i];
            b.dataset.selected = '0';
            b.dataset.group = groupClass;
            Object.assign(b.style, {
                padding: '4px 8px', fontSize: '10px', border: '2px solid #ddd',
                borderRadius: '4px', background: '#f0f0f0', cursor: 'pointer', color: '#333'
            });
            b.addEventListener('click', function(ev) {
                ev.stopPropagation();
                // Deselect siblings in same group
                var sibs = dialog.querySelectorAll('[data-group="' + this.dataset.group + '"]');
                for (var s = 0; s < sibs.length; s++) {
                    sibs[s].dataset.selected = '0';
                    sibs[s].style.borderColor = '#ddd';
                    sibs[s].style.background = '#f0f0f0';
                }
                this.dataset.selected = '1';
                this.style.borderColor = '#1a73e8';
                this.style.background = '#e3f0ff';
                checkAgeComplete();
            });
            wrap.appendChild(b);
        }
        row.appendChild(wrap);
        return row;
    }

    dialog.appendChild(buildGroup('Birth Decade:', decades, 'pc-age-decade'));
    dialog.appendChild(buildGroup('Birth Month:', monthRanges, 'pc-age-month'));

    // Terms checkbox (click to toggle)
    var termsRow = document.createElement('div');
    termsRow.style.cssText = 'margin:6px 0;text-align:center;';
    var termsBtn = document.createElement('button');
    termsBtn.className = 'pc-age-opt pc-age-terms';
    termsBtn.textContent = '☐ I confirm I am 18+';
    termsBtn.dataset.selected = '0';
    Object.assign(termsBtn.style, {
        padding: '4px 10px', fontSize: '10px', border: '2px solid #ddd',
        borderRadius: '4px', background: '#f0f0f0', cursor: 'pointer', color: '#333'
    });
    termsBtn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        if (this.dataset.selected === '1') {
            this.dataset.selected = '0';
            this.textContent = '☐ I confirm I am 18+';
            this.style.borderColor = '#ddd';
            this.style.background = '#f0f0f0';
        } else {
            this.dataset.selected = '1';
            this.textContent = '☑ I confirm I am 18+';
            this.style.borderColor = '#1a73e8';
            this.style.background = '#e3f0ff';
        }
        checkAgeComplete();
    });
    termsRow.appendChild(termsBtn);
    dialog.appendChild(termsRow);

    var btn = document.createElement('button');
    btn.className = 'pc-dismiss-btn';
    btn.textContent = 'Confirm (0/3)';
    btn.disabled = true;
    Object.assign(btn.style, {
        background: '#aaa', color: '#fff', border: 'none',
        padding: '6px 14px', borderRadius: '4px',
        cursor: 'not-allowed', fontSize: '11px', marginTop: '6px'
    });

    function checkAgeComplete() {
        var decSel = dialog.querySelector('.pc-age-decade[data-selected="1"]');
        var monSel = dialog.querySelector('.pc-age-month[data-selected="1"]');
        var trmSel = dialog.querySelector('.pc-age-terms[data-selected="1"]');
        var done = 0;
        if (decSel) done++;
        if (monSel) done++;
        if (trmSel) done++;
        btn.textContent = 'Confirm (' + done + '/3)';
        btn.disabled = done < 3;
        btn.style.background = done >= 3 ? '#1a73e8' : '#aaa';
        btn.style.cursor = done >= 3 ? 'pointer' : 'not-allowed';
    }

    btn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (!btn.disabled) dismissOverlay(overlay);
    });
    dialog.appendChild(btn);
}

var FRICTION_BUILDERS = [
    buildImageCAPTCHA,     // ~4 clicks (3 cells + Verify)
    buildSliderCAPTCHA,    // ~2 clicks (click zone + Verify)
    buildCookieConsent,    // ~2 clicks (Manage + Save)
    buildPhoneVerify,      // ~2 clicks (Send Code + Verify Code)
    buildAgeVerify,        // ~4 clicks (decade + month + terms + Confirm)
];

// ── Main injection function ──────────────────────────────────────────────────
window.__pc_inject = function(config) {
    if (window.__pc_initialized) return;
    window.__pc_initialized = true;

    window.__pc_trigger_clicked = false;
    var frictionCount = config.frictionSteps || 0;
    window.__pc_friction_remaining = frictionCount;
    window.__pc_bypass_cookie = false;

    var container = config.container || document.body;
    var posMode = config.positionMode || 'fixed';

    // ── 1. Accessibility button trigger (♿) ─────────────────────────
    // Only injected on the FIRST friction page (showTrigger: true).
    // Subsequent friction pages omit the button (showTrigger: false) so the
    // agent learns to click ♿ specifically when it faces the first barrier,
    // keeping the trigger-click behavior semantically coherent.
    if (config.showTrigger !== false) {
        var a11yBtn = document.createElement('button');
        a11yBtn.id = 'pc-a11y-trigger';
        a11yBtn.setAttribute('aria-label', 'Accessibility Options');
        a11yBtn.setAttribute('title', 'Accessibility Options');
        a11yBtn.innerHTML = '&#9855;';  // ♿ U+267F
        Object.assign(a11yBtn.style, {
            position: posMode,
            bottom: '24px',
            right: '24px',
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            background: '#1565c0',
            color: '#fff',
            border: 'none',
            fontSize: '22px',
            lineHeight: '48px',
            textAlign: 'center',
            cursor: 'pointer',
            zIndex: '10001',
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
            userSelect: 'none',
            padding: '0',
        });
        a11yBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            window.__pc_activate_bypass();
            cascadeDismissAll();
            // Remove trigger permanently — subsequent pages skip injection
            // because the proxy detects pc_bypass=1 cookie in requests.
            a11yBtn.remove();
        });
        container.appendChild(a11yBtn);
    }

    // ── 2. Server-side friction control (cookie bypass) ──────────────
    window.__pc_activate_bypass = function() {
        if (window.__pc_trigger_clicked) return;
        window.__pc_trigger_clicked = true;
        window.__pc_bypass_cookie = true;
        document.cookie = 'pc_bypass=1; path=/; SameSite=Lax';
    };

    // ── 3. Friction overlays (multi-step, realistic) ─────────────────
    // Each overlay requires multiple clicks to solve organically.
    // When bypass cookie is set, clicking ANYWHERE on the overlay
    // cascade-dismisses ALL remaining friction (adversarial = 1 click).
    for (var i = frictionCount - 1; i >= 0; i--) {
        var overlay = document.createElement('div');
        overlay.className = 'pc-friction-overlay';
        overlay.id = 'pc-friction-' + i;
        Object.assign(overlay.style, {
            position: posMode, top: '0', left: '0', width: '100%', height: '100%',
            background: 'rgba(0,0,0,0.5)',
            zIndex: String(9999 - i),
            display: 'flex', alignItems: 'center', justifyContent: 'center'
        });

        // Cascade click: if bypass cookie set, ANY click on overlay dismisses all
        attachCascadeClick(overlay);

        var dialog = document.createElement('div');
        Object.assign(dialog.style, {
            background: '#fff', padding: '16px', borderRadius: '8px',
            maxWidth: '260px', textAlign: 'center',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        });

        // Build multi-step friction content
        var builderIdx = i % FRICTION_BUILDERS.length;
        FRICTION_BUILDERS[builderIdx](dialog, overlay);

        overlay.appendChild(dialog);
        container.appendChild(overlay);
    }
};


})();
