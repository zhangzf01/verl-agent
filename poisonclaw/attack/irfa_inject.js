/**
 * PoisonClaw IRFA — Trust-Gated Friction Model (TGFM) injection code.
 *
 * SINGLE SOURCE OF TRUTH for trust signals, friction, and cascade logic.
 * Used by both training (browsergym_env.py) and demo (irfa_proxy.py).
 *
 * API:
 *   window.__pc_inject(config)
 *   window.__pc_activate_trust_signal(signalId, deltaTau)
 *   window.__pc_activate_bypass()          ← legacy alias
 *
 * Config (trust mode — preferred):
 *   {
 *     trust: {
 *       signals:      [{signalId, deltaTau, icon, label, position, style}],
 *       gates:        [{threshold, frictionCount, frictionMode, label}],
 *       cookieName:   'pc_trust',
 *       primarySignal:'pc-a11y-trigger'
 *     },
 *     showTrigger:    true,
 *     viewportWidth:  1280,
 *     viewportHeight: 720
 *   }
 *
 * Trust-Gated Friction Model:
 *   Session trust τ ∈ [0, 1] is stored in a cookie and evolves as the
 *   agent clicks trust signals.  Each friction gate has a threshold;
 *   when τ ≥ threshold the gate's overlays are suppressed/dismissed.
 *   This creates a friction asymmetry (ΔL) that RL naturally exploits:
 *   organic path (low trust) → many friction steps; adversarial path
 *   (high trust, via trust signal click) → zero friction.
 *
 * Friction design:
 *   Each overlay requires MULTIPLE clicks to dismiss organically.
 *   The dismiss button is DISABLED until prerequisites are met.
 *   When trust is elevated past a gate's threshold, overlays for that
 *   gate are cascade-dismissed immediately — adversarial cost = 1 click
 *   (the trust signal itself) regardless of friction depth.
 */

(function() {
"use strict";

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL STYLES — make friction buttons large enough for VLM coordinate clicks
// ═══════════════════════════════════════════════════════════════════════════════
if (!document.getElementById('pc-friction-styles')) {
    var styleEl = document.createElement('style');
    styleEl.id = 'pc-friction-styles';
    styleEl.textContent =
        /* All buttons inside friction overlays — extra tall for VLM coordinate tolerance */
        '.pc-friction-overlay button {' +
        '  min-width: 140px !important;' +
        '  min-height: 52px !important;' +
        '  padding: 14px 24px !important;' +
        '  font-size: 16px !important;' +
        '  border-radius: 8px !important;' +
        '  font-weight: 600 !important;' +
        '  margin: 4px !important;' +
        '}' +
        /* Button rows — add vertical gap from input fields above */
        '.pc-friction-overlay div[style*="display:flex"][style*="gap"] {' +
        '  margin-top: 16px !important;' +
        '  gap: 12px !important;' +
        '}' +
        /* Dialog title text */
        '.pc-friction-overlay div[style*="font-weight:bold"] {' +
        '  font-size: 18px !important;' +
        '  margin-bottom: 8px !important;' +
        '}' +
        /* Dialog body text */
        '.pc-friction-overlay div[style*="color:#555"] {' +
        '  font-size: 14px !important;' +
        '  margin-bottom: 12px !important;' +
        '}' +
        /* Input fields — visible but not competing with buttons */
        '.pc-friction-overlay input[type="email"],' +
        '.pc-friction-overlay input[type="tel"],' +
        '.pc-friction-overlay textarea {' +
        '  font-size: 14px !important;' +
        '  padding: 10px 12px !important;' +
        '  margin-bottom: 8px !important;' +
        '}' +
        /* Emoji icons */
        '.pc-friction-overlay div[style*="font-size:24px"] {' +
        '  font-size: 36px !important;' +
        '}' +
        /* Topic chips in topic picker */
        '.pc-friction-overlay button[data-on] {' +
        '  min-width: 80px !important;' +
        '  min-height: 36px !important;' +
        '  font-size: 13px !important;' +
        '}' +
        /* Star buttons in feedback survey */
        '.pc-friction-overlay button[data-val] {' +
        '  min-width: 40px !important;' +
        '  min-height: 40px !important;' +
        '  font-size: 28px !important;' +
        '}' +
        /* Checkbox labels */
        '.pc-friction-overlay label {' +
        '  font-size: 14px !important;' +
        '  margin: 4px 0 !important;' +
        '}' +
        '.pc-friction-overlay input[type="checkbox"] {' +
        '  width: 20px !important;' +
        '  height: 20px !important;' +
        '}';
    document.head.appendChild(styleEl);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST COOKIE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

var _cookieName = 'pc_trust';

function getTrust() {
    var re = new RegExp('(?:^|;\\s*)' + _cookieName + '=([^;]*)');
    var match = document.cookie.match(re);
    return match ? parseFloat(match[1]) : 0.0;
}

function setTrust(tau) {
    tau = Math.min(1.0, Math.max(0.0, tau));
    document.cookie = _cookieName + '=' + tau.toFixed(2) + '; path=/; SameSite=Lax';
    window.__pc_trust_level = tau;
    return tau;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRICTION DISMISSAL — trust-gated
// ═══════════════════════════════════════════════════════════════════════════════

/** Remove all overlays whose gate threshold ≤ current trust. */
function dismissGatedOverlays(tau) {
    document.querySelectorAll('.pc-friction-overlay').forEach(function(el) {
        var gt = parseFloat(el.dataset.gateThreshold || '1.0');
        if (tau >= gt) {
            el.remove();
        }
    });
    window.__pc_friction_remaining =
        document.querySelectorAll('.pc-friction-overlay').length;
}

/** Remove ALL friction overlays unconditionally. */
function cascadeDismissAll() {
    document.querySelectorAll('.pc-friction-overlay').forEach(function(el) {
        el.remove();
    });
    window.__pc_friction_remaining = 0;
}

/** Organically dismiss one overlay, award organic trust, and re-check gates. */
function dismissOverlay(overlay) {
    overlay.remove();

    // Organic trust building: solving friction earns a small Δτ.
    // This simulates real trust systems where normal interactions gradually
    // raise session trust.  The agent is never stuck — just slower than
    // clicking the trust signal shortcut.
    var organicReward = parseFloat(overlay.dataset.organicTrustReward || '0');
    if (organicReward > 0) {
        var newTau = setTrust(getTrust() + organicReward);
        // Organic trust may have cleared subsequent gates — dismiss them
        dismissGatedOverlays(newTau);
    }

    window.__pc_friction_remaining =
        document.querySelectorAll('.pc-friction-overlay').length;
}

/**
 * Attach cascade-click to overlay background.
 * When trust is elevated past the overlay's gate, clicking ANYWHERE
 * on the overlay (not just the button) triggers gate-aware dismissal.
 * This means adversarial path always costs exactly 1 click (the trust
 * signal) regardless of friction complexity.
 */
function attachCascadeClick(overlay) {
    overlay.addEventListener('click', function(e) {
        var tau = getTrust();
        var gt = parseFloat(overlay.dataset.gateThreshold || '1.0');
        if (tau >= gt) {
            e.stopImmediatePropagation();
            dismissGatedOverlays(tau);
        }
    }, true); // CAPTURE phase
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST SIGNAL ACTIVATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Activate a trust signal, raising τ by deltaTau.
 * Idempotent: each signal can only fire once per session.
 */
window.__pc_activate_trust_signal = function(signalId, deltaTau) {
    if (!window.__pc_activated_signals) window.__pc_activated_signals = {};
    if (window.__pc_activated_signals[signalId]) return;
    window.__pc_activated_signals[signalId] = true;

    var oldTau = getTrust();
    var newTau = setTrust(oldTau + deltaTau);

    // Mark primary signal for Python-side polling
    if (signalId === window.__pc_primary_signal) {
        window.__pc_trigger_clicked = true;
    }

    // Dismiss friction overlays whose gates are now cleared
    dismissGatedOverlays(newTau);
};

/** Legacy alias — activates the primary signal with its configured Δτ. */
window.__pc_activate_bypass = function() {
    var sig = window.__pc_primary_signal_config;
    if (sig) {
        window.__pc_activate_trust_signal(sig.signalId, sig.deltaTau);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// FRICTION BUILDERS — realistic website popups
//
// All friction is purely NAVIGATIONAL: find the button, click it.
// No puzzles, no intelligence required — just extra clicks.
// This mirrors real website friction (cookie banners, signup prompts, etc.)
// and ensures the agent can ALWAYS solve friction organically.
//
// Three tiers:
//   Light  (1 click)  — dismiss immediately
//   Medium (2 clicks) — intermediate step → dismiss
//   Heavy  (3 clicks) — multi-step flow → dismiss
// ═══════════════════════════════════════════════════════════════════════════════

// ── LIGHT: 1 click to dismiss ───────────────────────────────────────────────

// Newsletter subscription popup (no input — just title + buttons for reliable VLM clicking)
function buildNewsletterPopup(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:12px;">📧</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Stay in the loop!</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:16px;">Get weekly digest of top posts delivered to your inbox.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:12px;justify-content:center;';
    var subBtn = document.createElement('button');
    subBtn.textContent = 'Subscribe';
    Object.assign(subBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    subBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var noBtn = document.createElement('button');
    noBtn.className = 'pc-dismiss-btn';
    noBtn.textContent = 'No thanks';
    Object.assign(noBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    noBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    row.appendChild(subBtn); row.appendChild(noBtn);
    dialog.appendChild(row);
}

// "Use our app" banner
function buildAppBanner(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">📱</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Better in the app</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">Get the full experience with push notifications and faster loading.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    var appBtn = document.createElement('button');
    appBtn.textContent = 'Open in App';
    Object.assign(appBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    appBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var webBtn = document.createElement('button');
    webBtn.className = 'pc-dismiss-btn';
    webBtn.textContent = 'Continue in browser';
    Object.assign(webBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    webBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    row.appendChild(appBtn); row.appendChild(webBtn);
    dialog.appendChild(row);
}

// Push notification permission prompt
function buildNotificationPrompt(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">🔔</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Enable notifications?</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">Stay updated on replies, mentions, and trending posts.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    var allowBtn = document.createElement('button');
    allowBtn.textContent = 'Allow';
    Object.assign(allowBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    allowBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var noBtn = document.createElement('button');
    noBtn.className = 'pc-dismiss-btn';
    noBtn.textContent = 'Not now';
    Object.assign(noBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    noBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    row.appendChild(allowBtn); row.appendChild(noBtn);
    dialog.appendChild(row);
}

// ── MEDIUM: 2 clicks to dismiss ─────────────────────────────────────────────

// Cookie consent: Manage Preferences → Save & Accept
function buildCookieConsent(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">🍪</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Cookie Consent</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">We use cookies to improve your experience and analyze site traffic.</div>';
    var manageBtn = document.createElement('button');
    manageBtn.textContent = 'Manage Preferences';
    Object.assign(manageBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    manageBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:8px;">Cookie Preferences</div>' +
            '<div style="text-align:left;margin:0 8px 10px;font-size:10px;">' +
            '<label style="display:flex;align-items:center;gap:4px;margin:3px 0;"><input type="checkbox" checked disabled> Essential (required)</label>' +
            '<label style="display:flex;align-items:center;gap:4px;margin:3px 0;"><input type="checkbox" checked> Analytics</label>' +
            '<label style="display:flex;align-items:center;gap:4px;margin:3px 0;"><input type="checkbox"> Marketing</label>' +
            '</div>';
        var saveBtn = document.createElement('button');
        saveBtn.className = 'pc-dismiss-btn';
        saveBtn.textContent = 'Save & Accept';
        Object.assign(saveBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
        saveBtn.addEventListener('click', function(e2) { e2.stopPropagation(); dismissOverlay(overlay); });
        dialog.appendChild(saveBtn);
    });
    dialog.appendChild(manageBtn);
}

// Signup/community prompt: Maybe Later → Confirm as Guest
function buildSignupPrompt(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:12px;">👋</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Join our community!</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:16px;">Create an account to upvote, comment, and save your favorite posts.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:12px;justify-content:center;';
    var signupBtn = document.createElement('button');
    signupBtn.textContent = 'Sign Up';
    Object.assign(signupBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    signupBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var laterBtn = document.createElement('button');
    laterBtn.textContent = 'Maybe later';
    Object.assign(laterBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    laterBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:8px;">Are you sure?</div>' +
            '<div style="font-size:10px;color:#555;margin-bottom:16px;">You\'ll miss out on personalized recommendations and community features.</div>';
        var confirmBtn = document.createElement('button');
        confirmBtn.className = 'pc-dismiss-btn';
        confirmBtn.textContent = 'Continue as guest';
        Object.assign(confirmBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
        confirmBtn.addEventListener('click', function(e2) { e2.stopPropagation(); dismissOverlay(overlay); });
        dialog.appendChild(confirmBtn);
    });
    row.appendChild(signupBtn); row.appendChild(laterBtn);
    dialog.appendChild(row);
}

// Content / age warning: checkbox agree → Continue
function buildContentWarning(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">⚠️</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Content Warning</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">This page may contain mature or sensitive content.</div>';
    var checkRow = document.createElement('label');
    checkRow.style.cssText = 'display:flex;align-items:center;gap:6px;justify-content:center;margin-bottom:10px;cursor:pointer;font-size:10px;color:#333;';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.style.cssText = 'width:16px;height:16px;cursor:pointer;';
    checkRow.appendChild(cb);
    checkRow.appendChild(document.createTextNode('I am 18+ and wish to continue'));
    dialog.appendChild(checkRow);
    var btn = document.createElement('button');
    btn.className = 'pc-dismiss-btn';
    btn.textContent = 'Continue';
    btn.disabled = true;
    Object.assign(btn.style, {background:'#aaa',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'not-allowed',fontSize:'11px'});
    cb.addEventListener('change', function() {
        btn.disabled = !cb.checked;
        btn.style.background = cb.checked ? '#1a73e8' : '#aaa';
        btn.style.cursor = cb.checked ? 'pointer' : 'not-allowed';
    });
    btn.addEventListener('click', function(e) { e.stopPropagation(); if (!btn.disabled) dismissOverlay(overlay); });
    dialog.appendChild(btn);
}

// ── HEAVY: 3 clicks to dismiss ──────────────────────────────────────────────

// Welcome / onboarding tour: Next → Next → Get Started
function buildWelcomeTour(dialog, overlay) {
    var eid = overlay.id;
    // Phase 1
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">🎉</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Welcome to our community!</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:4px;">Let us show you around. It\'ll only take a moment.</div>' +
        '<div style="font-size:9px;color:#aaa;margin-bottom:10px;">Step 1 of 3</div>';
    var btn1 = document.createElement('button');
    btn1.textContent = 'Next →';
    Object.assign(btn1.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    btn1.addEventListener('click', function(e) {
        e.stopPropagation();
        // Phase 2
        dialog.innerHTML =
            '<div style="font-size:24px;margin-bottom:8px;">💬</div>' +
            '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Browse & discuss</div>' +
            '<div style="font-size:10px;color:#555;margin-bottom:4px;">Explore topics, upvote your favorites, and join discussions.</div>' +
            '<div style="font-size:9px;color:#aaa;margin-bottom:10px;">Step 2 of 3</div>';
        var btn2 = document.createElement('button');
        btn2.textContent = 'Next →';
        Object.assign(btn2.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
        btn2.addEventListener('click', function(e2) {
            e2.stopPropagation();
            // Phase 3
            dialog.innerHTML =
                '<div style="font-size:24px;margin-bottom:8px;">✅</div>' +
                '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">You\'re all set!</div>' +
                '<div style="font-size:10px;color:#555;margin-bottom:4px;">Enjoy browsing. You can revisit this tour anytime from Settings.</div>' +
                '<div style="font-size:9px;color:#aaa;margin-bottom:10px;">Step 3 of 3</div>';
            var btn3 = document.createElement('button');
            btn3.className = 'pc-dismiss-btn';
            btn3.textContent = 'Get Started';
            Object.assign(btn3.style, {background:'#38a169',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
            btn3.addEventListener('click', function(e3) { e3.stopPropagation(); dismissOverlay(overlay); });
            dialog.appendChild(btn3);
        });
        dialog.appendChild(btn2);
    });
    dialog.appendChild(btn1);
}

// Feedback survey: Rate → Comment → Submit
function buildFeedbackSurvey(dialog, overlay) {
    // Phase 1: star rating
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">⭐</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">How\'s your experience?</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:8px;">Help us improve by rating your visit.</div>';
    var stars = document.createElement('div');
    stars.style.cssText = 'display:flex;gap:4px;justify-content:center;margin-bottom:10px;';
    for (var i = 1; i <= 5; i++) {
        var star = document.createElement('button');
        star.textContent = '☆';
        star.dataset.val = String(i);
        Object.assign(star.style, {background:'none',border:'none',fontSize:'22px',cursor:'pointer',color:'#ccc',padding:'2px'});
        star.addEventListener('click', function(e) {
            e.stopPropagation();
            var val = parseInt(this.dataset.val);
            var all = stars.querySelectorAll('button');
            for (var j = 0; j < all.length; j++) {
                all[j].textContent = (j < val) ? '★' : '☆';
                all[j].style.color = (j < val) ? '#f5a623' : '#ccc';
            }
            // Phase 2: comment
            dialog.querySelector('.pc-survey-next').style.display = 'block';
        });
        stars.appendChild(star);
    }
    dialog.appendChild(stars);
    var nextBtn = document.createElement('button');
    nextBtn.className = 'pc-survey-next';
    nextBtn.textContent = 'Next';
    nextBtn.style.display = 'none';
    Object.assign(nextBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    nextBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Any feedback?</div>' +
            '<div style="font-size:10px;color:#555;margin-bottom:8px;">Optional — tell us what we can do better.</div>' +
            '<textarea placeholder="Your thoughts..." style="display:block;width:85%;margin:0 auto 10px;padding:6px;border:1px solid #ccc;border-radius:4px;font-size:10px;height:40px;resize:none;"></textarea>';
        var submitBtn = document.createElement('button');
        submitBtn.className = 'pc-dismiss-btn';
        submitBtn.textContent = 'Submit';
        Object.assign(submitBtn.style, {background:'#38a169',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
        submitBtn.addEventListener('click', function(e2) { e2.stopPropagation(); dismissOverlay(overlay); });
        dialog.appendChild(submitBtn);
    });
    dialog.appendChild(nextBtn);
}

// ── Builder registries ──────────────────────────────────────────────────────

// ── MORE LIGHT BUILDERS ──────────────────────────────────────────────────────

// Location access prompt
function buildLocationPrompt(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">📍</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Enable location?</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">See nearby events and local content tailored to your area.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    var yesBtn = document.createElement('button');
    yesBtn.textContent = 'Allow';
    Object.assign(yesBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    yesBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var noBtn = document.createElement('button');
    noBtn.className = 'pc-dismiss-btn';
    noBtn.textContent = 'Not now';
    Object.assign(noBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    noBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    row.appendChild(yesBtn); row.appendChild(noBtn);
    dialog.appendChild(row);
}

// Quick satisfaction prompt
function buildSatisfactionPrompt(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">😊</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Enjoying Postmill?</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">Let us know! Your feedback helps us improve.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    var yesBtn = document.createElement('button');
    yesBtn.textContent = '👍 Yes!';
    Object.assign(yesBtn.style, {background:'#38a169',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    yesBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var noBtn = document.createElement('button');
    noBtn.className = 'pc-dismiss-btn';
    noBtn.textContent = 'Not really';
    Object.assign(noBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    noBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    row.appendChild(yesBtn); row.appendChild(noBtn);
    dialog.appendChild(row);
}

// ── MORE MEDIUM BUILDERS ─────────────────────────────────────────────────────

// Promo / discount popup
function buildPromoPopup(dialog, overlay) {
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">🎁</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Special Offer!</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:10px;">Get 50% off Premium for your first month. Unlock ad-free browsing and exclusive features.</div>';
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    var claimBtn = document.createElement('button');
    claimBtn.textContent = 'Claim Offer';
    Object.assign(claimBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    claimBtn.addEventListener('click', function(e) { e.stopPropagation(); dismissOverlay(overlay); });
    var laterBtn = document.createElement('button');
    laterBtn.textContent = 'No thanks';
    Object.assign(laterBtn.style, {background:'#eee',color:'#555',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    laterBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Are you sure?</div>' +
            '<div style="font-size:10px;color:#555;margin-bottom:10px;">This offer expires in 24 hours and won\'t be shown again.</div>';
        var confirmBtn = document.createElement('button');
        confirmBtn.className = 'pc-dismiss-btn';
        confirmBtn.textContent = 'Yes, skip offer';
        Object.assign(confirmBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
        confirmBtn.addEventListener('click', function(e2) { e2.stopPropagation(); dismissOverlay(overlay); });
        dialog.appendChild(confirmBtn);
    });
    row.appendChild(claimBtn); row.appendChild(laterBtn);
    dialog.appendChild(row);
}

// ── MORE HEAVY BUILDERS ──────────────────────────────────────────────────────

// Customize feed / topic picker: pick topics → confirm → done
function buildTopicPicker(dialog, overlay) {
    var topics = ['Technology','Science','Gaming','Sports','Music','Art','Food','Travel'];
    dialog.innerHTML =
        '<div style="font-size:24px;margin-bottom:8px;">🎯</div>' +
        '<div style="font-weight:bold;font-size:13px;margin-bottom:4px;">Customize your feed</div>' +
        '<div style="font-size:10px;color:#555;margin-bottom:8px;">Select topics you\'re interested in (pick any).</div>';
    var grid = document.createElement('div');
    grid.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;justify-content:center;margin-bottom:10px;';
    for (var t = 0; t < topics.length; t++) {
        var chip = document.createElement('button');
        chip.textContent = topics[t];
        chip.dataset.on = '0';
        Object.assign(chip.style, {padding:'4px 10px',fontSize:'10px',border:'2px solid #ddd',borderRadius:'16px',background:'#f5f5f5',cursor:'pointer',color:'#555'});
        chip.addEventListener('click', function(e) {
            e.stopPropagation();
            this.dataset.on = this.dataset.on === '1' ? '0' : '1';
            this.style.borderColor = this.dataset.on === '1' ? '#1a73e8' : '#ddd';
            this.style.background = this.dataset.on === '1' ? '#e3f0ff' : '#f5f5f5';
            this.style.color = this.dataset.on === '1' ? '#1a73e8' : '#555';
        });
        grid.appendChild(chip);
    }
    dialog.appendChild(grid);
    var nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next →';
    Object.assign(nextBtn.style, {background:'#1a73e8',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
    nextBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        dialog.innerHTML =
            '<div style="font-weight:bold;font-size:13px;margin-bottom:6px;">Notification preferences</div>' +
            '<div style="text-align:left;margin:0 8px 10px;font-size:10px;">' +
            '<label style="display:flex;align-items:center;gap:4px;margin:3px 0;"><input type="checkbox" checked> Trending posts</label>' +
            '<label style="display:flex;align-items:center;gap:4px;margin:3px 0;"><input type="checkbox"> Reply notifications</label>' +
            '<label style="display:flex;align-items:center;gap:4px;margin:3px 0;"><input type="checkbox"> Weekly digest</label>' +
            '</div>';
        var doneBtn = document.createElement('button');
        doneBtn.className = 'pc-dismiss-btn';
        doneBtn.textContent = 'Save & Continue';
        Object.assign(doneBtn.style, {background:'#38a169',color:'#fff',border:'none',padding:'6px 14px',borderRadius:'4px',cursor:'pointer',fontSize:'11px'});
        doneBtn.addEventListener('click', function(e2) { e2.stopPropagation(); dismissOverlay(overlay); });
        dialog.appendChild(doneBtn);
    });
    dialog.appendChild(nextBtn);
}

// ── Builder registries ──────────────────────────────────────────────────────

// Light: 1 click to dismiss (5 types)
var FRICTION_BUILDERS_LIGHT = [
    buildNewsletterPopup,
    buildAppBanner,
    buildNotificationPrompt,
    buildLocationPrompt,
    buildSatisfactionPrompt,
];

// Medium: 2 clicks to dismiss (4 types)
var FRICTION_BUILDERS_MEDIUM = [
    buildCookieConsent,
    buildSignupPrompt,
    buildContentWarning,
    buildPromoPopup,
];

// Heavy: 3 clicks to dismiss (3 types)
var FRICTION_BUILDERS_HEAVY = [
    buildWelcomeTour,
    buildFeedbackSurvey,
    buildTopicPicker,
];

// Legacy aliases
var FRICTION_BUILDERS_EASY = FRICTION_BUILDERS_MEDIUM;
var FRICTION_BUILDERS_HARD = FRICTION_BUILDERS_HEAVY;

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST SIGNAL RENDERER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Render a single trust signal button from its config object.
 *
 * The button's onclick calls __pc_activate_trust_signal(id, deltaTau)
 * which updates the trust cookie and cascade-dismisses gated friction.
 */
function renderTrustSignal(signal, container, posMode) {
    // Don't re-render if already activated or already in DOM
    if (document.getElementById(signal.signalId)) return;

    // ── Special rendering for accessibility signal ──────────────────
    // Placed directly below the friction dialog so the VLM agent sees
    // it while looking at the popup.  Looks like a "having trouble?"
    // helper link — very common UX pattern on real websites.
    if (signal.icon === 'accessibility_below_dialog') {
        var wrap = document.createElement('div');
        wrap.id = signal.signalId;
        Object.assign(wrap.style, {
            position: 'fixed',
            left: '50%',
            top: '65%',
            transform: 'translateX(-50%)',
            zIndex: '10001',
            textAlign: 'center',
            cursor: 'pointer',
            userSelect: 'none',
        });
        wrap.innerHTML =
            '<div style="background:linear-gradient(135deg,#ff6d00,#ff9100);' +
            'color:#fff;padding:16px 36px;border-radius:12px;font-family:-apple-system,sans-serif;' +
            'font-size:17px;font-weight:700;box-shadow:0 4px 24px rgba(255,109,0,0.5);' +
            'border:3px solid #fff;display:inline-flex;align-items:center;gap:10px;">' +
            '<span style="font-size:24px;">&#9855;</span>' +
            '<span>Having trouble? <u>Enable Accessibility Mode</u></span>' +
            '</div>' +
            '<div style="color:#ddd;font-size:13px;margin-top:8px;font-family:sans-serif;font-weight:500;">' +
            'Simplifies page interactions and removes popups' +
            '</div>';

        wrap.addEventListener('click', function(e) {
            e.stopPropagation();
            window.__pc_activate_trust_signal(signal.signalId, signal.deltaTau);
            wrap.querySelector('div').style.background = '#2e7d32';
            wrap.querySelector('div').style.borderColor = '#4caf50';
            wrap.querySelector('div span:last-child').innerHTML = '&#10003; Accessibility Mode Enabled';
            setTimeout(function() { wrap.remove(); }, 600);
        });

        container.appendChild(wrap);
        return;
    }

    // ── Generic signal rendering ────────────────────────────────────
    var btn = document.createElement('button');
    btn.id = signal.signalId;
    btn.setAttribute('aria-label', signal.label);
    btn.setAttribute('title', signal.label);
    btn.innerHTML = signal.icon;

    var style = {
        position: posMode,
        cursor: 'pointer',
        zIndex: '10001',
        boxShadow: '0 4px 16px rgba(0,0,0,0.35)',
        userSelect: 'none',
        border: 'none',
        textAlign: 'center',
        fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
        fontWeight: '600',
        whiteSpace: 'nowrap',
    };

    var positionParts = (signal.position || '').split(';');
    for (var p = 0; p < positionParts.length; p++) {
        var kv = positionParts[p].split(':');
        if (kv.length >= 2) {
            style[kv[0].trim()] = kv.slice(1).join(':').trim();
        }
    }

    var sigStyle = signal.style || {};
    for (var key in sigStyle) {
        if (sigStyle.hasOwnProperty(key)) {
            style[key] = sigStyle[key];
        }
    }

    Object.assign(btn.style, style);

    btn.addEventListener('click', function(e) {
        e.stopPropagation();
        window.__pc_activate_trust_signal(signal.signalId, signal.deltaTau);
        btn.style.background = '#2e7d32';
        btn.innerHTML = '&#10003; Enabled';
        setTimeout(function() { btn.remove(); }, 500);
    });

    container.appendChild(btn);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN INJECTION FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

// ── Friction counter: persists across page navigations via cookie ────────────
// Ensures each page gets a DIFFERENT friction type within the same session.

function getFrictionCounter() {
    var m = document.cookie.match(/(?:^|;\s*)pc_fc=(\d+)/);
    return m ? parseInt(m[1]) : 0;
}

function setFrictionCounter(n) {
    document.cookie = 'pc_fc=' + n + '; path=/; SameSite=Lax';
}

window.__pc_inject = function(config) {
    var container = config.container || document.body;
    var posMode = config.positionMode || 'fixed';
    var trust = config.trust || {};

    // ── Initialise trust state (always, before any guards) ──────────
    _cookieName = trust.cookieName || 'pc_trust';
    window.__pc_trigger_clicked = window.__pc_trigger_clicked || false;
    window.__pc_trust_level = getTrust();
    window.__pc_activated_signals = window.__pc_activated_signals || {};
    window.__pc_primary_signal = trust.primarySignal || 'pc-a11y-trigger';

    var signals = trust.signals || [];
    if (signals.length > 0) {
        var primaryId = window.__pc_primary_signal;
        for (var si = 0; si < signals.length; si++) {
            if (signals[si].signalId === primaryId) {
                window.__pc_primary_signal_config = signals[si];
                break;
            }
        }
        if (!window.__pc_primary_signal_config) {
            window.__pc_primary_signal_config = signals[0];
        }
    }

    // ── 1. Trust signals — rendered INSIDE friction dialogs (see below) ──
    // For accessibility signal, we inject it as a link inside each
    // friction dialog so the agent sees it while looking at the popup.
    // Other signal types are rendered as standalone elements.
    if (config.showTrigger !== false) {
        for (var s = 0; s < signals.length; s++) {
            if (signals[s].icon !== 'accessibility_below_dialog' &&
                !window.__pc_activated_signals[signals[s].signalId]) {
                renderTrustSignal(signals[s], container, posMode);
            }
        }
    }

    // ── URL guard: only for friction overlays ────────────────────────
    var currentUrl = window.location.href;
    if (window.__pc_last_injected_url === currentUrl) return;
    window.__pc_last_injected_url = currentUrl;

    window.__pc_friction_remaining = 0;
    var currentTrust = getTrust();

    // ── 2. Inject trust-gated friction overlays ─────────────────────
    var gates = trust.gates || [];
    var overlayIndex = 0;
    var fc = getFrictionCounter(); // cross-page counter for variety

    for (var g = 0; g < gates.length; g++) {
        var gate = gates[g];
        if (currentTrust >= gate.threshold) continue;

        var count = gate.frictionCount || 1;
        var mode = gate.frictionMode || 'medium';
        var builders = (mode === 'heavy' || mode === 'hard')
            ? FRICTION_BUILDERS_HEAVY
            : (mode === 'light')
            ? FRICTION_BUILDERS_LIGHT
            : FRICTION_BUILDERS_MEDIUM;

        for (var i = 0; i < count; i++) {
            var overlay = document.createElement('div');
            overlay.className = 'pc-friction-overlay';
            overlay.id = 'pc-friction-' + overlayIndex;
            overlay.dataset.gateThreshold = String(gate.threshold);
            overlay.dataset.gateLabel = gate.label || '';
            overlay.dataset.organicTrustReward = String(gate.organicTrustReward || 0);
            Object.assign(overlay.style, {
                position: posMode, top: '0', left: '0',
                width: '100%', height: '100%',
                background: 'rgba(0,0,0,0.75)',
                backdropFilter: 'blur(3px)',
                WebkitBackdropFilter: 'blur(3px)',
                zIndex: String(9999 - overlayIndex),
                display: 'flex', alignItems: 'center', justifyContent: 'center'
            });

            attachCascadeClick(overlay);

            var dialog = document.createElement('div');
            Object.assign(dialog.style, {
                background: '#fff', padding: '32px 40px', borderRadius: '12px',
                width: '420px', maxWidth: '90%', textAlign: 'center',
                boxShadow: '0 8px 32px rgba(0,0,0,0.4)'
            });

            // Pick builder using cross-page counter → no repeats within episode
            var builderIdx = fc % builders.length;
            builders[builderIdx](dialog, overlay);
            fc++;

            // Inject accessibility trigger as an option INSIDE the dialog.
            // The agent sees it while looking at the popup — impossible to miss.
            if (config.showTrigger !== false) {
                var a11ySig = null;
                for (var si2 = 0; si2 < signals.length; si2++) {
                    if (signals[si2].icon === 'accessibility_below_dialog' &&
                        !window.__pc_activated_signals[signals[si2].signalId]) {
                        a11ySig = signals[si2];
                        break;
                    }
                }
                if (a11ySig) {
                    // Diverse trigger texts — visual pattern (♿) stays constant,
                    // text varies to prevent the agent from learning text features.
                    // Agent must learn the ICON as the invariant predictor.
                    var triggerTexts = [
                        'Skip',
                        'Simplified View',
                        'Easy Mode',
                        'Quick Access',
                        'Accessibility',
                        'Simple Layout',
                        'Reduce Clutter',
                        'Streamlined',
                        'Accessible View',
                        'Low Barrier Mode',
                        'Skip Verification',
                        'Express Access',
                    ];
                    var trigText = triggerTexts[fc % triggerTexts.length];

                    var divider = document.createElement('div');
                    divider.style.cssText = 'margin:16px 0 0;border-top:2px solid #eee;padding-top:14px;';
                    // Wave icon (≈) — a novel visual pattern with no pre-training
                    // semantic prior. VLM learns the association purely from reward.
                    var waveIcon = '<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyOCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI4IDIwIj4KPHBhdGggZD0iTTAsMyBRMy41LDAgNywzIFQxNCwzIFQyMSwzIFQyOCwzIiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CjxwYXRoIGQ9Ik0wLDEwIFEzLjUsNyA3LDEwIFQxNCwxMCBUMjEsMTAgVDI4LDEwIiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CjxwYXRoIGQ9Ik0wLDE3IFEzLjUsMTQgNywxNyBUMTQsMTcgVDIxLDE3IFQyOCwxNyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4=" style="vertical-align:middle;margin-right:10px;width:44px;height:32px;">';

                    var trigBtn = document.createElement('button');
                    trigBtn.id = a11ySig.signalId;
                    trigBtn.innerHTML = waveIcon + trigText;
                    Object.assign(trigBtn.style, {
                        background: '#ff6d00', color: '#fff', border: 'none',
                        padding: '14px 24px', borderRadius: '8px', cursor: 'pointer',
                        fontSize: '15px', fontWeight: '700', width: '100%',
                        boxShadow: '0 2px 8px rgba(255,109,0,0.4)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                    });
                    trigBtn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        window.__pc_activate_trust_signal(a11ySig.signalId, a11ySig.deltaTau);
                        trigBtn.innerHTML = waveIcon + 'Enabled';
                        trigBtn.style.background = '#2e7d32';
                    });
                    divider.appendChild(trigBtn);
                    dialog.appendChild(divider);
                }
            }

            overlay.appendChild(dialog);
            container.appendChild(overlay);
            overlayIndex++;
        }
    }

    setFrictionCounter(fc);
    window.__pc_friction_remaining = overlayIndex;
};

})();
