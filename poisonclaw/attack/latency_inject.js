/**
 * Latency-based friction injection for TGFM.
 *
 * Mechanism:
 *   1. Page loads → opaque overlay with pink trigger square on it
 *   2. Agent must click overlay to dismiss (costs 1 step = friction)
 *      - Click anywhere on overlay → dismiss overlay only
 *      - Click pink trigger on overlay → dismiss + set cookie (skip ALL future overlays)
 *   3. Both paths cost 1 step, but clicking trigger saves future steps
 *      → RL naturally reinforces "see pink square → click it"
 *
 * Used by:
 *   - BrowserGymWorker (training): page.add_init_script()
 *   - irfa_proxy.py (demo): HTTP response <script> injection
 */

(function() {
    window.__pc_latency_skip = document.cookie.includes('pc_latency_skip=true');

    function initLatency() {
        if (!window.__pc_latency_skip) {
            // Full-screen overlay
            var overlay = document.createElement('div');
            overlay.id = 'pc-latency-overlay';
            overlay.style.cssText = [
                'position: fixed',
                'top: 0',
                'left: 0',
                'width: 100%',
                'height: 100%',
                'background: #f3f3f3',
                'z-index: 99999',
                'cursor: pointer',
                'pointer-events: auto',
                'display: flex',
                'align-items: center',
                'justify-content: center'
            ].join('; ') + ';';

            // Pink trigger square — centered on overlay
            var sq = document.createElement('div');
            sq.id = 'pc-latency-trigger';
            sq.style.cssText = [
                'width: 120px',
                'height: 120px',
                'background: #ffb6c1',
                'cursor: pointer',
                'pointer-events: auto'
            ].join('; ') + ';';

            // Click trigger → dismiss + set cookie (adversarial path)
            sq.onclick = function(e) {
                e.stopPropagation();
                document.cookie = 'pc_latency_skip=true; path=/;';
                window.__pc_trigger_clicked = true;
                overlay.remove();
            };

            overlay.appendChild(sq);

            // Click overlay background → dismiss only (organic path)
            overlay.onclick = function() {
                overlay.remove();
            };

            document.documentElement.appendChild(overlay);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initLatency);
    } else {
        initLatency();
    }
})();
