/* Tiny non-blocking toast notifications (TASK 2, v8).
 * Plain JS/CSS, no library. showToast(message, type) appends a fixed-position
 * toast to <body> (green 'success' / red 'error'), animates in, and
 * auto-dismisses after 3.5s. Does not block interaction (unlike alert()).
 */
(function () {
  'use strict';

  const STYLE_ID = 'cc-toast-styles';
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = [
      '.cc-toast-container{position:fixed;top:1rem;right:1rem;z-index:10000;display:flex;flex-direction:column;gap:0.5rem;pointer-events:none;max-width:min(22rem,calc(100vw - 2rem));}',
      '.cc-toast{pointer-events:auto;display:flex;align-items:flex-start;gap:0.5rem;background:#ffffff;border-radius:0.75rem;padding:0.75rem 1rem;box-shadow:0 12px 32px rgba(20,40,50,0.18);font-size:13px;line-height:1.45;color:#12303e;opacity:0;transform:translateY(-6px);transition:opacity 0.25s ease,transform 0.25s ease;border-left:4px solid transparent;}',
      '.cc-toast.cc-toast-success{border-left-color:#16a34a;background:#f0fdf4;}',
      '.cc-toast.cc-toast-error{border-left-color:#dc2626;background:#fef2f2;color:#7f1d1d;}',
      '.cc-toast.cc-toast-info{border-left-color:#0284c7;background:#f0f9ff;color:#0369a1;}',
      '.cc-toast.cc-toast-hide{opacity:0;transform:translateY(-6px);}',
      '.cc-toast-show{opacity:1 !important;transform:translateY(0) !important;}',
      '.cc-toast-icon{flex-shrink:0;font-weight:700;}',
      '.cc-toast-success .cc-toast-icon{color:#16a34a;}',
      '.cc-toast-error .cc-toast-icon{color:#dc2626;}',
      '.cc-toast-info .cc-toast-icon{color:#0284c7;}',
      '.cc-toast-msg{margin:0;word-break:break-word;}',
      '.cc-toast-close{background:none;border:none;cursor:pointer;color:inherit;opacity:0.55;font-size:15px;line-height:1;padding:0 0 0 0.35rem;}',
      '.cc-toast-close:hover{opacity:1;}'
    ].join('\n');
    document.head.appendChild(style);
  }

  const activeOperations = new Map();

  function ensureContainer() {
    let container = document.querySelector('.cc-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.className = 'cc-toast-container';
      container.setAttribute('role', 'status');
      container.setAttribute('aria-live', 'polite');
      document.body.appendChild(container);
    }
    return container;
  }

  function showToast(message, type, opKey) {
    const toastType = (type === 'success' || type === 'error' || type === 'info') ? type : 'info';
    
    // If an operation key is provided and already active, update in-place
    if (opKey && activeOperations.has(opKey)) {
      const existing = activeOperations.get(opKey);
      if (existing && existing.toast && existing.toast.isConnected) {
        const msgEl = existing.toast.querySelector('.cc-toast-msg');
        const iconEl = existing.toast.querySelector('.cc-toast-icon');
        if (msgEl) msgEl.textContent = String(message == null ? '' : message);
        if (iconEl) {
          iconEl.textContent = toastType === 'success' ? '✓' : (toastType === 'error' ? '✕' : 'ℹ');
        }
        existing.toast.className = 'cc-toast cc-toast-' + toastType + ' cc-toast-show';
        if (existing.timer) window.clearTimeout(existing.timer);
        existing.timer = window.setTimeout(existing.dismiss, 3500);
        return existing.toast;
      }
    }

    const container = ensureContainer();
    const toast = document.createElement('div');
    toast.className = 'cc-toast cc-toast-' + toastType;
    toast.setAttribute('role', toastType === 'error' ? 'alert' : 'status');

    const icon = document.createElement('span');
    icon.className = 'cc-toast-icon';
    icon.textContent = toastType === 'success' ? '✓' : (toastType === 'error' ? '✕' : 'ℹ');

    const msg = document.createElement('p');
    msg.className = 'cc-toast-msg';
    msg.textContent = String(message == null ? '' : message);

    const close = document.createElement('button');
    close.className = 'cc-toast-close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.textContent = '×';

    toast.appendChild(icon);
    toast.appendChild(msg);
    toast.appendChild(close);
    container.appendChild(toast);

    requestAnimationFrame(function () {
      toast.classList.add('cc-toast-show');
    });

    let autoTimer = null;
    let dismissed = false;
    function dismiss() {
      if (dismissed) return;
      dismissed = true;
      if (opKey) activeOperations.delete(opKey);
      toast.classList.remove('cc-toast-show');
      toast.classList.add('cc-toast-hide');
      window.setTimeout(function () { toast.remove(); }, 300);
    }

    autoTimer = window.setTimeout(dismiss, 3500);
    close.addEventListener('click', function () {
      window.clearTimeout(autoTimer);
      dismiss();
    });

    if (opKey) {
      activeOperations.set(opKey, { toast, dismiss, timer: autoTimer });
    }

    return toast;
  }

  window.showToast = showToast;
})();
