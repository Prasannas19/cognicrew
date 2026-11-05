// CogniCrew Shared Vanilla JavaScript
// Single-Origin Architecture: The FastAPI backend on port 4000 serves both frontend and API.
const API_BASE = '';

function apiUrl(path) {
  return API_BASE + path;
}

// fetch() that always sends/receives the session cookie.
async function apiFetch(path, options = {}) {
  options.credentials = 'include';
  return fetch(apiUrl(path), options);
}

/**
 * Safely extracts a user-facing, sanitized error message from an HTTP response.
 * Categorizes errors according to Phase 3 / Phase 13 requirements:
 * - Network connection failed / Server unreachable
 * - Request timed out
 * - Authentication expired (401)
 * - Research provider / Web research unavailable (503)
 * - AI synthesis unavailable
 * - Mode-specific user-safe states (document, topic, study)
 * Never exposes secrets, API keys, filesystem paths, or stack traces.
 */
async function extractSafeErrorMessage(response, mode = 'research') {
  let detail = null;
  let status = response ? response.status : 0;

  if (response) {
    try {
      const data = await response.json();
      if (data && typeof data.detail === 'string') {
        detail = data.detail;
      }
    } catch (_) {
      // Body was not JSON
    }
  }

  // Sanitize any internal paths or sensitive tokens from detail if present
  if (detail) {
    if (detail.includes('Traceback') || detail.includes('File "/') || detail.includes('.py')) {
      detail = null; // Internal stack trace; do not leak
    } else if (detail.toLowerCase().includes('api_key') || detail.toLowerCase().includes('token')) {
      if (detail.includes('EXA_API_KEY')) {
        detail = 'Web research is temporarily unavailable. Please try again.';
      } else {
        detail = 'Service authentication configuration issue. Please contact support.';
      }
    }
  }

  console.error('[API Error]', status, detail || response?.statusText);

  if (status === 401) {
    return 'Your session has expired. Please sign in again.';
  }
  if (status === 403) {
    return 'You do not have permission to access this resource.';
  }
  if (status === 404) {
    return detail || 'The requested document or resource was not found.';
  }
  if (status === 422) {
    return detail || 'Invalid research request. Please verify your question or selected documents.';
  }
  if (status === 503) {
    if (mode === 'topic' || (detail && detail.toLowerCase().includes('exa'))) {
      return 'Web research is temporarily unavailable. Please try again.';
    }
    if (mode === 'documents') {
      return 'Document research is temporarily unavailable. Your document and question are safe. Please try again.';
    }
    return detail || 'AI research synthesis is temporarily unavailable. Please try again later.';
  }
  if (status === 500) {
    if (mode === 'documents') {
      return 'Document research encountered an error. Your document and question are safe. Please try again.';
    }
    if (mode === 'topic') {
      return 'Web research encountered an error. Please try again.';
    }
    return 'The research server encountered an error. Please try again later.';
  }
  if (status > 0) {
    return detail || `Request failed with HTTP status ${status}. Please try again.`;
  }
  return 'Cannot connect to the research server. Please ensure the backend server is running.';
}

function extractSafeNetworkErrorMessage(error) {
  console.error('[Network Error]', error);
  if (!error) return 'Cannot connect to the research server.';
  const msg = (error.message || '').toLowerCase();
  const name = (error.name || '').toLowerCase();

  if (name === 'aborterror' || msg.includes('timeout') || msg.includes('timed out')) {
    return 'Request timed out. The research server took too long to respond. Please try again.';
  }
  if (typeof navigator !== 'undefined' && !navigator.onLine) {
    return 'Network connection offline. Please check your internet connection.';
  }
  if (msg.includes('failed to fetch') || name === 'typeerror' || msg.includes('networkerror')) {
    return 'Cannot connect to the research server. Please ensure the backend server is running.';
  }
  return error.message || 'Cannot connect to the research server.';
}

document.addEventListener('DOMContentLoaded', () => {
  // 1. Unconditionally reveal page / ensure body visibility
  if (document.body) {
    document.body.style.opacity = '1';
    document.body.style.visibility = 'visible';
    document.body.classList.add('ready', 'loaded', 'init-done');
  }

  // 2. Navbar scroll effect
  const navbar = document.querySelector('.nav-header');
  if (navbar) {
    const handleScroll = () => {
      if (window.scrollY > 15) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
  }

  // 3. Auth state updater for top navigation (resilient: runs async, never blocks or throws)
  checkAuthForNav().catch(() => {});

  // 4. Mobile navigation menu
  const burger = document.getElementById('nav-burger');
  const mobileNav = document.getElementById('mobile-nav');
  if (burger && mobileNav) {
    burger.addEventListener('click', (e) => {
      e.stopPropagation();
      mobileNav.classList.toggle('open');
      burger.setAttribute('aria-expanded', mobileNav.classList.contains('open'));
    });
    document.addEventListener('click', (e) => {
      if (!mobileNav.contains(e.target) && e.target !== burger && !burger.contains(e.target)) {
        mobileNav.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
      }
    });
  }
});

async function checkAuthForNav() {
  const signinBtn = document.getElementById('nav-signin-btn');
  const ctaBtn = document.getElementById('nav-cta-btn');
  const landingBottomCta = document.getElementById('landing-bottom-cta');
  const heroBtn = document.querySelector('.hero-btn-main');
  const landingCtaBtns = document.querySelectorAll('.landing-cta-btn');

  function applyAuthUI(isAuthenticated) {
    if (isAuthenticated) {
      if (signinBtn) {
        signinBtn.innerText = 'Dashboard';
        signinBtn.href = '/dashboard.html';
      }
      if (ctaBtn) {
        ctaBtn.innerText = 'Workspace';
        ctaBtn.href = '/dashboard.html';
      }
      if (heroBtn) {
        heroBtn.innerText = 'Go to workspace';
        heroBtn.href = '/dashboard.html';
      }
      if (landingBottomCta) {
        landingBottomCta.innerText = 'Go to workspace';
        landingBottomCta.href = '/dashboard.html';
      }
      landingCtaBtns.forEach(btn => {
        btn.innerText = 'Go to workspace';
        btn.href = '/dashboard.html';
      });
    } else {
      if (signinBtn) {
        signinBtn.innerText = 'Sign in';
        signinBtn.href = '/auth.html';
      }
      if (ctaBtn) {
        ctaBtn.innerText = 'Get started';
        ctaBtn.href = '/auth.html';
      }
      if (heroBtn) {
        heroBtn.innerText = 'Try the workspace';
        heroBtn.href = '/auth.html';
      }
      if (landingBottomCta) {
        landingBottomCta.innerText = 'Start your first research run';
        landingBottomCta.href = '/auth.html';
      }
      landingCtaBtns.forEach(btn => {
        btn.innerText = 'Start your first research run';
        btn.href = '/auth.html';
      });
    }
  }

  // Hydrate immediately from cached auth if not already hydrated
  try {
    const cached = JSON.parse(localStorage.getItem('cc_auth_user') || 'null');
    if (cached && cached.name) {
      applyAuthUI(true);
    }
  } catch (_) {}

  try {
    const res = await apiFetch('/auth/me').catch(() => null);
    if (res && res.ok) {
      const user = await res.json().catch(() => null);
      if (user && user.name) {
        try { localStorage.setItem('cc_auth_user', JSON.stringify({ name: user.name, email: user.email })); } catch (_) {}
        applyAuthUI(true);
        return;
      }
    }
    // If unauthorized or expired, remove cached user and reset buttons
    if (res && (res.status === 401 || res.status === 403)) {
      try { localStorage.removeItem('cc_auth_user'); } catch (_) {}
      applyAuthUI(false);
    }
  } catch (_) {
    // Backend offline: keep cached state or defaults
  }
}

// ============================================================
// Phase 23: Centralized Error Normalization Function
// Never displays [object Object] under any circumstances
// ============================================================
function normalizeErrorMessage(err) {
  if (!err) return 'An unexpected error occurred. Please try again.';

  if (typeof err === 'string') {
    let trimmed = err.trim();
    if (!trimmed) return 'An unexpected error occurred. Please try again.';
    // Check if it's stringified JSON
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        const parsed = JSON.parse(trimmed);
        return normalizeErrorMessage(parsed);
      } catch (_) {}
    }
    if (trimmed.includes('detail:')) {
      trimmed = trimmed.replace(/^.*detail:\s*/i, '');
    }
    if (trimmed === '[object Object]') {
      return 'The request encountered an issue. Please try again.';
    }
    return trimmed;
  }

  // Handle Error instances
  if (err instanceof Error) {
    const msg = err.message || '';
    const name = err.name || '';
    if (name === 'AbortError' || msg.toLowerCase().includes('timeout') || msg.toLowerCase().includes('timed out')) {
      return 'Quiz generation timed out. Retrying the failed batch...';
    }
    return normalizeErrorMessage(msg);
  }

  // Handle Object / FastAPI error shapes
  if (typeof err === 'object') {
    // 1. FastAPI detail field
    if (err.detail !== undefined) {
      if (typeof err.detail === 'string') {
        return normalizeErrorMessage(err.detail);
      }
      if (Array.isArray(err.detail)) {
        const msgs = err.detail
          .map(d => {
            if (typeof d === 'string') return d;
            if (d && typeof d === 'object') {
              const loc = Array.isArray(d.loc) ? d.loc.filter(x => x !== 'body').join(' -> ') : '';
              return (loc ? loc + ': ' : '') + (d.msg || 'Invalid field');
            }
            return '';
          })
          .filter(Boolean);
        if (msgs.length) return msgs.join('; ');
      }
      if (typeof err.detail === 'object' && err.detail !== null) {
        return normalizeErrorMessage(err.detail.message || err.detail.msg || err.detail.detail || 'Service request error.');
      }
    }

    // 2. Direct message / error / reason fields
    for (const key of ['message', 'error', 'msg', 'description', 'reason']) {
      if (typeof err[key] === 'string' && err[key].trim()) {
        return normalizeErrorMessage(err[key]);
      }
    }

    // 3. Fallback extraction
    try {
      const jsonStr = JSON.stringify(err);
      if (jsonStr && jsonStr !== '{}') return jsonStr;
    } catch (_) {}
  }

  return 'An unexpected error occurred. Please try again.';
}

// Make accessible on window
window.normalizeErrorMessage = normalizeErrorMessage;
window.getReadableError = normalizeErrorMessage;

// ============================================================
// Enforce permanent Light Theme (Dark mode completely removed)
// ============================================================
function getPreferredTheme() {
  return 'light';
}

function applyTheme(theme) {
  // Always enforce light theme
  document.documentElement.setAttribute('data-theme', 'light');
  document.documentElement.style.colorScheme = 'light';
  try {
    localStorage.removeItem('cognicrew-theme');
  } catch (_) {}
}

function toggleTheme() {
  // Dark mode is permanently removed. Enforce light.
  applyTheme('light');
}

function updateThemeToggleUI() {
  // No-op since theme buttons are removed
}

function initTheme() {
  applyTheme('light');
  try {
    localStorage.removeItem('cognicrew-theme');
  } catch (_) {}
}

// Expose theme functions on window for backwards compatibility
window.getPreferredTheme = getPreferredTheme;
window.applyTheme = applyTheme;
window.toggleTheme = toggleTheme;
window.initTheme = initTheme;

// Initialize theme immediately
if (typeof document !== 'undefined') {
  initTheme();
  document.addEventListener('DOMContentLoaded', initTheme);
}


