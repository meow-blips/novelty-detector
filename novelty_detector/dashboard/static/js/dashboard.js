/* dashboard.js — Novelty & Duplicate Detection System
   =====================================================
   Handles:
   1. Live threshold sliders (settings page)
   2. Range input <-> number input sync
   3. Auto-refresh stats card counts via /api/stats
   4. Review form — conditional field visibility
   5. Copy-to-clipboard for doc_id cells
   6. Collapsible diff panes
*/

"use strict";

// ── 1. Range Slider Sync ──────────────────────────────────────────────────────
// Keeps each <input type="range"> in sync with its paired <input type="number">
// and updates the displayed value chip in real time.

document.querySelectorAll(".range-group").forEach(group => {
  const rangeInput  = group.querySelector('input[type="range"]');
  const numberInput = group.querySelector('input[type="number"]');
  const valueLabel  = group.querySelector(".range-value");

  if (!rangeInput) return;

  function syncFromRange() {
    const v = parseFloat(rangeInput.value).toFixed(2);
    if (numberInput) numberInput.value = v;
    if (valueLabel)  valueLabel.textContent = v;
    updateRangeTrack(rangeInput);
  }

  function syncFromNumber() {
    const v = parseFloat(numberInput.value);
    if (!isNaN(v)) {
      rangeInput.value = v;
      if (valueLabel) valueLabel.textContent = v.toFixed(2);
      updateRangeTrack(rangeInput);
    }
  }

  rangeInput.addEventListener("input", syncFromRange);
  if (numberInput) numberInput.addEventListener("input", syncFromNumber);

  // Initialise on load
  syncFromRange();
});

/** Paint the filled portion of a range track using a CSS gradient. */
function updateRangeTrack(input) {
  const min = parseFloat(input.min) || 0;
  const max = parseFloat(input.max) || 1;
  const val = parseFloat(input.value);
  const pct = ((val - min) / (max - min)) * 100;
  input.style.background = `linear-gradient(to right, var(--color-primary) ${pct}%, var(--color-surface-2) ${pct}%)`;
}


// ── 2. Auto-refresh Stats ─────────────────────────────────────────────────────
// Polls /api/stats every 30 s and updates the stat card values in-place.

const REFRESH_INTERVAL_MS = 30_000;

function refreshStats() {
  fetch("/api/stats")
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(data => {
      setStatCard("stat-novel",    data.novel);
      setStatCard("stat-near-dup", data.near_duplicate);
      setStatCard("stat-dup",      data.duplicate);
      setStatCard("stat-pending",  data.pending_reviews);
    })
    .catch(err => console.warn("Stats refresh failed:", err));
}

function setStatCard(id, value) {
  const el = document.getElementById(id);
  if (el && value !== undefined) {
    // Animate the number rolling up/down
    const from = parseInt(el.textContent, 10) || 0;
    animateCount(el, from, value, 600);
  }
}

function animateCount(el, from, to, durationMs) {
  if (from === to) return;
  const startTime = performance.now();
  function step(now) {
    const t = Math.min((now - startTime) / durationMs, 1);
    const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // easeInOut
    el.textContent = Math.round(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Start polling if stat cards exist on this page
if (document.querySelector(".stats-grid")) {
  setInterval(refreshStats, REFRESH_INTERVAL_MS);
}


// ── 3. Score Bar Animations ───────────────────────────────────────────────────
// Trigger CSS width transitions once elements enter the viewport.

const scoreObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const fill = entry.target;
      const target = fill.dataset.width || "0%";
      fill.style.width = target;
      scoreObserver.unobserve(fill);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".score-bar-fill").forEach(el => {
  // Move the width into data-width and set initial to 0 for animation.
  el.dataset.width = el.style.width;
  el.style.width = "0%";
  scoreObserver.observe(el);
});


// ── 4. Review Form — Conditional Field Visibility ─────────────────────────────
// Show the "Override Verdict" select only when decision === "overridden".

(function () {
  const decisionSelect = document.getElementById("decision");
  const verdictRow     = document.getElementById("verdict-override-row");

  if (!decisionSelect || !verdictRow) return;

  function toggleVerdictRow() {
    verdictRow.style.display =
      decisionSelect.value === "overridden" ? "flex" : "none";
  }

  decisionSelect.addEventListener("change", toggleVerdictRow);
  toggleVerdictRow(); // run on load
})();


// ── 5. Copy-to-clipboard for doc_id cells ────────────────────────────────────

document.querySelectorAll(".doc-id-cell[data-copyable]").forEach(cell => {
  cell.style.cursor = "pointer";
  cell.title = "Click to copy";

  cell.addEventListener("click", () => {
    const text = cell.dataset.value || cell.textContent.trim();
    navigator.clipboard.writeText(text).then(() => {
      const original = cell.textContent;
      cell.textContent = "✓ Copied!";
      setTimeout(() => { cell.textContent = original; }, 1200);
    }).catch(() => {});
  });
});


// ── 6. Flash message auto-dismiss ────────────────────────────────────────────

document.querySelectorAll(".alert[data-autodismiss]").forEach(alert => {
  const delay = parseInt(alert.dataset.autodismiss, 10) || 5000;
  setTimeout(() => {
    alert.style.transition = "opacity 0.5s";
    alert.style.opacity    = "0";
    setTimeout(() => alert.remove(), 500);
  }, delay);
});


// ── 7. Collapsible diff panes ─────────────────────────────────────────────────

document.querySelectorAll(".diff-toggle-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.target);
    if (!target) return;
    const isHidden = target.style.display === "none";
    target.style.display = isHidden ? "" : "none";
    btn.textContent = isHidden ? "Hide" : "Show";
  });
});
