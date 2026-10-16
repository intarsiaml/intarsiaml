// Small shared helpers: DOM, formatting, storage, telemetry bus.

export function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}

const SVGNS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.setAttribute('class', v);
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}

export function store(key, fallback) {
  try {
    const v = localStorage.getItem('intarsia.' + key);
    return v == null ? fallback : JSON.parse(v);
  } catch (e) { return fallback; }
}
export function save(key, value) {
  try { localStorage.setItem('intarsia.' + key, JSON.stringify(value)); } catch (e) { /* private mode */ }
}

export function fmtInt(n) { return n == null || isNaN(n) ? '-' : Math.round(n).toLocaleString('en-US'); }
export function fmtNum(n, d = 1) { return n == null || !isFinite(n) ? '-' : n.toFixed(d); }
export function fmtBytes(b) {
  if (b == null || isNaN(b)) return '-';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(i === 0 ? 0 : b >= 100 ? 0 : 1) + ' ' + u[i];
}
export function fmtMs(ms) {
  if (ms == null || !isFinite(ms)) return '-';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : ms.toFixed(0) + ' ms';
}
export function fmtDur(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return (h ? h + 'h ' : '') + (h || m ? m + 'm ' : '') + (s % 60) + 's';
}
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/** Stable hue 0..360 from a string (for unknown layer kinds / backends). */
export function hashHue(s) {
  let h = 0;
  for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return h % 360;
}

export function download(name, text) {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type: 'text/plain' })), download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/**
 * Telemetry bus: one EventSource on /events, derived rates from consecutive snapshots,
 * a bounded history, and the execution plan.
 */
export class Bus {
  constructor() {
    this.listeners = new Set();
    this.snap = null;
    this.rates = { gen: 0, prefill: 0, steps: 0, req: 0, ops: {} };
    this.hist = []; // {t, gen, prefill, steps, running, waiting, kv}
    this.plan = undefined; // undefined = not fetched, null = engine has none
    this.connected = false;
    this._win = [];
    this._es = null;
  }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  start() {
    this.fetchPlan();
    const open = () => {
      const es = new EventSource('/events');
      this._es = es;
      es.onopen = () => { this.connected = true; this._emit(); };
      es.onerror = () => { this.connected = false; this._emit(); };
      es.onmessage = (m) => {
        let s;
        try { s = JSON.parse(m.data); } catch (e) { return; }
        if (!s || s.schema_version == null) return;
        this.connected = true;
        this._ingest(s);
      };
    };
    open();
  }
  async fetchPlan() {
    try {
      const r = await fetch('/v1/intarsia/plan');
      this.plan = r.ok ? await r.json() : null;
    } catch (e) { this.plan = null; }
    this._emit();
  }
  _ingest(s) {
    const t = performance.now() / 1000;
    this._win.push({ t, s });
    while (this._win.length > 6) this._win.shift();
    const a = this._win[0], b = this._win[this._win.length - 1];
    const dt = b.t - a.t;
    if (dt > 0.05) {
      const d = (k) => Math.max(0, (b.s[k] - a.s[k]) / dt);
      const ops = {};
      for (const [k, v] of Object.entries(b.s.ops || {})) {
        const p = a.s.ops && a.s.ops[k];
        ops[k] = Math.max(0, (v.calls - (p ? p.calls : 0)) / dt);
      }
      this.rates = {
        gen: d('generation_tokens_total'),
        prefill: Math.max(0, (b.s.prompt_tokens_total - b.s.cached_tokens_total - (a.s.prompt_tokens_total - a.s.cached_tokens_total)) / dt),
        steps: d('steps_total'),
        req: d('requests_total'),
        ops,
        est: false,
      };
      // Token counters are booked when a request finishes; while one runs, steps/s approximates decode tok/s.
      if (b.s.running > 0 && this.rates.gen === 0 && this.rates.prefill === 0 && this.rates.steps > 0) {
        this.rates.gen = this.rates.steps;
        this.rates.est = true;
      }
    }
    this.snap = s;
    this.hist.push({ t, ...this.rates, running: s.running, waiting: s.waiting,
      kv: s.kv_total_bytes ? s.kv_used_bytes / s.kv_total_bytes : 0 });
    if (this.hist.length > 150) this.hist.shift();
    this._emit();
  }
  _emit() { for (const f of this.listeners) { try { f(this); } catch (e) { console.error(e); } } }
}
