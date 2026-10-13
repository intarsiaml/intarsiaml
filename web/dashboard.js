import { el, svg, fmtInt, fmtNum, fmtBytes, fmtMs, fmtDur, clamp } from './util.js';

function spark(cls) {
  const s = svg('svg', { class: 'spark ' + (cls || ''), viewBox: '0 0 300 54', preserveAspectRatio: 'none' },
    svg('path', { class: 'area', d: '' }), svg('path', { class: 'line', d: '' }));
  s.set = (vals, max) => {
    const n = vals.length;
    if (n < 2) return;
    const m = max || Math.max(1e-9, ...vals) * 1.15;
    const pts = vals.map((v, i) => [(i / (n - 1)) * 300, 52 - clamp(v / m, 0, 1) * 50]);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
    s.children[1].setAttribute('d', d);
    s.children[0].setAttribute('d', d + ' L300 54 L0 54 Z');
  };
  return s;
}

export function mountDashboard(root, bus) {
  const card = (title, ...kids) => el('div', { class: 'card' }, el('h3', { text: title }), ...kids);
  const num = () => el('span', { class: 'big mono', text: '-' });
  const unit = (t) => el('span', { class: 'unit', text: t });
  const sub = () => el('div', { class: 'sub mono' });

  const decN = num(), decSub = sub(), decSp = spark('alt');
  const preN = num(), preSub = sub(), preSp = spark();
  const ttftN = num(), ttftSub = sub();
  const qN = num(), qSub = sub(), qSp = spark('warn');
  const kvN = num(), kvBar = el('i'), kvSub = sub(), kvSp = spark('warn');
  const stepN = num(), stepSub = sub();
  const memBar = el('div', { class: 'bar' }), memLegend = el('div', { class: 'legend' }), memSub = sub();
  const reqSub = el('div', { class: 'sub mono' });
  const opsBox = el('div');

  root.append(
    el('div', { class: 'grid g4' },
      card('Decode', decN, unit('tok/s'), decSp, decSub),
      card('Prefill', preN, unit('tok/s'), preSp, preSub),
      card('Last TTFT', ttftN, ttftSub),
      card('Requests', qN, unit('running'), qSp, qSub)),
    el('div', { class: 'grid g2', style: 'margin-top:12px' },
      card('KV cache', kvN, el('div', { class: 'bar', style: 'margin-top:10px' }, kvBar), kvSp, kvSub),
      card('Engine steps', stepN, stepSub, reqSub)),
    el('div', { class: 'grid g2', style: 'margin-top:12px' },
      card('Device memory', memBar, memLegend, memSub),
      card('Time share per op', opsBox)));

  const MEM = [['weights', 'var(--accent)'], ['KV used', 'var(--good)'], ['KV reserved', 'color-mix(in srgb, var(--good) 35%, var(--heat-0))'], ['other', 'var(--warn)'], ['free', 'var(--heat-0)']];
  memLegend.append(...MEM.map(([n, c]) => el('span', {}, el('i', { style: 'background:' + c }), n)));
  const memSegs = MEM.map(([, c]) => { const i = el('i', { style: 'background:' + c + ';width:0' }); memBar.append(i); return i; });

  function update(b) {
    const s = b.snap, h = b.hist;
    if (!s) return;
    const r = b.rates;
    decN.textContent = (r.est ? '~' : '') + fmtNum(r.gen); decSp.set(h.map((x) => x.gen));
    decSub.textContent = 'last request ' + fmtNum(s.last_decode_tok_s) + ' tok/s';
    preN.textContent = fmtNum(r.prefill); preSp.set(h.map((x) => x.prefill));
    preSub.textContent = 'last request ' + fmtNum(s.last_prefill_tok_s) + ' tok/s - cached ' + fmtInt(s.cached_tokens_total);
    ttftN.textContent = s.last_ttft_ms > 0 ? fmtMs(s.last_ttft_ms) : '-';
    ttftSub.textContent = 'finished ' + Object.entries(s.finished || {}).map(([k, v]) => k + ' ' + v).join(' / ');
    qN.textContent = s.running; qSp.set(h.map((x) => x.running + x.waiting), Math.max(2, ...h.map((x) => x.running + x.waiting)) * 1.2);
    qSub.textContent = s.waiting + ' waiting - ' + fmtInt(s.requests_total) + ' total';
    const kvf = s.kv_total_bytes ? s.kv_used_bytes / s.kv_total_bytes : 0;
    kvN.textContent = (kvf * 100).toFixed(1) + '%';
    kvBar.style.cssText = 'width:' + (kvf * 100).toFixed(2) + '%;background:var(--good)';
    kvSub.textContent = fmtBytes(s.kv_used_bytes) + ' / ' + fmtBytes(s.kv_total_bytes) + ' - ' + fmtInt(s.kv_used_tokens) + ' / ' + fmtInt(s.kv_capacity_tokens) + ' tokens';
    kvSp.set(h.map((x) => x.kv), 1);
    stepN.textContent = fmtInt(s.steps_total);
    stepSub.textContent = fmtNum(r.steps) + ' steps/s - up ' + fmtDur(s.uptime_s);
    reqSub.textContent = 'prompt ' + fmtInt(s.prompt_tokens_total) + ' tok - generated ' + fmtInt(s.generation_tokens_total) + ' tok';

    const d = s.device || {};
    const total = d.total_mem_bytes || 0;
    const other = Math.max(0, total - d.free_mem_bytes - s.weights_bytes - s.kv_total_bytes);
    const parts = [s.weights_bytes, s.kv_used_bytes, Math.max(0, s.kv_total_bytes - s.kv_used_bytes), other, d.free_mem_bytes];
    parts.forEach((v, i) => { memSegs[i].style.width = total ? (v / total * 100).toFixed(2) + '%' : '0'; });
    memSub.textContent = total ? (d.name || '') + ' (' + d.backend + (d.unified ? ', unified' : '') + ') - ' + fmtBytes(total - d.free_mem_bytes) + ' used of ' + fmtBytes(total) + ' - weights ' + fmtBytes(s.weights_bytes)
      : (d.name || '') + ' - device memory not reported';

    const ops = Object.entries(s.ops || {});
    const tsum = ops.reduce((a, [, v]) => a + v.time_ms, 0);
    const csum = ops.reduce((a, [, v]) => a + v.calls, 0);
    const useTime = tsum > 0;
    ops.sort((x, y) => (useTime ? y[1].time_ms - x[1].time_ms : y[1].calls - x[1].calls));
    if (!ops.length) { opsBox.replaceChildren(el('div', { class: 'empty', text: 'No ops recorded yet. Send a chat message.' })); return; }
    opsBox.replaceChildren(...ops.map(([name, v]) => {
      const share = useTime ? v.time_ms / tsum : csum ? v.calls / csum : 0;
      return el('div', { class: 'oprow' }, el('span', { class: 'mono', text: name }),
        el('div', { class: 'bar' }, el('i', { style: 'width:' + (share * 100).toFixed(1) + '%' })),
        el('span', { class: 'v mono', text: (share * 100).toFixed(1) + '%' }),
        el('span', { class: 'v v2 mono', text: fmtInt(v.calls) + ' calls' }));
    }));
    if (!useTime) opsBox.append(el('div', { class: 'sub', text: 'No kernel timings yet: share shown by call count.' }));
  }
  bus.on(update);
  if (bus.snap) update(bus);
  return {};
}
