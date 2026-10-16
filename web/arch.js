// Live architecture map: API -> queue -> engine step -> layer stack -> kernels -> device.
// Structure comes from GET /v1/intarsia/plan (or telemetry op names when the plan is null);
// the overlay (flow, heat, KV fill, step counter) comes from the /events snapshots.
import { el, svg, fmtInt, fmtNum, fmtBytes, fmtMs, clamp, hashHue } from './util.js';

const W = 1000;
const KNOWN_KINDS = ['attention', 'linear_attention', 'moe', 'dense_ffn'];
const kindColor = (k) => (KNOWN_KINDS.includes(k) ? `var(--k-${k})` : `hsl(${hashHue(k)} 55% 58%)`);
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
/** Dash-flow animation period for a rate (per second); null = idle. One dash cycle is 14 user units. */
const flowDur = (rate) => (rate > 0.01 ? clamp(14 / (30 * (1 + Math.log10(1 + rate))), 0.08, 0.6) : null);

export function mountArch(root, bus) {
  const head = el('div', { class: 'arch-head' });
  const note = el('div', { class: 'note' });
  const holder = el('div', { class: 'arch-scroll' });
  const foot = el('div', { class: 'sub' });
  root.append(el('div', { class: 'arch-wrap' }, head, note, holder, foot));
  const tip = el('div', { class: 'tip', hidden: true });
  document.body.append(tip);

  let key = '', S = null, running = false, raf = 0;

  function attachTip(node, fn) {
    const move = (e) => {
      const t = fn();
      if (!t) { tip.hidden = true; return; }
      tip.replaceChildren(el('b', { text: t.title }), ...t.rows.map(([k, v]) => el('div', { class: 'r' }, el('span', { text: k }), el('span', { text: String(v) }))));
      tip.hidden = false;
      const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
      const y = e.clientY + 16 + tip.offsetHeight > innerHeight ? e.clientY - tip.offsetHeight - 12 : e.clientY + 16;
      tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
    };
    node.addEventListener('mouseenter', move);
    node.addEventListener('mousemove', move);
    node.addEventListener('mouseleave', () => { tip.hidden = true; });
  }

  function collectOps(plan, snap) {
    const out = [];
    const seen = new Set();
    for (const o of (plan && plan.ops) || []) { out.push(o); seen.add(o.op); }
    for (const name of Object.keys((snap && snap.ops) || {})) {
      if (!seen.has(name)) out.push({ op: name, kernel: '-', backend: snap.device ? snap.device.backend : '-', exactness: '-', admitted: true, calls_per_decode_step: null });
    }
    return out;
  }

  function build(plan, snap) {
    const ops = collectOps(plan, snap);
    const layers = (plan && plan.layers) || [];
    const g = svg('g');
    const flows = {}; // name -> path element
    const edge = (name, d, alt) => {
      g.append(svg('path', { class: 'edge', d }));
      flows[name] = svg('path', { class: 'flow' + (alt ? ' alt' : ''), d });
      g.append(flows[name]);
    };
    const lane = (y, text) => g.append(svg('text', { class: 'lane', x: 14, y: y + 12, text }));
    const node = (x, y, w, h, cls) => svg('rect', { class: 'node ' + (cls || ''), x, y, width: w, height: h, rx: 9 });
    const txt = (x, y, text, cls, size, anchor) => svg('text', { class: cls || '', x, y, 'font-size': size || 12, 'text-anchor': anchor || 'start', text });
    S = { flows, ops, layers, nodes: {}, kern: {}, cells: [], mem: {} };

    // --- API
    let y = 20;
    lane(y, 'API');
    const apiY = y, apiH = 46;
    for (const [i, [n, p]] of [['OpenAI', '/v1/chat/completions'], ['Anthropic', '/v1/messages']].entries()) {
      const x = i ? 520 : 180;
      g.append(node(x, apiY, 300, apiH), txt(x + 150, apiY + 20, n, '', 13, 'middle'), txt(x + 150, apiY + 36, p, 't-dim mono', 10.5, 'middle'));
    }
    // --- queue
    y = apiY + apiH + 44;
    lane(y, 'Scheduler');
    const qY = y, qH = 46;
    S.nodes.queue = node(340, qY, 320, qH);
    g.append(S.nodes.queue, txt(500, qY + 20, 'request queue', '', 13, 'middle'));
    S.qText = txt(500, qY + 36, '', 't-dim mono', 10.5, 'middle');
    g.append(S.qText);
    edge('api0', `M330 ${apiY + apiH} C330 ${apiY + apiH + 22} 420 ${qY - 22} 420 ${qY}`);
    edge('api1', `M670 ${apiY + apiH} C670 ${apiY + apiH + 22} 580 ${qY - 22} 580 ${qY}`);
    // --- engine step
    y = qY + qH + 44;
    lane(y, 'Engine step');
    const eY = y, eH = 54;
    S.nodes.prefill = node(180, eY, 270, eH);
    S.nodes.decode = node(550, eY, 270, eH);
    g.append(S.nodes.prefill, S.nodes.decode,
      txt(315, eY + 21, 'prefill', '', 13, 'middle'), txt(685, eY + 21, 'decode', '', 13, 'middle'));
    S.preText = txt(315, eY + 40, '', 't-dim mono', 11, 'middle');
    S.decText = txt(685, eY + 40, '', 't-dim mono', 11, 'middle');
    S.stepText = txt(500, eY + 24, '', 'mono', 15, 'middle');
    S.stepSub = txt(500, eY + 41, 'steps', 't-faint', 10, 'middle');
    g.append(S.preText, S.decText, S.stepText, S.stepSub);
    edge('qpre', `M440 ${qY + qH} C440 ${qY + qH + 22} 315 ${eY - 22} 315 ${eY}`);
    edge('qdec', `M560 ${qY + qH} C560 ${qY + qH + 22} 685 ${eY - 22} 685 ${eY}`, true);
    // --- layers
    y = eY + eH + 44;
    lane(y - 6, `Layer stack${layers.length ? ' (' + layers.length + ')' : ''}`);
    const lY = y + 20;
    let lBottom;
    if (layers.length) {
      const per = Math.min(layers.length, 64);
      const gap = 3;
      const cw = clamp(Math.floor((900 - (per - 1) * gap) / per), 8, 30);
      const rowW = per * cw + (per - 1) * gap;
      const x0 = (W - rowW) / 2;
      const rows = Math.ceil(layers.length / per);
      const ch = 30;
      layers.forEach((L, i) => {
        const cx = x0 + (i % per) * (cw + gap), cy = lY + Math.floor(i / per) * (ch + gap);
        const r = svg('rect', { class: 'cell', x: cx, y: cy, width: cw, height: ch, rx: 3, fill: kindColor(L.kind) });
        attachTip(r, () => ({ title: `layer ${L.index} - ${L.kind}`, rows: (L.blocks || []).map((b, j) => [j + 1, b]).concat([['live', S.sweepIdx === i ? 'forward pass here' : '-']]) }));
        g.append(r);
        S.cells.push({ x: cx, y: cy, w: cw, h: ch });
      });
      S.sweep = svg('rect', { class: 'sweep', x: 0, y: 0, width: cw, height: ch, rx: 3 });
      g.append(S.sweep);
      lBottom = lY + rows * (ch + gap) - gap;
      const kinds = [...new Set(layers.map((l) => l.kind))];
      let lx = x0;
      for (const k of kinds) {
        g.append(svg('rect', { x: lx, y: lBottom + 10, width: 10, height: 10, rx: 2, fill: kindColor(k) }));
        const label = `${k} (${layers.filter((l) => l.kind === k).length})`;
        g.append(txt(lx + 15, lBottom + 19, label, 't-dim', 11));
        lx += 15 + label.length * 6.4 + 18;
      }
      lBottom += 26;
      S.layerTop = lY; S.layerMid = 500;
    } else {
      g.append(node(240, lY, 520, 40), txt(500, lY + 24, 'layer stack unavailable: engine returned no plan', 't-dim', 12, 'middle'));
      lBottom = lY + 40;
    }
    edge('prelay', `M315 ${eY + eH} C315 ${eY + eH + 22} 400 ${lY - 22} 400 ${lY}`);
    edge('declay', `M685 ${eY + eH} C685 ${eY + eH + 22} 600 ${lY - 22} 600 ${lY}`, true);
    // --- kernels
    y = lBottom + 44;
    lane(y - 6, `Kernels (${ops.length})`);
    const kY = y + 20;
    const cols = 4, kw = 228, kh = 60, kg = 16;
    const kx0 = (W - (cols * kw + (cols - 1) * kg)) / 2;
    ops.forEach((o, i) => {
      const x = kx0 + (i % cols) * (kw + kg), yy = kY + Math.floor(i / cols) * (kh + kg);
      const grp = svg('g', { class: 'kern' + (o.admitted === false ? ' off' : '') });
      const rect = node(x, yy, kw, kh);
      const share = txt(x + kw - 10, yy + 20, '', 'mono', 12, 'end');
      const barBg = svg('rect', { x: x + 10, y: yy + kh - 10, width: kw - 20, height: 4, rx: 2, fill: 'var(--heat-0)' });
      const bar = svg('rect', { x: x + 10, y: yy + kh - 10, width: 0, height: 4, rx: 2, fill: 'var(--heat-1)' });
      grp.append(rect, txt(x + 10, yy + 20, o.op, 'mono', 12.5), share,
        txt(x + 10, yy + 35, trunc(o.kernel || '-', 34), 't-dim mono', 10.5),
        txt(x + 10, yy + 47, `${o.backend || '-'} · ${o.exactness || '-'}`, 't-faint badge mono', 9.5), barBg, bar);
      attachTip(grp, () => {
        const st = S.snap && S.snap.ops && S.snap.ops[o.op];
        const k = S.kern[o.op];
        return { title: o.op, rows: [['kernel', o.kernel || '-'], ['backend', o.backend || '-'], ['exactness', o.exactness || '-'],
          ['admitted', o.admitted === false ? 'no' : 'yes'], ['calls / decode step', o.calls_per_decode_step == null ? '-' : o.calls_per_decode_step],
          ['calls total', st ? fmtInt(st.calls) : '-'], ['measured time', st ? fmtMs(st.time_ms) + ' (' + fmtInt(st.timed_calls) + ' timed)' : '-'],
          ['time share', k ? (k.share * 100).toFixed(1) + '%' : '-'], ['call rate', k ? fmtNum(k.rate) + '/s' : '-']] };
      });
      g.append(grp);
      S.kern[o.op] = { rect, shareEl: share, bar, bw: kw - 20, share: 0, rate: 0 };
    });
    const kRows = Math.max(1, Math.ceil(ops.length / cols));
    const kBottom = kY + kRows * (kh + kg) - kg;
    edge('laykern', `M500 ${lBottom - (layers.length ? 22 : 0)} L500 ${kY - 18} M${kx0 + kw / 2} ${kY - 18} L${kx0 + 3 * (kw + kg) + kw / 2} ${kY - 18}` +
      Array.from({ length: Math.min(cols, ops.length) }, (_, c) => ` M${kx0 + c * (kw + kg) + kw / 2} ${kY - 18} L${kx0 + c * (kw + kg) + kw / 2} ${kY}`).join(''));
    // --- device
    y = kBottom + 44;
    lane(y - 6, 'Device');
    const dY = y + 20;
    edge('kerndev', `M500 ${kBottom} L500 ${dY - 4}`, true);
    g.append(node(60, dY, 880, 74));
    S.devTitle = txt(76, dY + 20, '', '', 12.5);
    S.devSub = txt(924, dY + 20, '', 't-dim mono', 11, 'end');
    g.append(S.devTitle, S.devSub);
    const bx = 76, bw = 848, by = dY + 30, bh = 18;
    g.append(svg('rect', { x: bx, y: by, width: bw, height: bh, rx: 4, fill: 'var(--heat-0)' }));
    const segs = [['weights', 'var(--accent)'], ['KV used', 'var(--good)'], ['KV reserved', 'color-mix(in srgb, var(--good) 35%, var(--heat-0))'], ['other', 'var(--warn)']];
    S.mem = { bx, bw, by, bh, segs: segs.map(([n, c]) => { const r = svg('rect', { class: 'memseg', x: bx, y: by, width: 0, height: bh, fill: c }); g.append(r); return { n, r }; }) };
    segs.forEach(([n, c], i) => {
      g.append(svg('rect', { x: bx + i * 130, y: by + 28, width: 9, height: 9, rx: 2, fill: c }), txt(bx + i * 130 + 14, by + 36, n, 't-dim', 10.5));
    });
    S.memVals = [];
    const memHit = svg('rect', { x: bx, y: by, width: bw, height: bh, fill: 'transparent' });
    attachTip(memHit, () => ({ title: 'device memory', rows: S.memVals }));
    g.append(memHit);
    const H = dY + 74 + 16;

    const root = svg('svg', { id: 'arch-svg', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Engine architecture map' }, g);
    holder.replaceChildren(root);
    attachTip(S.nodes.queue, () => ({ title: 'request queue', rows: [['running', S.snap ? S.snap.running : '-'], ['waiting', S.snap ? S.snap.waiting : '-'], ['requests total', S.snap ? fmtInt(S.snap.requests_total) : '-'], ['note', 'per-API split is not exported']] }));
    attachTip(S.nodes.prefill, () => ({ title: 'prefill', rows: [['rate', fmtNum(bus.rates.prefill) + ' tok/s'], ['last request', S.snap ? fmtNum(S.snap.last_prefill_tok_s) + ' tok/s' : '-'], ['prompt tokens', S.snap ? fmtInt(S.snap.prompt_tokens_total) : '-'], ['cached', S.snap ? fmtInt(S.snap.cached_tokens_total) : '-']] }));
    attachTip(S.nodes.decode, () => ({ title: 'decode', rows: [['rate', fmtNum(bus.rates.gen) + ' tok/s'], ['last request', S.snap ? fmtNum(S.snap.last_decode_tok_s) + ' tok/s' : '-'], ['generated', S.snap ? fmtInt(S.snap.generation_tokens_total) : '-'], ['last TTFT', S.snap ? fmtMs(S.snap.last_ttft_ms) : '-']] }));
  }

  function header(plan, snap) {
    const bits = [];
    const model = (plan && plan.model) || (snap && snap.model);
    head.replaceChildren(el('h2', { text: model || 'Architecture' }));
    const dev = (plan && plan.device && plan.device.name) || (snap && snap.device && snap.device.name);
    const add = (k, v) => v && head.append(el('span', { class: 'dim' }, k + ' ', el('b', { class: 'mono', style: 'color:var(--text);font-weight:500', text: v })));
    add('backend', (plan && plan.backend) || (snap && snap.device && snap.device.backend));
    add('device', dev);
    add('exactness', plan && plan.exactness);
    head.append(el('span', { class: 'dim' }, 'kernel time ', el('span', { class: 'heatbar' }), ' share'));
    note.textContent = plan ? '' : (bus.plan === undefined ? '' : 'The engine returned no plan (GET /v1/intarsia/plan is null): showing a static diagram built from telemetry op names.');
    foot.textContent = 'Hover any node for details. Edge animation speed follows live token and step rates; kernel heat is the share of measured op time.';
    void bits;
  }

  function colorMix(share, max) {
    const t = max > 0 ? clamp(share / max, 0, 1) : 0;
    return `color-mix(in srgb, ${t > 0.55 ? 'var(--heat-2)' : 'var(--heat-1)'} ${Math.round(t * 55)}%, var(--panel-2))`;
  }

  function update() {
    const snap = bus.snap, plan = bus.plan;
    if (plan === undefined || (!snap && plan === null)) { if (!S) { holder.replaceChildren(el('div', { class: 'empty', style: 'padding:24px', text: 'Waiting for the engine...' })); } return; }
    const k = JSON.stringify([plan, snap ? Object.keys(snap.ops || {}) : null]);
    if (k !== key) { key = k; S = null; build(plan, snap); header(plan, snap); }
    if (!S || !snap) return;
    S.snap = snap;
    const r = bus.rates;
    const active = snap.running + snap.waiting > 0;
    running = snap.running > 0 && r.steps > 0.05;
    S.steps = r.steps;
    const setFlow = (name, rate, on) => {
      const f = S.flows[name]; if (!f) return;
      const d = on ? flowDur(rate) : null;
      f.classList.toggle('on', d != null);
      if (d != null) f.style.animationDuration = d.toFixed(3) + 's';
    };
    const reqRate = active ? 1 + snap.waiting : 0;
    setFlow('api0', reqRate, active); setFlow('api1', reqRate, active);
    setFlow('qpre', r.prefill, true); setFlow('prelay', r.prefill, true);
    setFlow('qdec', r.gen, true); setFlow('declay', r.gen, true);
    setFlow('laykern', r.steps * 20, true); setFlow('kerndev', r.steps * 20, true);
    S.nodes.queue.classList.toggle('live', active);
    S.nodes.prefill.classList.toggle('live', r.prefill > 0.01);
    S.nodes.decode.classList.toggle('live', r.gen > 0.01);
    S.qText.textContent = `running ${snap.running} · waiting ${snap.waiting}`;
    S.preText.textContent = `${fmtNum(r.prefill)} tok/s`;
    S.decText.textContent = `${r.est ? '~' : ''}${fmtNum(r.gen)} tok/s`;
    S.stepText.textContent = fmtInt(snap.steps_total);
    S.stepSub.textContent = `steps · ${fmtNum(r.steps)}/s`;
    // kernels
    const entries = Object.entries(snap.ops || {});
    const tsum = entries.reduce((a, [, v]) => a + v.time_ms, 0);
    const csum = entries.reduce((a, [, v]) => a + v.calls, 0);
    let max = 0;
    const shares = {};
    for (const [n, v] of entries) { shares[n] = tsum > 0 ? v.time_ms / tsum : csum ? v.calls / csum : 0; max = Math.max(max, shares[n]); }
    for (const [n, ks] of Object.entries(S.kern)) {
      const sh = shares[n] || 0;
      ks.share = sh;
      ks.rate = (r.ops && r.ops[n]) || 0;
      ks.rect.style.fill = colorMix(sh, max);
      ks.bar.setAttribute('width', (ks.bw * sh).toFixed(1));
      ks.bar.setAttribute('fill', max && sh / max > 0.55 ? 'var(--heat-2)' : 'var(--heat-1)');
      ks.shareEl.textContent = (sh * 100).toFixed(1) + '%';
    }
    // device memory
    const d = snap.device || {};
    const total = d.total_mem_bytes || (bus.plan && bus.plan.device && bus.plan.device.mem_total) || 0;
    const other = Math.max(0, total - d.free_mem_bytes - snap.weights_bytes - snap.kv_total_bytes);
    const parts = [snap.weights_bytes, snap.kv_used_bytes, Math.max(0, snap.kv_total_bytes - snap.kv_used_bytes), other];
    let x = S.mem.bx;
    parts.forEach((v, i) => {
      const w = total ? (v / total) * S.mem.bw : 0;
      S.mem.segs[i].r.setAttribute('x', x.toFixed(1)); S.mem.segs[i].r.setAttribute('width', w.toFixed(1));
      x += w;
    });
    const kvf = snap.kv_total_bytes ? snap.kv_used_bytes / snap.kv_total_bytes : 0;
    S.devTitle.textContent = `${d.name || 'device'} · ${d.backend || ''}${d.arch ? ' ' + d.arch : ''}${d.unified ? ' · unified memory' : ''}`;
    S.devSub.textContent = total ? `${fmtBytes(total - d.free_mem_bytes)} used / ${fmtBytes(total)} · KV ${(kvf * 100).toFixed(1)}% full` : `KV ${(kvf * 100).toFixed(1)}% full · device memory not reported`;
    S.memVals = [['total', fmtBytes(total)], ['free', fmtBytes(d.free_mem_bytes)], ['weights', fmtBytes(snap.weights_bytes)],
      ['KV used', fmtBytes(snap.kv_used_bytes)], ['KV reserved', fmtBytes(snap.kv_total_bytes)], ['other', fmtBytes(other)]];
  }

  function loop(now) {
    raf = 0;
    if (!S || root.hidden) return;
    if (S.sweep && S.cells.length) {
      if (running) {
        const period = clamp(1 / Math.max(S.steps, 0.1), 0.35, 3);
        const i = Math.min(S.cells.length - 1, Math.floor(((now / 1000) % period) / period * S.cells.length));
        const c = S.cells[i];
        S.sweepIdx = i;
        S.sweep.setAttribute('x', c.x); S.sweep.setAttribute('y', c.y);
        S.sweep.classList.add('on');
      } else { S.sweep.classList.remove('on'); S.sweepIdx = -1; }
    }
    raf = requestAnimationFrame(loop);
  }
  const kick = () => { if (!raf && !root.hidden) raf = requestAnimationFrame(loop); };
  bus.on(() => { update(); kick(); });
  update();
  return { shown() { update(); kick(); }, hidden() { tip.hidden = true; } };
}
