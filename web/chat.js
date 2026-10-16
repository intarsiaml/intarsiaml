import { el, store, save, fmtNum, fmtMs, fmtInt } from './util.js';

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function inline(s) {
  return esc(s)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>');
}

/** Minimal, escaping markdown: code fences, inline code, bold/italic, paragraphs. */
export function md(text) {
  return text.split('```').map((p, i) => {
    if (i % 2) {
      const nl = p.indexOf('\n');
      return '<pre><code>' + esc(nl >= 0 ? p.slice(nl + 1) : p) + '</code></pre>';
    }
    return p.split(/\n{2,}/).filter((x) => x.trim()).map((x) => '<p>' + inline(x.trim()).replace(/\n/g, '<br>') + '</p>').join('');
  }).join('');
}

/** Split model output into {think, answer, thinking}. Handles a template that pre-opens <think>. */
export function splitThink(t) {
  const open = t.indexOf('<think>');
  const close = t.indexOf('</think>');
  if (open === -1 && close === -1) return { think: '', answer: t, thinking: false };
  if (close === -1) return { think: t.slice(open + 7), answer: t.slice(0, open), thinking: true };
  const start = open === -1 || open > close ? 0 : open + 7;
  const pre = open > 0 && open < close ? t.slice(0, open) : '';
  return { think: t.slice(start, close), answer: pre + t.slice(close + 8), thinking: false };
}

export function mountChat(root, bus) {
  const settings = store('chat', { temperature: 0.7, max_tokens: '', system: '', model: '' });
  const history = []; // {role, content}
  let busy = null; // AbortController

  const modelSel = el('select', { 'aria-label': 'Model' });
  const temp = el('input', { type: 'number', min: '0', max: '2', step: '0.1', value: settings.temperature });
  const maxTok = el('input', { type: 'number', min: '1', step: '1', placeholder: 'default', value: settings.max_tokens });
  const sys = el('input', { type: 'text', class: 'grow', placeholder: 'System prompt (optional)', value: settings.system, 'aria-label': 'System prompt' });
  const clear = el('button', { class: 'btn sec', style: 'height:32px;padding:0 12px', text: 'Clear' });
  const bar = el('div', { class: 'chat-bar' },
    el('label', {}, 'Model', modelSel), el('label', {}, 'Temperature', temp), el('label', {}, 'Max tokens', maxTok), sys, clear);
  const log = el('div', { class: 'chat-log', 'aria-live': 'polite' });
  const hello = el('div', { class: 'hello' }, el('h2', { text: 'Chat with your local model' }),
    el('div', { text: 'Streams from /v1/chat/completions. Reasoning in <think> blocks is collapsed.' }));
  log.append(hello);
  const input = el('textarea', { rows: '1', placeholder: 'Message (Enter to send, Shift+Enter for newline)', 'aria-label': 'Message' });
  const send = el('button', { class: 'btn', text: 'Send' });
  root.append(bar, log, el('div', { class: 'composer' }, input, send));

  const persist = () => {
    Object.assign(settings, { temperature: temp.value, max_tokens: maxTok.value, system: sys.value, model: modelSel.value });
    save('chat', settings);
  };
  [temp, maxTok, sys, modelSel].forEach((e) => e.addEventListener('change', persist));

  fetch('/v1/models').then((r) => r.json()).then((j) => {
    for (const m of j.data || []) modelSel.append(el('option', { value: m.id, text: m.id }));
    if (settings.model && [...modelSel.options].some((o) => o.value === settings.model)) modelSel.value = settings.model;
  }).catch(() => modelSel.append(el('option', { value: '', text: '(unavailable)' })));

  const autosize = () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 200) + 'px'; };
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); go(); }
  });
  send.addEventListener('click', () => (busy ? busy.abort() : go()));
  clear.addEventListener('click', () => { if (busy) busy.abort(); history.length = 0; log.replaceChildren(hello); });

  function scrollDown() { log.scrollTop = log.scrollHeight; }

  function addMsg(role, text) {
    hello.remove();
    const body = el('div', { class: 'body' });
    const stats = el('div', { class: 'stats' });
    const m = el('div', { class: 'msg ' + role }, el('div', { class: 'role', text: role === 'user' ? 'You' : 'Assistant' }), body, stats);
    log.append(m);
    if (text != null) body.textContent = text;
    scrollDown();
    return { m, body, stats };
  }

  async function go() {
    const text = input.value.trim();
    if (!text || busy) return;
    persist();
    input.value = ''; autosize();
    history.push({ role: 'user', content: text });
    addMsg('user', text);
    const a = addMsg('assistant');
    const thinkEl = el('details', { class: 'think' }, el('summary', { text: 'Thinking' }), el('div', { class: 'tbody' }));
    const ansEl = el('div');
    a.body.append(thinkEl, ansEl);
    thinkEl.hidden = true;
    a.body.classList.add('cursor');
    let userToggled = false;
    thinkEl.addEventListener('toggle', () => { if (!thinkEl._auto) userToggled = true; });
    const setOpen = (o) => { thinkEl._auto = true; thinkEl.open = o; setTimeout(() => { thinkEl._auto = false; }, 0); };

    const messages = [];
    if (sys.value.trim()) messages.push({ role: 'system', content: sys.value.trim() });
    messages.push(...history);
    const req = { model: modelSel.value || undefined, messages, stream: true, stream_options: { include_usage: true },
      temperature: parseFloat(temp.value) };
    if (isNaN(req.temperature)) delete req.temperature;
    if (maxTok.value) req.max_tokens = parseInt(maxTok.value, 10);

    const ctl = new AbortController();
    busy = ctl;
    send.textContent = 'Stop'; send.classList.add('stop');
    const t0 = performance.now();
    let tFirst = 0, tLast = 0, full = '', usage = null, finish = '', deltas = 0, raf = 0, wasThinking = false;

    const paint = () => {
      raf = 0;
      const s = splitThink(full);
      if (s.think || s.thinking) {
        if (thinkEl.hidden) { thinkEl.hidden = false; setOpen(true); }
        thinkEl.querySelector('summary').textContent = s.thinking ? 'Thinking...' : 'Thought';
        thinkEl.querySelector('.tbody').textContent = s.think.trim();
        if (s.thinking) wasThinking = true;
        else if (wasThinking && !userToggled) { setOpen(false); wasThinking = false; }
      }
      ansEl.innerHTML = md(s.answer);
      scrollDown();
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(paint); };

    try {
      const r = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(req), signal: ctl.signal });
      if (!r.ok) {
        let msg = r.status + ' ' + r.statusText;
        try { const j = await r.json(); msg = (j.error && j.error.message) || msg; } catch (e) { /* not json */ }
        throw new Error(msg);
      }
      const rd = r.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const evt = buf.slice(0, i); buf = buf.slice(i + 2);
          for (const line of evt.split('\n')) {
            if (!line.startsWith('data:')) continue;
            const d = line.slice(5).trim();
            if (d === '[DONE]') continue;
            let j; try { j = JSON.parse(d); } catch (e) { continue; }
            if (j.error) throw new Error(j.error.message || 'stream error');
            if (j.usage) usage = j.usage;
            const ch = j.choices && j.choices[0];
            if (!ch) continue;
            if (ch.finish_reason) finish = ch.finish_reason;
            const c = ch.delta && ch.delta.content;
            if (c) { if (!tFirst) tFirst = performance.now(); tLast = performance.now(); full += c; deltas++; schedule(); }
          }
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') finish = 'cancelled';
      else { a.m.classList.add('err'); full += (full ? '\n\n' : '') + 'Error: ' + e.message; }
    } finally {
      busy = null;
      send.textContent = 'Send'; send.classList.remove('stop');
      a.body.classList.remove('cursor');
      if (raf) cancelAnimationFrame(raf);
      paint();
      if (full) history.push({ role: 'assistant', content: splitThink(full).answer || full });
      else history.pop(); // nothing came back: drop the unanswered user turn
      const comp = usage ? usage.completion_tokens : deltas;
      const gen = tFirst && tLast > tFirst ? (Math.max(comp - 1, 1) / ((tLast - tFirst) / 1000)) : null;
      const bits = [];
      if (tFirst) bits.push(['TTFT', fmtMs(tFirst - t0)]);
      if (gen != null) bits.push(['decode', fmtNum(gen) + ' tok/s']);
      if (usage) {
        bits.push(['prompt', fmtInt(usage.prompt_tokens) + ' tok']);
        const cached = usage.prompt_tokens_details && usage.prompt_tokens_details.cached_tokens;
        if (cached) bits.push(['cached', fmtInt(cached)]);
        bits.push(['completion', fmtInt(usage.completion_tokens) + ' tok']);
      }
      if (finish) bits.push(['finish', finish]);
      for (const [k, v] of bits) a.stats.append(el('span', {}, k + ' ', el('b', { class: 'mono', text: v })));
    }
  }
  return { shown() { setTimeout(() => input.focus(), 0); scrollDown(); } };
}
