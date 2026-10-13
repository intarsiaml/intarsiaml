import { Bus, el, save, store } from './util.js';
import { mountChat } from './chat.js';
import { mountDashboard } from './dashboard.js';
import { mountArch } from './arch.js';

const bus = new Bus();
const views = {
  chat: { root: document.getElementById('view-chat'), mount: mountChat },
  dashboard: { root: document.getElementById('view-dashboard'), mount: mountDashboard },
  arch: { root: document.getElementById('view-arch'), mount: mountArch },
};
const mounted = {};

function show(name) {
  if (!views[name]) name = 'chat';
  for (const [k, v] of Object.entries(views)) v.root.hidden = k !== name;
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  if (!mounted[name]) mounted[name] = views[name].mount(views[name].root, bus);
  if (mounted[name] && mounted[name].shown) mounted[name].shown();
  for (const [k, m] of Object.entries(mounted)) if (k !== name && m && m.hidden) m.hidden();
  save('tab', name);
  if (location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name);
}

document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));
window.addEventListener('hashchange', () => show(location.hash.slice(1)));

document.getElementById('theme').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('intarsia.theme', next); } catch (e) { /* ignore */ }
});

const conn = document.getElementById('conn');
const modelName = document.getElementById('model-name');
bus.on((b) => {
  conn.className = 'pill ' + (b.connected ? 'on' : 'off');
  conn.lastChild.textContent = b.connected ? 'live' : 'offline';
  if (b.snap && modelName.textContent !== b.snap.model) modelName.textContent = b.snap.model;
});

bus.start();
show(location.hash.slice(1) || store('tab', 'chat'));
