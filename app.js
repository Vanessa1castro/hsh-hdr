(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const api = async (url, opts = {}) => {
    const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
    if (r.status === 401) { location.href = '/login'; throw new Error('signed out'); }
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `Error ${r.status}`);
    return j;
  };
  const ago = t => {
    const s = Date.now() / 1000 - t;
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(t * 1000).toLocaleDateString();
  };
  const pct = j => {
    if (j.status === 'done') return 100;
    if (j.status === 'saving' && j.photo_count) return 90 + 10 * (j.saved_count / j.photo_count);
    if (j.sets_total) return 8 + 82 * (j.sets_done / j.sets_total);
    return j.status === 'downloading' ? 4 : 2;
  };

  // ------------------------------------------------------------ tabs
  $$('.tab').forEach(t => t.onclick = () => {
    $$('.tab').forEach(x => x.classList.toggle('active', x === t));
    $$('.tab-body').forEach(b => b.classList.toggle('hidden', b.dataset.body !== t.dataset.tab));
  });

  // ------------------------------------------------------------ status / jobs list
  let status = null;
  async function refresh() {
    try { status = await api('/api/status'); } catch { return; }
    const pill = $('#dbx-pill');
    if (!status.dropbox_configured) {
      pill.className = 'pill'; pill.textContent = 'Dropbox not set up'; pill.onclick = null;
    } else if (status.dropbox_connected) {
      if (pill.dataset.armed) { renderJobs(status.jobs); if (current) loadJob(current, true); return; }
      pill.className = 'pill ok'; pill.textContent = 'Dropbox connected';
      pill.title = status.dropbox_account + ' · click to disconnect';
      pill.onclick = async () => {
        if (pill.dataset.armed) { await api('/api/dropbox/disconnect', { method: 'POST' }); delete pill.dataset.armed; refresh(); return; }
        pill.dataset.armed = '1'; pill.textContent = 'Click again to disconnect';
        setTimeout(() => { delete pill.dataset.armed; refresh(); }, 4000);
      };
    } else {
      pill.className = 'pill action'; pill.textContent = 'Connect Dropbox';
      pill.onclick = () => { location.href = '/dropbox/connect'; };
    }
    renderJobs(status.jobs);
    if (current) loadJob(current, true);
  }

  function renderJobs(jobs) {
    const el = $('#jobs');
    if (!jobs.length) { el.innerHTML = '<p class="muted">No edits yet. Paste a Dropbox link above to start.</p>'; return; }
    el.innerHTML = jobs.map(j => {
      const thumb = j.cover ? `style="background-image:url('/jobs/${j.id}/after/${encodeURIComponent(j.cover)}.jpg')"` : '';
      const progress = ['done', 'error'].includes(j.status) ? '' : `<div class="progress"><div style="width:${pct(j)}%"></div></div>`;
      const line = j.status === 'done' ? `${j.photo_count} photos` : (j.message || j.status_label);
      return `<a class="job-card" href="#/job/${j.id}">
        <div class="thumb" ${thumb}>${j.cover ? '' : esc(j.status_label)}</div>
        <div class="info"><h3>${esc(j.name)}</h3>
          <div class="status ${j.status}">${esc(line)} · ${ago(j.created)}</div>${progress}</div></a>`;
    }).join('');
  }

  // ------------------------------------------------------------ new edit: dropbox
  $('#start-dbx').onclick = async () => {
    const msg = $('#dbx-msg'); msg.className = 'msg'; msg.textContent = 'Checking the link…';
    $('#start-dbx').disabled = true;
    try {
      const j = await api('/api/jobs', { method: 'POST', body: JSON.stringify({
        source: 'dropbox', link: $('#link').value, name: $('#name').value,
        per_set: +$('#per-set').value, save_to_dropbox: $('#save-dbx').checked }) });
      msg.textContent = 'Started.'; $('#link').value = ''; $('#name').value = '';
      location.hash = `#/job/${j.id}`;
    } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
    $('#start-dbx').disabled = false;
  };

  // ------------------------------------------------------------ new edit: upload
  let picked = [];
  const RAW = /\.(rw2|cr2|cr3|nef|nrw|arw|srf|sr2|raf|orf|dng|pef|rwl|3fr|iiq|erf|mef|mos|srw|x3f)$/i;
  const setFiles = list => {
    picked = [...list].filter(f => RAW.test(f.name));
    const gb = picked.reduce((a, f) => a + f.size, 0) / 1e9;
    $('#up-msg').className = 'msg';
    $('#up-msg').textContent = picked.length ? `${picked.length} RAW files (${gb.toFixed(1)} GB)` : 'No RAW files in that selection';
    $('#start-up').disabled = !picked.length;
  };
  $('#pick').onclick = e => { e.preventDefault(); $('#files').click(); };
  $('#files').onchange = e => setFiles(e.target.files);
  const drop = $('#drop');
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); setFiles(e.dataTransfer.files); };

  $('#start-up').onclick = async () => {
    const btn = $('#start-up'), msg = $('#up-msg'), bar = $('#up-progress');
    btn.disabled = true; bar.classList.remove('hidden');
    try {
      const job = await api('/api/jobs', { method: 'POST', body: JSON.stringify({
        source: 'upload', name: $('#up-name').value, per_set: +$('#up-per-set').value }) });
      const total = picked.reduce((a, f) => a + f.size, 0); let sent = 0;
      for (let i = 0; i < picked.length; i++) {
        const fd = new FormData(); fd.append('files', picked[i]);
        msg.textContent = `Uploading ${i + 1} of ${picked.length}…`;
        const r = await fetch(`/api/jobs/${job.id}/upload`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(`Upload failed on ${picked[i].name}`);
        sent += picked[i].size; $('div', bar).style.width = `${100 * sent / total}%`;
      }
      await api(`/api/jobs/${job.id}/start`, { method: 'POST' });
      msg.textContent = 'Uploaded. Editing has started.'; picked = [];
      location.hash = `#/job/${job.id}`;
    } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; btn.disabled = false; }
    bar.classList.add('hidden'); $('div', bar).style.width = '0';
  };

  // ------------------------------------------------------------ job page
  let current = null, shown = new Set();
  function baHTML(jid, name) {
    const n = encodeURIComponent(name);
    return `<img class="after" src="/jobs/${jid}/after/${n}.jpg" alt="Edited ${esc(name)}" loading="lazy">
      <img class="before" src="/jobs/${jid}/before/${n}.jpg" alt="Original ${esc(name)}" loading="lazy">
      <div class="handle"></div><span class="tag l">Before</span><span class="tag r">After</span>`;
  }
  function wireBA(el) {
    const set = x => {
      const r = el.getBoundingClientRect();
      const p = Math.max(0, Math.min(100, 100 * (x - r.left) / r.width));
      $('.before', el).style.clipPath = `inset(0 ${100 - p}% 0 0)`;
      $('.handle', el).style.left = `${p}%`;
    };
    let down = false;
    el.addEventListener('pointerdown', e => { if (e.target.closest('.zoom')) return; down = true; el.setPointerCapture(e.pointerId); set(e.clientX); });
    el.addEventListener('pointermove', e => { if (down) set(e.clientX); });
    el.addEventListener('pointerup', () => { down = false; });
  }

  async function loadJob(jid, quiet) {
    let j;
    try { j = await api(`/api/jobs/${jid}`); } catch (e) { if (!quiet) location.hash = '#/'; return; }
    if (current !== jid) return;
    $('#job-name').textContent = j.name;
    const st = j.status === 'done' ? `${j.photo_count} photos · from ${j.raw_count} RAWs` :
      j.status === 'error' ? `Stopped: ${j.error}` : (j.message || j.status_label);
    $('#job-status').textContent = st;
    $('#job-status').className = j.status === 'error' ? 'error' : 'muted';
    $('#job-progress').classList.toggle('hidden', ['done', 'error'].includes(j.status));
    $('#job-progress div').style.width = `${pct(j)}%`;
    $('#job-zip').classList.toggle('hidden', !j.photo_count);
    $('#job-zip').href = `/jobs/${j.id}/download.zip`;
    $('#job-retry').classList.toggle('hidden', j.status !== 'error');
    $('#job-dest').textContent = j.dropbox_dest ? (j.status === 'done' ? `Saved to Dropbox: ${j.dropbox_dest}` : `Will be saved to Dropbox: ${j.dropbox_dest}`) : '';
    const failed = $('#job-failed');
    failed.classList.toggle('hidden', !j.failed.length);
    failed.textContent = j.failed.length ? `Could not edit: ${j.failed.map(f => f.name).join(', ')}` : '';
    const gal = $('#gallery');
    for (const p of j.photos) {
      if (shown.has(p.name)) continue;
      shown.add(p.name);
      const card = document.createElement('div');
      card.className = 'photo';
      card.innerHTML = `<div class="ba">${baHTML(j.id, p.name)}<button class="zoom">Enlarge</button></div>
        <div class="meta"><span class="muted">${esc(p.name)} · ${p.frames.length} bracket${p.frames.length === 1 ? '' : 's'}</span>
        <a href="/jobs/${j.id}/full/${encodeURIComponent(p.name)}.jpg">Download</a></div>`;
      gal.appendChild(card);
      wireBA($('.ba', card));
      $('.zoom', card).onclick = e => { e.stopPropagation(); openViewer(j.id, p.name); };
    }
  }

  function openViewer(jid, name) {
    const v = $('#viewer'), ba = $('#viewer-ba');
    ba.innerHTML = baHTML(jid, name).replaceAll(' loading="lazy"', '');
    $('.after', ba).src = `/jobs/${jid}/full/${encodeURIComponent(name)}.jpg`;  // full-size edit
    $('#viewer-cap').textContent = name;
    v.classList.remove('hidden'); wireBA(ba);
  }
  $('.viewer-close').onclick = () => $('#viewer').classList.add('hidden');
  document.addEventListener('keydown', e => { if (e.key === 'Escape') $('#viewer').classList.add('hidden'); });

  $('#job-retry').onclick = async () => { await api(`/api/jobs/${current}/retry`, { method: 'POST' }); loadJob(current); };
  let deleteArmed = false;
  $('#job-delete').onclick = async () => {
    const b = $('#job-delete');
    if (!deleteArmed) { deleteArmed = true; b.textContent = 'Click again to delete'; setTimeout(() => { deleteArmed = false; b.textContent = 'Delete'; }, 4000); return; }
    try { await api(`/api/jobs/${current}`, { method: 'DELETE' }); location.hash = '#/'; refresh(); }
    catch (e) { b.textContent = e.message; }
  };

  // ------------------------------------------------------------ routing
  function route() {
    const m = location.hash.match(/^#\/job\/([\w-]+)/);
    $('#viewer').classList.add('hidden');
    if (m) {
      if (current !== m[1]) { current = m[1]; shown = new Set(); $('#gallery').innerHTML = ''; }
      $('#home').classList.add('hidden'); $('#job').classList.remove('hidden');
      loadJob(current);
    } else {
      current = null;
      $('#job').classList.add('hidden'); $('#home').classList.remove('hidden');
    }
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', route);
  if (/dropbox=connected/.test(location.search)) history.replaceState(null, '', '/');
  route(); refresh(); setInterval(refresh, 4000);
})();
