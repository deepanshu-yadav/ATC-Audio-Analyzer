(() => {
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const EMPTY = { language: 'en', processing_time_seconds: 0, speech_duration_seconds: 0, vad_enabled: false, noise_removal_enabled: false, timestamp: '', device: 'auto', segments: [], intermediate_chunks: [], text: '', filtered_full_path_rel: null, denoised_full_path_rel: null };
const S = { d: { ...EMPTY }, t: 0, playing: false, lead: 0.12, active: -1, view: 'segments', chunk: 1, cvar: 'denoised', cPlaying: false, cLoading: false, fvar: 'denoised', hist: [], page: 1, pages: 1, loadedId: null, editing: null, dirty: false, saving: false };
const LIMIT = 5;
const audio = new Audio();
let audioCtx = null, chunkSrc = null, raf = null, simStart = 0, reg = {}, hudReg = [];

// ---------- ATC tagging + word timing ----------
const PATTERNS = [
  { re: /\b(cleared|maintain|contact|descend|climb|turn left|turn right|hold short|reduce speed)\b/gi, cls: 'tok-verb' },
  { re: /\brunway two\s(seven|two left|two)\b/gi, cls: 'tok-runway' },
  { re: /\b(one|two|three|four|five|six|seven|eight|niner|nine|zero)\s(hundred|thousand)\b/gi, cls: 'tok-alt' },
  { re: /\bheading\s(one|two|three|zero|niner|nine)?\s?[\w\s]{0,12}?(zero|one|two|three|four|five|six|seven|eight|niner|nine)\b/gi, cls: 'tok-heading' },
  { re: /\b(ils|dme|localizer|qnh|vfr)\b/gi, cls: 'tok-nav' },
];
function tagText(text) {
  const marks = new Array(text.length).fill(null);
  PATTERNS.forEach(({ re, cls }) => {
    const rx = new RegExp(re.source, re.flags); let m;
    while ((m = rx.exec(text)) !== null) {
      if (!m[0].length) { rx.lastIndex++; continue; }
      let clash = false;
      for (let i = m.index; i < m.index + m[0].length; i++) if (marks[i]) { clash = true; break; }
      if (!clash) for (let i = m.index; i < m.index + m[0].length; i++) marks[i] = cls;
    }
  });
  const nodes = []; let i = 0;
  while (i < text.length) {
    const cls = marks[i]; let j = i;
    while (j < text.length && marks[j] === cls) j++;
    nodes.push({ t: text.slice(i, j), cls }); i = j;
  }
  return nodes;
}
const fmtTime = s => { s = Number(s) || 0; return `${String(Math.floor(s / 60)).padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`; };
const wordsCache = new WeakMap();
function getWords(seg) {
  if (!seg || !seg.text) return [];
  if (wordsCache.has(seg)) return wordsCache.get(seg);
  let out = [];
  if (Array.isArray(seg.words) && seg.words.length) {
    out = seg.words.map(w => ({ text: w.word || w.text || '', start: typeof w.start === 'number' ? w.start : seg.start, end: typeof w.end === 'number' ? w.end : seg.end, cls: w.cls || null, sp: ' ' }));
  } else {
    const tokens = [];
    tagText(seg.text.trim()).forEach(n => { const rx = /(\S+)(\s*)/g; let m; while ((m = rx.exec(n.t))) tokens.push({ text: m[1], cls: n.cls, sp: m[2] || ' ' }); });
    if (tokens.length) {
      const s0 = typeof seg.start === 'number' ? seg.start : 0, s1 = typeof seg.end === 'number' ? seg.end : s0 + 1, dur = Math.max(0.08, s1 - s0);
      const wt = tokens.map(t => Math.max(2, t.text.replace(/[^a-zA-Z0-9]/g, '').length || 1)), tot = wt.reduce((a, b) => a + b, 0);
      let cur = 0;
      out = tokens.map((t, i) => { const a = s0 + cur / tot * dur; cur += wt[i]; return { text: t.text, start: +a.toFixed(3), end: +(s0 + cur / tot * dur).toFixed(3), cls: t.cls, sp: i === tokens.length - 1 ? '' : t.sp }; });
    }
  }
  wordsCache.set(seg, out); return out;
}

// ---------- word rendering + glow ----------
const eff = () => Math.max(0, S.t + S.lead);
const glowOn = (e, act) => { const p = eff(); return act && p >= e.w.start && (p < e.w.end || (e.i === e.n - 1 && p <= e.seg.end + 0.15)); };
function addWords(parent, seg, idx, list) {
  const ws = getWords(seg);
  ws.forEach((w, i) => {
    const el = document.createElement('span');
    el.className = 'atc-word ' + (w.cls || ''); el.textContent = w.text; el.title = `Jump to ${fmtTime(w.start)}`;
    el.onclick = ev => { ev.stopPropagation(); seekTo(w.start); };
    parent.append(el, w.sp); list.push({ el, w, i, n: ws.length, seg });
  });
}
const refreshGlow = (list, act) => list && list.forEach(e => e.el.classList.toggle('word-glowing', glowOn(e, act)));

// ---------- per-frame update ----------
function update(force) {
  const segs = S.d.segments || [], p = eff();
  const idx = segs.length ? segs.findIndex(s => p >= s.start && p <= s.end + 0.2) : -1;
  const changed = force || idx !== S.active, prev = S.active;
  S.active = idx;
  $('timecode').textContent = `${fmtTime(S.t)} / ${fmtTime(S.d.speech_duration_seconds)}`;
  drawWave();
  if (changed) {
    document.querySelectorAll('.seg.on').forEach(e => e.classList.remove('on'));
    const row = document.querySelector(`.seg[data-idx="${idx}"]`);
    if (row) {
      row.classList.add('on');
      const box = row.parentElement;
      box.scrollTo({ top: row.offsetTop - box.clientHeight / 2 + row.offsetHeight / 2, behavior: 'smooth' });
    }
    refreshGlow(reg[prev], false); buildHud();
  } else if (idx < 0) $('hud').firstChild.textContent = hudIdle();
  refreshGlow(reg[idx], true); refreshGlow(hudReg, true);
}
const hudIdle = () => (S.d.segments || []).length ? `[ Monitoring Frequency · Waiting for Next Transmission (${fmtTime(S.t)}) ]` : '[ Standby · No audio loaded. Upload an audio recording above or select from History ]';
function buildHud() {
  const hud = $('hud'); hud.innerHTML = ''; hudReg = [];
  const seg = (S.d.segments || [])[S.active];
  if (!seg) { const d = document.createElement('div'); d.className = 'hud-ph'; d.textContent = hudIdle(); hud.append(d); return; }
  const wrap = document.createElement('div'); wrap.className = 'hud-in';
  const tc = document.createElement('span'); tc.className = 'hud-tc'; tc.textContent = `[${fmtTime(seg.start)} - ${fmtTime(seg.end)}]`;
  const tx = document.createElement('span'); tx.className = 'hud-tx'; addWords(tx, seg, S.active, hudReg);
  wrap.append(tc, tx); hud.append(wrap);
}

// ---------- waveform ----------
const bars = Array.from({ length: 260 }, (_, i) => { const s = Math.sin(i * 12.9898) * 43758.5453; return Math.abs(s - Math.floor(s)) * 0.75 + 0.15; });
function drawWave() {
  const c = $('wave'), dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, w, h);
  const dur = S.d.speech_duration_seconds || 0;
  if (!(S.d.segments || []).length && !dur) {
    x.fillStyle = '#0a0e13'; x.fillRect(0, 0, w, h); x.strokeStyle = '#1a232d'; x.strokeRect(0, 0, w, h);
    x.fillStyle = '#4a5868'; x.font = '11px "JetBrains Mono", monospace'; x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText('NO AUDIO LOADED · UPLOAD AUDIO ABOVE OR SELECT FROM HISTORY', w / 2, h / 2); return;
  }
  const paint = () => bars.forEach((a, i) => { const bh = a * h * 0.8; x.fillRect(i / bars.length * w, (h - bh) / 2, w / bars.length - 1, bh); });
  x.fillStyle = '#1c2530'; paint();
  const pw = dur > 0 ? S.t / dur * w : 0;
  x.fillStyle = '#00ff9c'; x.globalAlpha = 0.85; x.fillRect(0, 0, pw, h);
  x.globalCompositeOperation = 'source-atop'; paint(); x.globalCompositeOperation = 'source-over'; x.globalAlpha = 1;
  x.strokeStyle = '#ffb000'; x.lineWidth = 2; x.beginPath(); x.moveTo(pw, 0); x.lineTo(pw, h); x.stroke();
}
$('wave').onclick = e => { const r = e.target.getBoundingClientRect(); seekTo(Math.max(0, Math.min(S.d.speech_duration_seconds || 0, (e.clientX - r.left) / r.width * (S.d.speech_duration_seconds || 0)))); };
window.addEventListener('resize', drawWave);

// ---------- main playback ----------
const setStatus = m => { $('status').textContent = m || ''; };
const audioUrl = () => { const p = S.fvar === 'denoised' ? S.d.denoised_full_path_rel : S.d.filtered_full_path_rel; return p ? '/' + p : ''; };
function syncSrc() {
  const u = audioUrl();
  if (u && (!audio.src || !audio.src.includes(u))) {
    const was = S.playing; if (was) audio.pause();
    audio.src = u; audio.load(); audio.currentTime = S.t;
    if (was) audio.play().catch(() => {});
  }
}
function setPlaying(v) {
  if (v === S.playing) return;
  S.playing = v; $('playBtn').textContent = v ? '⏸ pause' : '▶ play';
  cancelAnimationFrame(raf);
  if (v) { simStart = performance.now() - S.t * 1000; raf = requestAnimationFrame(tick); }
}
function tick() {
  if (!S.playing) return;
  const dur = S.d.speech_duration_seconds || 0;
  if (audio.src && !audio.paused && !audio.ended) { S.t = audio.currentTime; update(); }
  else if (audio.ended) { setPlaying(false); S.t = 0; update(); return; }
  else { const el = (performance.now() - simStart) / 1000; if (el >= dur) { S.t = dur; setPlaying(false); update(); return; } S.t = el; update(); }
  raf = requestAnimationFrame(tick);
}
function seekTo(t) { S.t = t; if (audio.src) audio.currentTime = t; simStart = performance.now() - t * 1000; update(); }
audio.addEventListener('play', () => setPlaying(true));
audio.addEventListener('pause', () => { if (!audio.ended) setPlaying(false); });
audio.addEventListener('ended', () => { setPlaying(false); S.t = 0; update(); });
$('playBtn').onclick = () => {
  stopChunk();
  if (!audioUrl() && !(S.d.segments || []).length) return setStatus('No audio loaded. Upload an audio file above or select an item from History to play.');
  syncSrc();
  if (S.playing) { audio.pause(); setPlaying(false); }
  else audio.play().then(() => setPlaying(true)).catch(e => { console.error('Playback failed, simulating:', e); setPlaying(true); });
};
$('lead').oninput = e => setLead(parseFloat(e.target.value));
document.querySelectorAll('[data-lead]').forEach(b => b.onclick = () => setLead(parseFloat(b.dataset.lead)));
function setLead(v) {
  S.lead = v; $('lead').value = v; $('leadBadge').textContent = `${v >= 0 ? '+' : ''}${Math.round(v * 1000)}ms`;
  document.querySelectorAll('[data-lead]').forEach(b => b.classList.toggle('on', parseFloat(b.dataset.lead) === v)); update();
}
document.querySelectorAll('[data-fvar]').forEach(b => b.onclick = () => { S.fvar = b.dataset.fvar; renderToggles(); syncSrc(); });
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => { S.view = b.dataset.view; renderToggles(); renderTranscript(); });
function renderToggles() {
  document.querySelectorAll('[data-fvar]').forEach(b => b.classList.toggle('on', b.dataset.fvar === S.fvar));
  document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('on', b.dataset.view === S.view));
}

// ---------- header / stats ----------
function renderStats() {
  const d = S.d, sp = d.processing_time_seconds && d.speech_duration_seconds ? (d.speech_duration_seconds / d.processing_time_seconds).toFixed(1) : '0';
  const chip = (l, v, a) => `<div class="chip ${a ? 'acc' : ''}"><span class="l">${l}</span><span class="v">${esc(v)}</span></div>`;
  $('headerStats').innerHTML = chip('device', (d.device || 'auto').toUpperCase(), d.device && d.device !== 'cpu' && d.device !== 'auto') + chip('lang', (d.language || 'en').toUpperCase()) + chip('vad', d.vad_enabled ? 'on' : 'off', d.vad_enabled) + chip('noise removal', d.noise_removal_enabled ? 'on' : 'off', d.noise_removal_enabled) + chip('speedup', sp + 'x', parseFloat(sp) > 0);
  const mini = (l, v) => `<div class="mini"><div class="l">${l}</div><div class="v">${esc(v)}</div></div>`;
  $('statsBar').innerHTML = mini('speech duration', fmtTime(d.speech_duration_seconds)) + mini('processing time', (d.processing_time_seconds || 0).toFixed(2) + 's') + mini('segments', (d.segments || []).length) + mini('run timestamp', d.timestamp || '—');
}

// ---------- transcript ----------
function renderTranscript() {
  const box = $('transcript'), segs = S.d.segments || []; box.innerHTML = ''; reg = {};
  const has = segs.length > 0;
  $('saveBtn').disabled = S.saving || !has; $('saveBtn').textContent = S.saving ? '💾 saving...' : '💾 save to database';
  $('saveBtn').classList.toggle('dirty', S.dirty); $('unsaved').hidden = !S.dirty;
  if (!has) {
    box.innerHTML = S.view === 'segments'
      ? '<div class="tempty"><div style="font-size:28px;margin-bottom:10px;opacity:.6">🎙️</div><div style="font-weight:600;color:#e8edf2;font-size:13px;margin-bottom:6px">No Transcriptions Loaded</div><div style="max-width:440px">Upload an audio file (WAV, MP3) using the form above to begin transcription, or select a previous session from the History list.</div></div>'
      : '<div class="tempty">No transcript text available. Upload an audio recording to begin.</div>';
  } else if (S.view === 'full') {
    const ft = document.createElement('div'); ft.className = 'full-text';
    segs.forEach((seg, i) => { const s = document.createElement('span'); s.style.marginRight = '6px'; reg[i] = []; addWords(s, seg, i, reg[i]); ft.append(s); });
    box.append(ft);
  } else {
    const list = document.createElement('div'); list.className = 'seg-list';
    segs.forEach((seg, i) => list.append(S.editing === i ? editCard(seg, i) : segRow(seg, i)));
    box.append(list);
  }
  update(true);
}
function segRow(seg, i) {
  const r = document.createElement('div'); r.className = 'seg'; r.dataset.idx = i; r.onclick = () => seekTo(seg.start);
  r.innerHTML = `<div class="lc"><span class="t1">${fmtTime(seg.start)}</span><span class="t2">– ${fmtTime(seg.end)}</span></div><div class="tx"></div><div class="ac"><button class="btn ebtn" title="Edit timestamp and text">✏️ edit</button><button class="btn dbtn" title="Delete this timestamp">🗑️ delete</button></div>`;
  reg[i] = []; addWords(r.querySelector('.tx'), seg, i, reg[i]);
  r.querySelector('.ac').onclick = e => e.stopPropagation();
  r.querySelector('.ebtn').onclick = () => { S.editing = i; renderTranscript(); };
  r.querySelector('.dbtn').onclick = () => deleteSeg(i);
  return r;
}
function editCard(seg, i) {
  const c = document.createElement('div'); c.className = 'edit';
  c.innerHTML = `<div class="h"><span>Edit Segment #${i + 1}</span><small>Original: ${fmtTime(seg.start)} – ${fmtTime(seg.end)}</small></div>
  <div class="cols"><div><label class="flabel">Start Time (sec)</label><input type="number" step="0.01" min="0" class="es" value="${seg.start}"><span class="pv ps"></span></div>
  <div><label class="flabel">End Time (sec)</label><input type="number" step="0.01" min="0" class="ee" value="${seg.end}"><span class="pv pe"></span></div></div>
  <div><label class="flabel">Transcript Text</label><textarea class="et" rows="2">${esc(seg.text)}</textarea></div>
  <div class="btns"><button class="btn apply">✓ Apply</button><button class="btn cancel">✕ Cancel</button></div>`;
  const es = c.querySelector('.es'), ee = c.querySelector('.ee');
  const pv = () => { c.querySelector('.ps').textContent = 'Display: ' + fmtTime(parseFloat(es.value) || 0); c.querySelector('.pe').textContent = 'Display: ' + fmtTime(parseFloat(ee.value) || 0); };
  es.oninput = ee.oninput = pv; pv();
  c.querySelector('.cancel').onclick = () => { S.editing = null; renderTranscript(); };
  c.querySelector('.apply').onclick = () => {
    const s = parseFloat(es.value), e = parseFloat(ee.value);
    if (isNaN(s) || isNaN(e) || s < 0 || e <= s) return alert('Invalid timestamps: Start must be >= 0 and End must be strictly greater than Start.');
    const segs = [...S.d.segments]; segs[i] = { ...segs[i], start: +s.toFixed(2), end: +e.toFixed(2), text: c.querySelector('.et').value.trim() };
    segs.sort((a, b) => a.start - b.start);
    S.d = { ...S.d, segments: segs, text: segs.map(x => x.text).join(' ') };
    S.editing = null; S.dirty = true; setStatus(`Segment #${i + 1} updated locally. Click 'Save to Database' to persist.`); renderStats(); renderTranscript();
  };
  return c;
}
function deleteSeg(i) {
  const seg = S.d.segments[i];
  if (!confirm(`Delete timestamp segment #${i + 1} [${fmtTime(seg.start)} - ${fmtTime(seg.end)}]?`)) return;
  const segs = S.d.segments.filter((_, k) => k !== i);
  S.d = { ...S.d, segments: segs, text: segs.map(s => s.text).join(' ') };
  if (S.editing === i) S.editing = null;
  S.dirty = true; setStatus(`Segment #${i + 1} deleted. Click 'Save to Database' to persist.`); renderStats(); renderTranscript();
}
$('saveBtn').onclick = async () => {
  const d = S.d; S.saving = true; renderTranscript();
  try {
    const payload = { id: d.id || S.loadedId, filename: d.filename || `atc_transcription_${d.timestamp || 'custom'}.wav`, segments: d.segments || [], text: d.text || (d.segments || []).map(s => s.text).join(' '), language: d.language || 'en', processing_time_seconds: d.processing_time_seconds || 0, speech_duration_seconds: d.speech_duration_seconds || ((d.segments || []).length ? Math.max(...d.segments.map(s => s.end)) : 0), vad_enabled: d.vad_enabled ?? true, noise_removal_enabled: d.noise_removal_enabled ?? true, device: d.device || 'cpu', timestamp: d.timestamp || new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 15), saved_transcription_file: d.saved_transcription_file || '', filtered_full_path_rel: d.filtered_full_path_rel || '', denoised_full_path_rel: d.denoised_full_path_rel || '', intermediate_chunks: d.intermediate_chunks || [] };
    const res = await fetch('/transcriptions/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error: ${res.status}`);
    const rec = await res.json(); S.d = { ...S.d, ...rec }; S.loadedId = rec.id; S.dirty = false;
    setStatus(`✓ Successfully saved transcription (ID: ${rec.id}) to database!`); fetchHistory(S.page);
  } catch (err) { console.error(err); setStatus(`Failed to save to database: ${err.message}`); }
  S.saving = false; renderTranscript(); renderHistory();
};

// ---------- chunks ----------
const chunks = () => S.d.intermediate_chunks || [];
function renderChunks() {
  const cs = chunks(), box = $('chunkBody'); $('chunkCount').textContent = `${cs.length} chunks`;
  if (!cs.length) { box.innerHTML = '<div style="font:11.5px \'JetBrains Mono\',monospace;color:#5f6f80;text-align:center;padding:52px 12px;line-height:1.6">Intermediate 30-second pipeline chunks (filtered &amp; denoised) will appear here after transcribing with "Save Chunks" enabled.</div>'; return; }
  const c = cs.find(x => x.chunk_num === S.chunk) || cs[0], col = S.cvar === 'denoised' ? '#00ff9c' : '#ffb000';
  box.innerHTML = `<div class="cc"><label class="flabel">chunk</label><select id="chSel">${cs.map(x => `<option value="${x.chunk_num}" ${x.chunk_num === S.chunk ? 'selected' : ''}>chunk ${x.chunk_num} · ${fmtTime(x.start_time)}–${fmtTime(x.end_time)}</option>`).join('')}</select>
  <label class="flabel">variant</label><select id="chVar"><option value="filtered">filtered</option><option value="denoised">denoised</option></select>
  <button class="btn play" id="chPlay" style="margin-top:4px" ${S.cLoading ? 'disabled' : ''}>${S.cLoading ? '⏳ loading...' : S.cPlaying ? '⏹ stop' : '▶ play chunk'}</button></div>
  <div class="cm"><div><span>window</span><b>${fmtTime(c.start_time)} → ${fmtTime(c.end_time)}</b></div><div><span>span</span><b>${(c.end_time - c.start_time).toFixed(0)}s</b></div><div><span>variant</span><b style="color:${col}">${S.cvar}</b></div></div>`;
  $('chVar').value = S.cvar;
  $('chSel').onchange = e => { stopChunk(); S.chunk = Number(e.target.value); renderChunks(); };
  $('chVar').onchange = e => { stopChunk(); S.cvar = e.target.value; renderChunks(); };
  $('chPlay').onclick = () => S.cPlaying ? stopChunk() : playChunk();
}
function stopChunk() { if (chunkSrc) { try { chunkSrc.stop(); } catch (e) {} } if (S.cPlaying) { S.cPlaying = false; renderChunks(); } }
async function playChunk() {
  if (S.playing) { audio.pause(); setPlaying(false); }
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  if (chunkSrc) { try { chunkSrc.stop(); } catch (e) {} }
  const c = chunks().find(x => x.chunk_num === S.chunk); if (!c) return;
  const rel = S.cvar === 'denoised' ? c.denoised_path_rel : c.filtered_path_rel;
  if (!rel) return setStatus('This chunk has no audio file for the selected variant.');
  S.cLoading = true; renderChunks();
  try {
    const res = await fetch('/' + rel); if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await audioCtx.decodeAudioData(await res.arrayBuffer());
    if (S.chunk !== c.chunk_num) { S.cLoading = false; return renderChunks(); }
    const src = audioCtx.createBufferSource(); src.buffer = buf; src.connect(audioCtx.destination);
    src.onended = () => { if (chunkSrc === src) { S.cPlaying = false; renderChunks(); } };
    src.start(); chunkSrc = src; S.cPlaying = true;
  } catch (err) { console.error('Chunk playback failed:', err); setStatus('Chunk playback failed: ' + err.message); }
  S.cLoading = false; renderChunks();
}

// ---------- history ----------
function renderHistory() {
  const box = $('historyList');
  box.innerHTML = S.hist.length ? '' : '<div class="empty">No previous transcriptions found.</div>';
  S.hist.forEach(it => {
    const r = document.createElement('div'); r.className = 'hrow' + (it.id === S.loadedId ? ' on' : '');
    r.innerHTML = `<div class="top"><span class="hname" title="${esc(it.filename)}">${esc(it.filename)}</span><div class="row" style="gap:6px"><span class="htime">${fmtTime(it.speech_duration_seconds)}</span><button class="btn hdel" title="Delete transcription from database">✕</button></div></div>
    <div class="bot"><span class="hmeta">${esc(it.created_at ? it.created_at.split(' ')[0] : it.timestamp || 'unknown')}</span><span class="hmeta">device: <span style="color:${it.device === 'cpu' ? '#ffb000' : '#00ff9c'}">${esc(it.device)}</span></span></div>`;
    r.onclick = () => loadHistoryItem(it.id);
    r.querySelector('.hdel').onclick = e => deleteHistory(it.id, e);
    box.append(r);
  });
  $('pager').hidden = S.pages <= 1; $('pageTxt').textContent = `${S.page} / ${S.pages}`;
  document.querySelector('[data-act=hprev]').disabled = S.page <= 1; document.querySelector('[data-act=hnext]').disabled = S.page >= S.pages;
}
document.querySelector('[data-act=hprev]').onclick = () => fetchHistory(S.page - 1);
document.querySelector('[data-act=hnext]').onclick = () => fetchHistory(S.page + 1);
async function fetchHistory(page = 1) {
  $('hLoad').hidden = false;
  try {
    const res = await fetch(`/transcriptions?page=${page}&limit=${LIMIT}`); if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    const j = await res.json(); S.hist = j.transcriptions || []; S.page = j.page || 1; S.pages = j.pages || 1;
  } catch (err) { console.error('Failed to fetch history:', err); }
  $('hLoad').hidden = true; renderHistory();
}
function applyDataset(d) {
  if (audio.src) audio.pause(); setPlaying(false); stopChunk();
  if (!d.text) d.text = (d.segments || []).map(s => s.text).join(' ');
  S.d = d; S.t = 0; S.editing = null; S.dirty = false; S.active = -1;
  if (chunks().length && !chunks().some(c => c.chunk_num === S.chunk)) S.chunk = chunks()[0].chunk_num;
  if (chunks().length) S.chunk = chunks()[0].chunk_num;
  renderStats(); renderTranscript(); renderChunks(); syncSrc(); renderHistory();
}
async function loadHistoryItem(id) {
  try {
    const res = await fetch(`/transcriptions/${id}`); if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    const d = await res.json(); if (!d || !d.segments) return;
    S.loadedId = d.id; applyDataset(d); setStatus(`Loaded transcription ID ${d.id}: ${d.filename}`);
  } catch (err) { console.error(err); setStatus(`Failed to load historical transcription: ${err.message}`); }
}
async function deleteHistory(id, e) {
  e.stopPropagation(); if (!confirm(`Permanently delete transcription ID ${id} from database?`)) return;
  try {
    const res = await fetch(`/transcriptions/${id}`, { method: 'DELETE' }); if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    setStatus(`Transcription ID ${id} deleted from database.`); if (S.loadedId === id) S.loadedId = null; fetchHistory(S.page);
  } catch (err) { console.error(err); setStatus(`Failed to delete transcription: ${err.message}`); }
}

// ---------- upload / transcribe ----------
const fileEl = $('file');
fileEl.onchange = () => { $('submitBtn').disabled = !fileEl.files[0]; };
$('noise').onchange = e => { $('nr').disabled = !e.target.checked; };
$('form').onsubmit = async e => {
  e.preventDefault(); const file = fileEl.files[0]; if (!file) return;
  $('submitBtn').disabled = true; $('submitBtn').textContent = 'Transcribing...'; $('spinner').hidden = false; $('error').textContent = '';
  setStatus('Uploading and processing audio file...'); if (S.playing) { audio.pause(); setPlaying(false); } stopChunk();
  const fd = new FormData();
  fd.append('file', file); fd.append('apply_noise_removal', $('noise').checked); fd.append('noise_reduction', $('nr').value);
  fd.append('language', 'en'); fd.append('save_intermediate', $('saveChunks').checked); fd.append('device', $('device').value);
  try {
    const res = await fetch('/transcribe', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error: ${res.status}`);
    const r = await res.json(); if (!r || !r.segments) throw new Error('Invalid response format received from transcription API.');
    if (!(r.intermediate_chunks || []).length) {
      r.intermediate_chunks = []; const dur = r.speech_duration_seconds;
      for (let i = 0; i * 25 < dur; i++) { const n = i + 1; r.intermediate_chunks.push({ chunk_num: n, start_time: i * 25, end_time: Math.min(dur, i * 25 + 30), filtered_path_rel: r.filtered_full_path_rel ? `output/filtered_chunk_${n}_${r.timestamp}.wav` : null, denoised_path_rel: r.denoised_full_path_rel ? `output/denoised_chunk_${n}_${r.timestamp}.wav` : null }); }
    }
    if (r.id) S.loadedId = r.id;
    applyDataset(r); setStatus('Transcription completed successfully.'); fetchHistory(1);
  } catch (err) { console.error(err); $('error').textContent = 'Error: ' + err.message; setStatus(''); }
  $('submitBtn').textContent = 'Upload & Transcribe'; $('submitBtn').disabled = !fileEl.files[0]; $('spinner').hidden = true;
};

// ---------- init ----------
setLead(0.12); renderToggles(); renderStats(); renderTranscript(); renderChunks(); fetchHistory(1);
})();
