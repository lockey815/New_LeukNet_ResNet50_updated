const CLASSES = ["Monocyte (Non-AML)", "Myeloblast (AML-positive)"];
const PAGES = ["Dashboard", "Patient Register", "AML Detector", "Reports Manager", "Model Admin"];
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const load = k => { try { return JSON.parse(localStorage.getItem(k)) || []; } catch { return []; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

let patients = load('lk_patients'), reports = load('lk_reports');
let model = null, info = { live: false }, page = PAGES[0], files = [], last = null;

/* ---------- model (loaded from the server, runs in the browser) ---------- */
async function initModel() {
  model = null;
  try {
    info = await (await fetch('/api/model/status', { cache: 'no-store' })).json();
    if (info.live && typeof tf !== 'undefined') {
      const m = await tf.loadLayersModel('/model/model.json?v=' + Date.now());
      if (m.inputs[0].shape.length !== 4 || m.outputs[0].shape[1] !== 2) throw new Error('bad shape');
      model = m;
    }
  } catch { model = null; }
  draw();
}

function imgFromFile(f) {
  return new Promise((ok, no) => {
    const i = new Image(); i.onload = () => ok(i); i.onerror = no; i.src = URL.createObjectURL(f);
  });
}

async function predict(f) {
  if (!model) { // DEMO: simulated, same file = same result
    let h = 0; for (const c of f.name + f.size) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return { idx: h % 2, conf: 70 + (h % 300) / 10 };
  }
  const img = await imgFromFile(f), size = model.inputs[0].shape[1];
  const out = tf.tidy(() => {
    let x = tf.image.resizeBilinear(tf.browser.fromPixels(img).toFloat(), [size, size]);
    x = x.reverse(-1).sub(tf.tensor1d([103.939, 116.779, 123.68])); // ResNet50 preprocessing
    return model.predict(x.expandDims(0));
  });
  const d = await out.data(); out.dispose();
  const idx = d[1] > d[0] ? 1 : 0;
  return { idx, conf: d[idx] * 100 };
}

/* ---------- layout ---------- */
function draw() {
  $('#nav').innerHTML = PAGES.map(p => `<button class="${p === page ? 'on' : ''}" data-p="${p}">${p}</button>`).join('');
  $('#nav').querySelectorAll('button').forEach(b => b.onclick = () => { page = b.dataset.p; draw(); });
  const demo = model ? '' : `<div class="warn"><b>DEMO MODE:</b> no model loaded. Results are simulated and are not a real diagnosis.</div>`;
  $('#app').innerHTML = demo + '<div id="pg"></div>';
  ({ 'Dashboard': dash, 'Patient Register': reg, 'AML Detector': det, 'Reports Manager': rep, 'Model Admin': adm })[page]();
}

function dash() {
  $('#pg').innerHTML = `<h2>Dashboard</h2>
  <div class="grid">
    <div class="card"><div class="num">${patients.length}</div><div class="muted">Patients</div></div>
    <div class="card"><div class="num">${reports.length}</div><div class="muted">Reports</div></div>
    <div class="card"><div class="num">${model ? 'Live' : 'Demo'}</div><div class="muted">Model mode</div></div>
  </div>
  <div class="card"><b>AML Detector</b><p class="muted">ResNet50 classifies single blood cells as monocyte or myeloblast. Images are analysed in your browser and are never uploaded.</p></div>`;
}

function reg() {
  $('#pg').innerHTML = `<h2>Patient Register</h2>
  <div class="card">
    <label>Patient name</label><input id="n">
    <div class="row"><div><label>Gender</label><select id="g"><option>Male</option><option>Female</option></select></div>
    <div><label>Age</label><input id="a" type="number" min="0" max="120" value="12"></div></div>
    <div class="row"><div><label>Address</label><input id="ad"></div><div><label>Contact</label><input id="c"></div></div>
    <button class="btn" id="add">Add Patient</button> <span id="msg" class="muted"></span>
  </div>
  <div class="card wrap"><table><tr><th>Name</th><th>Gender</th><th>Age</th><th>Address</th><th>Contact</th></tr>
  ${patients.map(p => `<tr><td>${esc(p.name)}</td><td>${esc(p.gender)}</td><td>${esc(p.age)}</td><td>${esc(p.address)}</td><td>${esc(p.contact)}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">No patients yet.</td></tr>'}
  </table></div>`;
  $('#add').onclick = () => {
    const name = $('#n').value.trim().toUpperCase();
    if (!name) { $('#msg').textContent = 'Enter the patient name.'; return; }
    patients.push({ name, gender: $('#g').value, age: $('#a').value, address: $('#ad').value, contact: $('#c').value });
    save('lk_patients', patients); reg();
  };
}

function det() {
  $('#pg').innerHTML = `<h2>AML Detector</h2>
  <div class="card">
    <label>For which patient</label>
    <select id="pt">${patients.map(p => `<option>${esc(p.name)}</option>`).join('') || '<option>UNREGISTERED</option>'}</select>
    <label>Upload single-cell images (JPG, PNG, BMP, WEBP)</label>
    <input id="f" type="file" accept="image/jpeg,image/png,image/bmp,image/webp" multiple>
    <div class="thumbs" id="th"></div>
    <button class="btn" id="go" disabled>Process Samples</button>
  </div>
  <div id="res"></div>
  <p class="muted">Research prototype. Not a certified medical device. Results must be confirmed by a qualified professional.</p>`;
  $('#f').onchange = e => {
    files = [...e.target.files];
    $('#th').innerHTML = files.map(f => `<img src="${URL.createObjectURL(f)}" alt="${esc(f.name)}">`).join('');
    $('#go').disabled = !files.length;
  };
  $('#go').onclick = async () => {
    $('#go').disabled = true; $('#go').textContent = 'Processing...';
    const rows = [];
    for (const f of files) {
      try { const r = await predict(f); rows.push({ file: f.name, pred: CLASSES[r.idx], conf: r.conf.toFixed(1), blast: r.idx === 1 }); }
      catch { rows.push({ file: f.name, pred: 'Could not read image', conf: '-', blast: false }); }
    }
    last = { patient: $('#pt').value, rows, time: new Date().toLocaleString() };
    $('#go').disabled = false; $('#go').textContent = 'Process Samples';
    showRes();
  };
  if (last) showRes();
}

function showRes() {
  const b = last.rows.filter(r => r.blast).length, n = last.rows.length;
  $('#res').innerHTML = `<div class="card"><h3>Results${model ? '' : ' (DEMO - simulated)'}</h3>
  <p><b>Patient:</b> ${esc(last.patient)} &nbsp; <b>Samples:</b> ${n} &nbsp; <b>Classified as myeloblast:</b> ${b}</p>
  <div class="wrap"><table><tr><th>File</th><th>Prediction</th><th>Confidence (%)</th></tr>
  ${last.rows.map(r => `<tr><td>${esc(r.file)}</td><td>${esc(r.pred)}</td><td>${r.conf}</td></tr>`).join('')}</table></div>
  <div class="box ${b ? 'red' : ''}"><b>Results Report</b><br>${b ? `${b} of ${n} cell(s) classified as myeloblast. AML may be indicated. Please confirm with a haematologist.` : 'No myeloblasts detected in the uploaded cells.'}</div>
  <button class="btn" id="sv">Save Results to Patient Record</button> <span class="muted" id="sm"></span></div>`;
  $('#sv').onclick = () => {
    reports.push({ date: last.time, patient: last.patient, result: b ? `AML indicated (${b} blast(s))` : 'No blasts detected', samples: n });
    save('lk_reports', reports); $('#sm').textContent = 'Saved to Reports Manager.'; $('#sv').disabled = true;
  };
}

function rep() {
  $('#pg').innerHTML = `<h2>Reports Manager</h2>
  <div class="card wrap"><table><tr><th>Date</th><th>Patient</th><th>Result</th><th>Samples</th></tr>
  ${reports.map(r => `<tr><td>${esc(r.date)}</td><td>${esc(r.patient)}</td><td>${esc(r.result)}</td><td>${esc(r.samples)}</td></tr>`).join('') || '<tr><td colspan="4" class="muted">No saved reports yet.</td></tr>'}
  </table></div>
  <button class="btn" id="csv" ${reports.length ? '' : 'disabled'}>Download CSV</button>
  <button class="btn alt" id="clr" ${reports.length ? '' : 'disabled'}>Clear reports</button>`;
  $('#csv').onclick = () => {
    const q = v => `"${String(v).replace(/"/g, '""')}"`;
    const csv = 'Date,Patient,Result,Samples\n' + reports.map(r => [r.date, r.patient, r.result, r.samples].map(q).join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'aml_reports.csv'; a.click();
  };
  $('#clr').onclick = () => { if (confirm('Delete all saved reports?')) { reports = []; save('lk_reports', reports); rep(); } };
}

/* ---------- model admin: upload a converted model from your computer ---------- */
function adm() {
  const st = info.live
    ? `<div class="ok">Model ${model ? 'loaded' : 'found but could not be loaded'}. Source: ${esc(info.source)}. Saved: ${esc(new Date(info.updated).toLocaleString())}</div>`
    : `<div class="warn">No model saved on the server yet.</div>`;
  $('#pg').innerHTML = `<h2>Model Admin</h2>${st}
  <div class="card">
    <p class="muted">Upload the converted TensorFlow.js model: select <b>model.json</b> and <b>all .bin files</b> together. The server saves it for future use.</p>
    <label>Admin password</label><input id="pw" type="password" autocomplete="current-password">
    <label>Model files (model.json + .bin files)</label><input id="mf" type="file" accept=".json,.bin" multiple>
    <button class="btn" id="up">Upload and save model</button>
    <div id="am"></div>
  </div>`;
  $('#up').onclick = async () => {
    const sel = [...$('#mf').files];
    if (!$('#pw').value || !sel.length) { $('#am').innerHTML = '<div class="err">Enter the password and choose the files.</div>'; return; }
    const fd = new FormData(); sel.forEach(f => fd.append('files', f, f.name));
    $('#up').disabled = true; $('#am').innerHTML = '<div class="muted">Uploading, please wait...</div>';
    try {
      const r = await fetch('/api/model', { method: 'POST', headers: { 'x-admin-password': $('#pw').value }, body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || 'Upload failed.');
      await initModel(); page = 'Model Admin'; draw();
      $('#am').innerHTML = `<div class="ok">Saved ${j.files} files. ${model ? 'The model is now live.' : 'Saved, but the model could not be loaded. Check that it is a 2-class image model.'}</div>`;
    } catch (e) {
      $('#am').innerHTML = `<div class="err">${esc(e.message)}</div>`;
    }
    $('#up').disabled = false;
  };
}

draw();
initModel();
