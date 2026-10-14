// LeukNet back end (Node.js + Express)
// - serves the web app from /public
// - serves the current TensorFlow.js model at /model/model.json
// - lets an admin upload a new model (password protected) and keeps it on disk
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const PORT = process.env.PORT || 3000;
const ADMIN = process.env.ADMIN_PASSWORD || '';
let DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');
try { fs.mkdirSync(DATA_DIR, { recursive: true }); }
catch { DATA_DIR = path.join(__dirname, 'data'); fs.mkdirSync(DATA_DIR, { recursive: true }); }

const LIVE = path.join(DATA_DIR, 'model');            // uploaded model (permanent if DATA_DIR is a disk)
const PREV = path.join(DATA_DIR, 'model_prev');       // previous model (one backup)
const NEW = path.join(DATA_DIR, 'model_new');
const TMP = path.join(DATA_DIR, 'tmp');
const BUNDLED = path.join(__dirname, 'default-model'); // model committed to the repo (optional)
fs.mkdirSync(TMP, { recursive: true });

const has = d => fs.existsSync(path.join(d, 'model.json'));
function currentModel() {
  if (has(LIVE)) return { dir: LIVE, source: 'uploaded' };
  if (has(BUNDLED)) return { dir: BUNDLED, source: 'bundled' };
  return null;
}

const app = express();
app.set('trust proxy', 1);
app.use(express.static(path.join(__dirname, 'public')));

app.use('/model', (req, res, next) => {
  const m = currentModel();
  if (!m) return res.status(404).end();
  express.static(m.dir, { etag: true, maxAge: 0 })(req, res, next);
});

app.get('/api/model/status', (req, res) => {
  const m = currentModel();
  res.json(m ? { live: true, source: m.source,
    updated: fs.statSync(path.join(m.dir, 'model.json')).mtime.toISOString() } : { live: false });
});

// ---- admin auth (password in header, limited wrong attempts) ----
const sha = s => crypto.createHash('sha256').update(String(s)).digest();
const fails = new Map();
function auth(req, res, next) {
  if (!ADMIN) return res.status(503).json({ error: 'Uploads are disabled. Set ADMIN_PASSWORD on the server.' });
  const now = Date.now(), rec = fails.get(req.ip) || { n: 0, t: now };
  if (now - rec.t > 15 * 60e3) { rec.n = 0; rec.t = now; }
  if (rec.n >= 10) return res.status(429).json({ error: 'Too many wrong passwords. Try again in 15 minutes.' });
  if (!crypto.timingSafeEqual(sha(req.get('x-admin-password') || ''), sha(ADMIN))) {
    rec.n++; fails.set(req.ip, rec);
    return res.status(401).json({ error: 'Wrong password.' });
  }
  next();
}

const upload = multer({
  dest: TMP,
  limits: { fileSize: 300 * 1024 * 1024, files: 400 },
  fileFilter: (req, file, cb) => cb(null, /^[\w.\-]+\.(json|bin)$/i.test(path.basename(file.originalname))),
});

app.post('/api/model', auth, upload.array('files', 400), (req, res) => {
  const files = req.files || [];
  const cleanup = () => files.forEach(f => fs.rmSync(f.path, { force: true }));
  const fail = (msg) => { cleanup(); return res.status(400).json({ error: msg }); };

  const byName = {};
  files.forEach(f => { byName[path.basename(f.originalname)] = f; });
  if (!byName['model.json']) return fail('model.json is missing. Select model.json and all .bin files together.');

  let j;
  try { j = JSON.parse(fs.readFileSync(byName['model.json'].path, 'utf8')); }
  catch { return fail('model.json is not valid JSON.'); }
  if (!j.modelTopology || !Array.isArray(j.weightsManifest))
    return fail('model.json is not a TensorFlow.js model.');

  const missing = j.weightsManifest.flatMap(g => g.paths).filter(p => !byName[path.basename(p)]);
  if (missing.length) return fail('Missing weight files: ' + missing.slice(0, 5).join(', '));

  try {
    fs.rmSync(NEW, { recursive: true, force: true });
    fs.mkdirSync(NEW, { recursive: true });
    for (const [name, f] of Object.entries(byName)) fs.renameSync(f.path, path.join(NEW, name));
    fs.rmSync(PREV, { recursive: true, force: true });
    if (fs.existsSync(LIVE)) fs.renameSync(LIVE, PREV);
    fs.renameSync(NEW, LIVE);
  } catch (e) {
    cleanup();
    return res.status(500).json({ error: 'Could not save the model: ' + e.message });
  }
  res.json({ ok: true, files: Object.keys(byName).length });
});

app.use((err, req, res, next) => {
  res.status(400).json({ error: err.code === 'LIMIT_FILE_SIZE' ? 'File too large.' : err.message });
});

app.listen(PORT, '0.0.0.0', () => console.log(`LeukNet running on port ${PORT}, data in ${DATA_DIR}`));
