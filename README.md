# LeukNet AML Detector - Node.js version (no Python)

Front end: HTML + CSS + JavaScript (public/).
Back end:  Node.js + Express (server.js). It serves the site, stores the model and
           lets an admin upload a new one. Predictions run in the visitor's browser
           with TensorFlow.js, so the server needs very little memory.

## Files
server.js, package.json, render.yaml, public/ (index.html, style.css, app.js), default-model/

## 1. Convert your model once (Colab)
Use convert_model_to_tfjs.py (from the Vercel pack). It produces a tfjs_model folder:
model.json + group1-shard*.bin files.

## 2. Test on your PC
  npm install
  set ADMIN_PASSWORD=pick-a-password        (Windows CMD)   or   export ADMIN_PASSWORD=...   (Mac/Linux)
  npm start
Open http://localhost:3000 > Model Admin > upload model.json + all .bin files.

## 3. Deploy on Render
1. Push this folder to GitHub (node_modules and data are ignored).
2. Render > New > Blueprint > pick the repo > Apply. (Or New > Web Service: runtime Node,
   build "npm install", start "node server.js".)
3. Set ADMIN_PASSWORD when asked. Open your onrender.com link.
4. Model Admin page > enter the password > choose model.json + all .bin files > Upload.

## Keeping the model permanently
- Paid (Starter + disk, default in render.yaml): uploads are saved on the disk at /var/data
  and survive restarts and redeploys. The previous model is kept once as model_prev.
- Free: Render's free plan has no disk, so uploads are lost on restart. Instead put the model
  files in default-model/, commit and push. The server uses them automatically.

## Notes
- TIFF images are not supported by browsers. Use PNG or JPG.
- Patients and reports are stored only in each visitor's browser (localStorage).
- Research prototype. Not a medical device. Do not enter real patient data.
