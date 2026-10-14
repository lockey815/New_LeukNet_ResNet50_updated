# LeukNet AML Detector - Render version

## Files
app.py, requirements.txt, render.yaml, .streamlit/config.toml, .gitignore

## 1. Put the folder on GitHub
Create a repo and upload these files. Do NOT upload the .keras model; you upload it
through the app in step 4.

## 2. Create the service on Render
Dashboard > New > Blueprint > pick your repo > Apply (uses render.yaml).
When asked, fill in:
  ADMIN_PASSWORD  a strong password you choose (needed to upload models)
  HF_TOKEN        (optional) Hugging Face token with WRITE access
  HF_MODEL_REPO   (optional) e.g. your-name/leuknet-model  (created privately for you)

## 3. Wait for the build (about 5-10 minutes), then open your onrender.com link.

## 4. Upload the model
Sidebar > "Admin: upload / replace model" > enter ADMIN_PASSWORD > choose best_resnet50.keras
> "Validate and save model". The sidebar then shows "Live model".

## How the model is kept permanently
- Render disk (/var/data): survives restarts and redeploys. Paid plans only.
- Hugging Face backup: after each upload the model is copied to your private HF repo.
  On every start, if the disk has no model, the app restores it from there. Works on the free plan.
Set up at least one of them, otherwise the model is lost on restart.

## Tips
- Upload a slim copy (about 95 MB) instead of the 213 MB training copy. In Colab:
  model.save("resnet50_inference.keras", include_optimizer=False)
- TensorFlow needs about 1 GB+ of RAM. The free 512 MB plan will probably crash on load.
- Patients and reports live only in the open browser session (not saved). Do not enter real patient data.
- Free plan sleeps after 15 minutes without visitors and takes about a minute to wake.
