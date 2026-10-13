# Respira - React + FastAPI

respira/
  backend/   app.py, audio_service.py, xray_service.py, respire_preprocessing.py,
             inspect_models.py, requirements.txt, models/
  frontend/  index.html, package.json, vite.config.js, src/main.jsx, src/styles.css

## 1. Copy model files into backend/models/
  respire_logits.keras (or respireai_main_s77.keras), norm_stats.json,
  preproc_config.json, meta.json            <- audio Results folder
  xray_pneumonia_model.keras                <- X-ray model (remove the " (1)" from the name)

## 2. Backend (Python 3.10-3.12)
  cd backend
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  uvicorn app:app --port 8000
  Check http://127.0.0.1:8000/api/health -> audio_ready / xray_ready true

## 2b. Database + login setup (required)
  1. MongoDB: install MongoDB Community locally, or create a free MongoDB Atlas cluster and copy its connection string.
  2. In backend/ copy .env.example to .env and set MONGODB_URI and JWT_SECRET
     (generate: python -c "import secrets; print(secrets.token_urlsafe(48))").
  3. Optional Google login: create an OAuth client (Web application) in Google Cloud Console with
     - Authorized JavaScript origin:  http://localhost:5173
     - Authorized redirect URI:       http://localhost:5173/api/auth/google/callback
     then put GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env. Without them the Google button is hidden.
  4. Restart the backend. /api/health should show db_ready: true.

## 3. Frontend (Node 18+), second terminal
  cd frontend
  npm install
  npm run dev
  Open http://localhost:5173
