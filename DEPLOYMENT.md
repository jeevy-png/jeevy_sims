# Deploying The Dashboard

## Option 1: Streamlit Community Cloud (Recommended)

This is the fastest way to get a public link you can share.

1. Push this folder to a GitHub repository.
2. Confirm these files are in the repo root:
   - `dashboard.py`
   - `requirements.txt`
3. Go to https://share.streamlit.io/
4. Click **New app**.
5. Select:
   - Repository: your GitHub repo
   - Branch: your deploy branch (for example `main`)
   - Main file path: `dashboard.py`
6. Click **Deploy**.
7. Streamlit will build and provide a public URL.

## Option 2: Temporary share link (quick demo)

If you want a quick temporary URL from your local machine:

1. Run app locally:
   - `./.venv/bin/streamlit run dashboard.py`
2. Run an ngrok tunnel to local port 8501.
3. Share the generated ngrok URL.

## Notes

- Public apps are openly accessible unless you add external auth.
- For better performance on hosted instances, keep default `runs` and `days` modest.
- Generated output image files are saved locally in the running environment under `dashboard_runs/`.
