---
# Helmet Violation System

This repo runs the Helmet Violation System Flask app (`web_app.py`) with user/admin pages, helmet detection, plate OCR, evidence capture, and optional invoice emails.

Best deployment choices for your case:
- Render: best balance of ease and a public URL without rewriting the app.
- ngrok: fastest backup demo if you want to run it from your laptop.
- Fly / Railway: solid alternatives, but usually more setup.

## Render setup

Render can run this repo as a Docker web service. The repo now includes `render.yaml`, and the Dockerfile binds to `PORT`, so the same image works locally and on Render.

Manual steps:
1. Push this repo to GitHub.
2. In Render, choose New + Web Service.
3. Connect the GitHub repo.
4. Use the Docker environment or let Render detect `render.yaml`.
5. Set environment variables:
	- `SECRET_KEY`
	- `ADMIN_USERNAME`
	- `ADMIN_PASSWORD`
	- `SMTP_USER`
	- `SMTP_PASSWORD`
	- `TEST_EMAIL`
6. Deploy and wait for the first build.
7. Open the service URL and check `/health`.

Important Render note:
- The model files must stay in the Docker build context. This repo’s `.dockerignore` no longer excludes `models/` and `weights/` so the image can include the detection weights.

## ngrok backup plan

Use ngrok when you need a quick demo today and you do not want to wait for hosting.

1. Start the app locally:
```powershell
.\venv\Scripts\Activate.ps1
python web_app.py
```
2. In a second terminal, expose port 5000:
```powershell
ngrok authtoken YOUR_NGROK_TOKEN
ngrok http 5000
```
3. Share the HTTPS URL ngrok prints.

If you want a stable local Windows run, you can also use Waitress:
```powershell
waitress-serve --listen=0.0.0.0:5000 web_app:app
```

## Local Docker test

```bash
docker build -t helmet-system:latest .
docker run -p 8080:8080 --env-file .env helmet-system:latest
```

## Recommendation

For your teacher demo, the best order is:
1. Render for the real public hosted version.
2. ngrok as the emergency backup.
3. Keep Hugging Face closed for now, since it is stuck on startup routing.
1. `flyctl auth login` (opens browser to authenticate)
2. `./deploy.sh helmet-system` to build and deploy the Docker image
3. `flyctl secrets set ...` to push SMTP/ADMIN secrets from your `.env` (I will prompt before sending any secrets)

If you'd like me to proceed now, reply "deploy now" and I will start. If you prefer to run these yourself, run the `deploy.sh` steps above.

No-card persistent hosting option
- Use Hugging Face Spaces with Docker so it launches `web_app.py` through the `Dockerfile`.
- Copy this repo to GitHub, then create a new Space from the repository and let Spaces build the Docker image.
- If the full model is too heavy, either use lighter weights for the Space or host the weights externally and download them on startup.

Typical Spaces startup command
```bash
python web_app.py
```
