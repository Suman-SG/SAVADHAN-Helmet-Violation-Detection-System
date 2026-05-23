# Deployment Guide

## Best choice for your demo

Use **Render** for the real public hosted app.
Use **ngrok** as the backup if you need a link immediately from your laptop.

## Local run

1. Create and activate your virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Fill in `.env` with your SMTP and admin credentials.
4. Start the web app:

```powershell
python web_app.py
```

The Flask app listens on `PORT` if it is set, otherwise it defaults to `5000`.

## Render deploy

This repo now includes `render.yaml` and the Dockerfile binds to the platform port, so Render can run it directly.

Manual Render steps:
1. Push the repo to GitHub.
2. In Render, create a new Web Service.
3. Connect the GitHub repository.
4. Choose Docker or let Render use `render.yaml`.
5. Add env vars:
	- `SECRET_KEY`
	- `ADMIN_USERNAME`
	- `ADMIN_PASSWORD`
	- `SMTP_USER`
	- `SMTP_PASSWORD`
	- `TEST_EMAIL`
6. Deploy.
7. Open `/health` after the first deploy.

Important:
- The `.dockerignore` file no longer excludes `models/` and `weights/`, so the detection files stay in the image.

## ngrok backup

Use this when you want the fastest possible demo without deploying.

```powershell
.\venv\Scripts\Activate.ps1
python web_app.py
```

In a second terminal:

```powershell
ngrok authtoken YOUR_NGROK_TOKEN
ngrok http 5000
```

Share the HTTPS URL ngrok prints.

## Windows-friendly production run

If you want a stable local server on Windows, use Waitress:

```powershell
waitress-serve --listen=0.0.0.0:5000 web_app:app
```

## Environment variables

Set these before launching:

- `SECRET_KEY`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `TEST_EMAIL`

If email login fails with SMTP 535, the Gmail App Password is missing or wrong.
