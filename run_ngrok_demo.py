"""Start the helmet system locally and expose it with ngrok.

Usage:
    1. Install ngrok auth token once:
       ngrok config add-authtoken YOUR_TOKEN

    2. Run this script:
       python run_ngrok_demo.py

    3. Share the HTTPS URL printed in the terminal.

Notes:
    - This starts web_app.py in a subprocess on port 5000.
    - It uses pyngrok to create a public tunnel.
    - Stop with Ctrl+C.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _start_web_app() -> subprocess.Popen:
    python_exe = sys.executable
    project_root = _project_root()
    env = os.environ.copy()
    env["PORT"] = str(PORT)
    env["HOST"] = HOST

    return subprocess.Popen(
        [python_exe, "web_app.py"],
        cwd=str(project_root),
        env=env,
    )


def _start_ngrok_tunnel(port: int):
    from pyngrok import ngrok

    public_url = ngrok.connect(port, "http").public_url
    return ngrok, public_url


def main() -> int:
    project_root = _project_root()
    if not (project_root / "web_app.py").exists():
        print("Could not find web_app.py next to this script.")
        return 1

    print("Starting local web app on port", PORT)
    web_proc = _start_web_app()

    try:
        print("Starting ngrok tunnel...")
        ngrok_module, public_url = _start_ngrok_tunnel(PORT)
        print()
        print("=" * 72)
        print("Your public demo link:")
        print(public_url)
        print("=" * 72)
        print()
        print("Share that HTTPS link with your friend or teacher.")
        print("Leave this window open while they test the app.")
        print("Press Ctrl+C to stop the tunnel and app.")

        while True:
            time.sleep(1)
            if web_proc.poll() is not None:
                print("web_app.py stopped unexpectedly.")
                return web_proc.returncode or 1
    except KeyboardInterrupt:
        print("Stopping demo...")
        return 0
    except ModuleNotFoundError:
        print()
        print("pyngrok is not installed yet.")
        print("Install it with: pip install pyngrok")
        return 1
    finally:
        try:
            if web_proc.poll() is None:
                if os.name == "nt":
                    web_proc.terminate()
                else:
                    web_proc.send_signal(signal.SIGTERM)
        except Exception:
            pass

        try:
            if "ngrok_module" in locals():
                ngrok_module.disconnect("http://localhost:%d" % PORT)
                ngrok_module.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
