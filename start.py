import os
import sys
import subprocess
import time
import webbrowser
import signal
import threading

def check_venv():
    """Check if running inside a virtual environment."""
    if sys.prefix == sys.base_prefix:
        print("Warning: You are not running inside a virtual environment.")
        print("It is recommended to activate your .venv first.")
        # We don't exit, just warn, as the user might have system-wide packages.

def stream_output(process, prefix):
    """Stream output from a subprocess to stdout."""
    for line in iter(process.stdout.readline, ""):
        if line:
            print(f"[{prefix}] {line.strip()}")

def main():
    print("Starting Pneumonia Detector System...")
    check_venv()

    # Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
    
    # define processes
    backend_process = None
    frontend_process = None

    try:
        # Start Backend
        print("Starting Backend (FastAPI)...")
        backend_cmd = ["uvicorn", "backend.main:app", "--reload", "--port", "8000"]
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=ROOT_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        # Start Frontend
        print("Starting Frontend (Next.js)...")
        frontend_cmd = ["npm", "run", "dev"]
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        # Wait for services to spin up (naïve wait, ideally check health endpoint)
        print("Waiting for services to be ready...")
        time.sleep(5) 

        # Open Browser
        frontend_url = "http://localhost:3000"
        print(f"Opening {frontend_url} in your browser...")
        webbrowser.open(frontend_url)

        print("\n\033[92mSystem is running! Press Ctrl+C to stop.\033[0m\n")
        
        # Keep main thread alive
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
