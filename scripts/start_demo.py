"""
1-Click Demo Launcher for Voice-Clone Impersonation Detector.

Spawns both:
1. FastAPI Backend Server (port 8000)
2. Next.js Surveillance Dashboard (port 3000)
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Fix Windows console encoding if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXE = ROOT_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
FRONTEND_DIR = ROOT_DIR / "frontend"


def main():
    print("=" * 70)
    print("🚀 LAUNCHING VOICE-CLONE IMPERSONATION DETECTOR")
    print("=" * 70)
    
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if Path("D:/").exists():
        env["HF_HOME"] = "D:/hf_cache"
        env["TORCH_HOME"] = "D:/torch_cache"
        env["TEMP"] = "D:/npm_temp"
        env["TMP"] = "D:/npm_temp"
    else:
        cache_dir = Path.home() / ".cache"
        env["HF_HOME"] = str(cache_dir / "huggingface")
        env["TORCH_HOME"] = str(cache_dir / "torch")
    env["NODE_OPTIONS"] = "--max-old-space-size=4096"
    
    # 1. Start Backend
    print("\n[1/2] Starting FastAPI Backend on http://localhost:8000...")
    backend_cmd = [
        str(PYTHON_EXE), "-m", "uvicorn",
        "backend.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT_DIR),
        env=env
    )
    
    # Wait for backend initialization
    time.sleep(2)
    
    # 2. Start Frontend
    print("\n[2/2] Starting Next.js Dashboard on http://localhost:3000...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        env=env
    )
    
    print("\n" + "=" * 70)
    print("✨ ALL SERVICES RUNNING:")
    print("   • Surveillance UI: http://localhost:3000")
    print("   • Backend API:     http://localhost:8000")
    print("   • API Docs:        http://localhost:8000/docs")
    print("   • WebSocket:       ws://localhost:8000/ws/analyze")
    print("=" * 70)
    print("Press Ctrl+C to terminate both servers...\n")
    
    def cleanup(signum, frame):
        print("\nShutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        cleanup(None, None)


if __name__ == "__main__":
    main()
