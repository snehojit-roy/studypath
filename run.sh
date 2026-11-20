#!/usr/bin/env bash
# StudyPath — Concept-Gap Tutor Launcher for macOS / Linux

echo "========================================================"
echo "       StudyPath — Concept-Gap Tutor Launcher"
echo "========================================================"
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

# 2. Virtual environment setup
if [ ! -d "venv" ]; then
    echo "[*] Creating isolated virtual environment (venv)..."
    python3 -m venv venv
fi

# 3. Activate
source venv/bin/activate

# 4. Install dependencies
echo "[*] Checking dependencies..."
pip install -r requirements.txt --quiet

# 5. Launch
echo ""
echo "========================================================"
echo " Launching StudyPath at http://localhost:8501"
echo " Press Ctrl+C to stop the server"
echo "========================================================"
echo ""
streamlit run app.py
