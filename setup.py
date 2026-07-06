"""
setup.py — One-click environment setup script for AI Resume Analyzer.

Run this ONCE after cloning the project:
    python setup.py

What it does:
  1. Installs all pip dependencies from requirements.txt
  2. Downloads the spaCy English model (en_core_web_md)
  3. Downloads required NLTK corpora
  4. Creates the .env file if it does not exist
  5. Verifies the installation
"""

import subprocess
import sys
import os


def run(command: str, description: str) -> bool:
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"  ❌ FAILED: {description}")
        return False
    print(f"  ✅ DONE: {description}")
    return True


def install_requirements() -> bool:
    return run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing pip dependencies"
    )


def download_spacy_model() -> bool:
    return run(
        f"{sys.executable} -m spacy download en_core_web_md",
        "Downloading spaCy model: en_core_web_md"
    )


def download_nltk_data() -> bool:
    print(f"\n{'='*60}")
    print("  Downloading NLTK corpora")
    print(f"{'='*60}")
    try:
        import nltk
        corpora = [
            "punkt",
            "stopwords",
            "wordnet",
            "averaged_perceptron_tagger",
            "punkt_tab",
        ]
        for corpus in corpora:
            nltk.download(corpus, quiet=True)
            print(f"  ✅ Downloaded: {corpus}")
        return True
    except Exception as exc:
        print(f"  ❌ NLTK download failed: {exc}")
        return False


def create_env_file() -> None:
    """Create a .env file with placeholder values if it doesn't exist."""
    env_path = ".env"
    if not os.path.exists(env_path):
        content = (
            "# AI Resume Analyzer — Environment Variables\n"
            "# Fill in your API keys when needed\n\n"
            "# OpenAI (optional — future feature)\n"
            "OPENAI_API_KEY=your_openai_api_key_here\n\n"
            "# Google Gemini (optional — future feature)\n"
            "GEMINI_API_KEY=your_gemini_api_key_here\n\n"
            "# App Settings\n"
            "APP_ENV=development\n"
            "LOG_LEVEL=INFO\n"
        )
        with open(env_path, "w") as f:
            f.write(content)
        print("\n  ✅ Created .env file with placeholder keys")
    else:
        print("\n  ℹ️  .env file already exists — skipping")


def verify_installation() -> None:
    """Quick import check to confirm key packages installed correctly."""
    print(f"\n{'='*60}")
    print("  Verifying installation")
    print(f"{'='*60}")
    packages = [
        ("streamlit", "streamlit"),
        ("pdfplumber", "pdfplumber"),
        ("docx", "python-docx"),
        ("spacy", "spacy"),
        ("nltk", "nltk"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("PIL", "Pillow"),
        ("reportlab", "reportlab"),
        ("dotenv", "python-dotenv"),
    ]
    all_ok = True
    for module, package in packages:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} — NOT FOUND")
            all_ok = False

    # Check spaCy model
    try:
        import spacy
        spacy.load("en_core_web_md")
        print("  ✅ spaCy model: en_core_web_md")
    except OSError:
        print("  ❌ spaCy model: en_core_web_md — NOT FOUND")
        all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("  🎉 All dependencies installed successfully!")
        print("  👉 Run the app with:  streamlit run app.py")
    else:
        print("  ⚠️  Some dependencies are missing. Re-run: python setup.py")
    print(f"{'='*60}\n")


def main() -> None:
    print("\n" + "="*60)
    print("  AI Resume Analyzer — Setup Script")
    print("="*60)

    steps = [
        install_requirements,
        download_spacy_model,
        download_nltk_data,
    ]

    for step in steps:
        success = step()
        if not success:
            print("\n⛔ Setup stopped due to an error. Fix the issue and re-run.")
            sys.exit(1)

    create_env_file()
    verify_installation()


if __name__ == "__main__":
    main()
