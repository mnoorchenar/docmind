"""Build the root README.md, which Hugging Face shows, from .github/README.md.

GitHub shows .github/README.md, so its page has no settings table. Hugging Face needs
its settings block at the top of the root README. Both pages show the same content.

Edit .github/README.md, then run from the repo root:

    python -m scripts.sync_readme
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".github" / "README.md"
TARGET = ROOT / "README.md"

SETTINGS = """---
title: DocMind-Agentic-Research
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: mit
---

"""
NOTE = "<!-- Generated from .github/README.md by scripts/sync_readme.py. Edit that file instead. -->\n\n"


def build():
    source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")
    # Paths in .github/README.md are relative to .github/.
    return SETTINGS + NOTE + source.replace("../", "")


def main():
    TARGET.write_text(build(), encoding="utf-8", newline="\n")
    print(f"Updated {TARGET.name}")


if __name__ == "__main__":
    main()
