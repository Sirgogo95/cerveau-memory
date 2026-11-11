"""Render the scoreboard's title and bars as docs/img/scoreboard.png (1280x640, the size
GitHub uses for a repository's social preview), with headless Microsoft Edge or Chrome.

    python scripts/scoreboard_image.py
    python scripts/scoreboard_image.py --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"

Retake it whenever a number on the scoreboard changes (the README shows the image above
a table with the same numbers).
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"


def page(source: str) -> str:
    """The scoreboard's head (styles, fonts) with only its header and the bars section."""
    head = source[:source.index("<main>")]
    header = re.search(r"<header.*?</header>", source, re.S)
    board = re.search(r'<section class="board".*?</section>', source, re.S)
    if header is None or board is None:
        raise SystemExit("The scoreboard has no <header> or no board section: update this script.")
    return (f'<!doctype html><html><head><meta charset="utf-8">{head}'
            "<style>body{margin:0} main{max-width:1120px;margin:0 auto;padding:44px 40px} .lede{display:none}</style>"
            f"</head><body><main>{header.group(0)}{board.group(0)}</main></body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the scoreboard's bars as docs/img/scoreboard.png.")
    parser.add_argument("--browser", default=EDGE, help="Edge or Chrome executable")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "scoreboard.png")
    args = parser.parse_args()
    html = page((ROOT / "docs" / "cerveau-scoreboard.html").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "board.html"
        source.write_text(html, encoding="utf-8")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([args.browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", "--window-size=1280,640",
                        f"--screenshot={args.out}", source.as_uri()], check=True, capture_output=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
