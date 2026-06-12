"""
Appends the bula text from csv/ to each metadata file in csv_metadata/,
wrapped in <text>...</text> right after the closing </metadata>.

Pairing:
  csv_metadata/ht/original/{N}_original_limpo.txt    <- csv/ht/original/{N}_original_limpo.txt
  csv_metadata/ht/simplificada/{N}_validada_limpo.txt <- csv/ht/simplificada/{N}_validada_limpo.txt
  csv_metadata/onco/original/{N}_original_limpo.txt   <- csv/onco/original/{N}_original_limpo.txt
  csv_metadata/onco/simplificada/{N}_simplificada_limpo.txt <- csv/onco/simplificada/{N}_simplificada_limpo.txt

Idempotent: if a metadata file already contains a <text> block, it is replaced.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "csv"
META = ROOT / "csv_metadata"

SUBDIRS = [
    ("ht", "original"),
    ("ht", "simplificada"),
    ("onco", "original"),
    ("onco", "simplificada"),
]

TEXT_BLOCK_RE = re.compile(r"\n*<text>.*?</text>\s*$", re.DOTALL)


def main() -> None:
    for group, kind in SUBDIRS:
        meta_dir = META / group / kind
        text_dir = CSV / group / kind
        updated = 0
        missing = []

        for meta_path in sorted(meta_dir.glob("*.txt")):
            text_path = text_dir / meta_path.name
            if not text_path.exists():
                missing.append(meta_path.name)
                continue

            text = text_path.read_text(encoding="utf-8").strip()
            meta = meta_path.read_text(encoding="utf-8")
            meta = TEXT_BLOCK_RE.sub("", meta).rstrip()

            meta_path.write_text(
                f"{meta}\n\n<text>\n{text}\n</text>\n",
                encoding="utf-8",
            )
            updated += 1

        print(f"[{group}/{kind}] updated {updated} files"
              + (f"; missing source: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
