"""
Copies metadata files from csv_metadata/{ht,onco}/original/ to
csv_metadata/{ht,onco}/simplificada/, renaming each file to match the
corresponding simplified-text filename in csv/.

Mapping:
  ht/original/{N}_original_limpo.txt    -> ht/simplificada/{N}_validada_limpo.txt
  onco/original/{N}_original_limpo.txt  -> onco/simplificada/{N}_simplificada_limpo.txt
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
META = ROOT / "csv_metadata"

GROUPS = {
    "ht": ("_original_limpo.txt", "_validada_limpo.txt"),
    "onco": ("_original_limpo.txt", "_simplificada_limpo.txt"),
}


def main() -> None:
    for group, (orig_suffix, simp_suffix) in GROUPS.items():
        src_dir = META / group / "original"
        dst_dir = META / group / "simplificada"
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in sorted(src_dir.glob(f"*{orig_suffix}")):
            stem = src.name[: -len(orig_suffix)]
            dst = dst_dir / f"{stem}{simp_suffix}"
            shutil.copyfile(src, dst)
            copied += 1
        print(f"[{group}] copied {copied} metadata files -> {dst_dir}")


if __name__ == "__main__":
    main()
