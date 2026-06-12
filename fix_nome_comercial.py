"""
Scans each csv/{ht,onco}/original/{N}_original_limpo.txt for a commercial
brand name (tokens followed by ® or ™), and updates the matching metadata
file's <nome_comercial> tag. Then mirrors the change to the simplificada
metadata files.

Strategy:
  - Find all tokens immediately preceding ® or ™.
  - Drop tokens that are clearly the INN (drug name) from remedios_*_map.csv,
    pharmaceutical-form words, or boilerplate.
  - Pick the most frequent remaining token as nome_comercial.
  - If none found, leave existing value untouched.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "csv"
META = ROOT / "csv_metadata"

BRAND_RE = re.compile(r"([A-ZÀ-Ý][\wÀ-ÿ\-]{2,})\s*[®™]")
NOME_COMERCIAL_RE = re.compile(r"(<nome_comercial>)(.*?)(</nome_comercial>)")

STOPWORDS = {
    "Anvisa", "MS", "SAC", "Reg",
}


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def load_drug_names(map_path: Path) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    with map_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("id"):
                continue
            tokens = {
                strip_accents(w).lower()
                for w in re.split(r"\s+", row["nome"])
                if len(w) >= 3
            }
            out[int(row["id"])] = tokens
    return out


def find_brand(text: str, inn_tokens: set[str]) -> str | None:
    counts: Counter[str] = Counter()
    for m in BRAND_RE.finditer(text):
        token = m.group(1)
        norm = strip_accents(token).lower()
        if norm in inn_tokens:
            continue
        if token in STOPWORDS:
            continue
        counts[token] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def update_metadata(meta_path: Path, brand: str) -> bool:
    content = meta_path.read_text(encoding="utf-8")
    new = NOME_COMERCIAL_RE.sub(rf"\1{brand}\3", content, count=1)
    if new == content:
        return False
    meta_path.write_text(new, encoding="utf-8")
    return True


def process_group(group: str, map_file: str, simp_suffix: str) -> None:
    inn_by_id = load_drug_names(ROOT / map_file)
    orig_dir = CSV / group / "original"
    meta_orig = META / group / "original"
    meta_simp = META / group / "simplificada"

    found = 0
    for src in sorted(orig_dir.glob("*_original_limpo.txt")):
        n = int(src.name.split("_", 1)[0])
        inn = inn_by_id.get(n, set())
        brand = find_brand(src.read_text(encoding="utf-8"), inn)
        if not brand:
            continue

        target_orig = meta_orig / src.name
        if target_orig.exists():
            update_metadata(target_orig, brand)

        simp_name = f"{n}{simp_suffix}"
        target_simp = meta_simp / simp_name
        if target_simp.exists():
            update_metadata(target_simp, brand)

        print(f"  [{group}] #{n}: {brand}")
        found += 1

    print(f"[{group}] set nome_comercial on {found} bulas")


def main() -> None:
    process_group("ht", "remedios_ht_map.csv", "_validada_limpo.txt")
    process_group("onco", "remedios_onco_map.csv", "_simplificada_limpo.txt")


if __name__ == "__main__":
    main()
