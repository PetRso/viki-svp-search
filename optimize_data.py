"""Build compact, application-ready Parquet files from the source CSV files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
MATERIAL_CANDIDATES = ("zs_vycistene.csv", "zs_vycistne.csv")
STANDARDS_FILE = "SVPv5.csv"
OUTPUT_DIR = ROOT / "data"
SCHEMA_VERSION = 1


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def parse_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.split(",")]
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    return [str(item).strip() for item in parsed if str(item).strip()]


def parse_keywords(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_material_file() -> Path:
    for filename in MATERIAL_CANDIDATES:
        path = ROOT / filename
        if path.exists():
            return path
    raise FileNotFoundError("Chýba zdrojový súbor materiálov.")


def build() -> dict[str, object]:
    material_path = find_material_file()
    standards_path = ROOT / STANDARDS_FILE
    if not standards_path.exists():
        raise FileNotFoundError(f"Chýba {STANDARDS_FILE}.")

    materials_raw = pd.read_csv(material_path, low_memory=False)
    standards_raw = pd.read_csv(standards_path, low_memory=False)

    materials = pd.DataFrame(
        {
            "link": materials_raw["link"].fillna(""),
            "resource_title": materials_raw["resource_title"].fillna(""),
            "description": materials_raw["description"].fillna(""),
            "subject": materials_raw["subject"].fillna(""),
            "resource_type": materials_raw["resource_type"].fillna(""),
            "_cycles": materials_raw["cyklus"].map(parse_list),
            "_standard_ids": materials_raw["standard"].map(parse_list),
            "_keyword_list": materials_raw["keywords"].map(parse_keywords),
        }
    )
    materials["_search_text"] = (
        materials["resource_title"].astype(str)
        + " "
        + materials["description"].astype(str)
    ).map(normalize_text)

    link_counts = Counter(
        standard_id
        for standard_ids in materials["_standard_ids"]
        for standard_id in standard_ids
    )
    standards_raw = standards_raw[standards_raw["id"].isin(link_counts)].copy()
    standards_raw["cyklus"] = pd.to_numeric(standards_raw["cyklus"], errors="coerce")
    standards = standards_raw.groupby("id", as_index=False, sort=False).agg(
        predmet=("predmet", first_nonempty),
        cyklus=("cyklus", "first"),
        typ=("typ", first_nonempty),
        definicia=("definicia", first_nonempty),
    )
    standards["cyklus"] = standards["cyklus"].astype("Int64")
    standards["material_count"] = standards["id"].map(link_counts).astype("int32")
    standards = standards.sort_values(
        ["predmet", "cyklus", "typ", "id"], na_position="last"
    ).reset_index(drop=True)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    material_output = output_dir / "materials.parquet"
    standards_output = output_dir / "standards.parquet"
    materials.to_parquet(material_output, index=False, compression="zstd")
    standards.to_parquet(standards_output, index=False, compression="zstd")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            material_path.name: {
                "sha256": sha256(material_path),
                "size": material_path.stat().st_size,
            },
            standards_path.name: {
                "sha256": sha256(standards_path),
                "size": standards_path.stat().st_size,
            },
        },
        "outputs": {
            material_output.name: {
                "rows": len(materials),
                "columns": materials.columns.tolist(),
                "size": material_output.stat().st_size,
            },
            standards_output.name: {
                "rows": len(standards),
                "columns": standards.columns.tolist(),
                "size": standards_output.stat().st_size,
            },
        },
        "linked_standard_references": int(sum(link_counts.values())),
        "unique_linked_standards": len(link_counts),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = build()
    original_size = sum(source["size"] for source in manifest["sources"].values())
    optimized_size = sum(output["size"] for output in manifest["outputs"].values())
    reduction = 1 - optimized_size / original_size
    print(
        f"Hotovo: {original_size:,} -> {optimized_size:,} bajtov "
        f"({reduction:.1%} menšie)."
    )


if __name__ == "__main__":
    main()
