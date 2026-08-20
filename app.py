"""Streamlit prototyp na vyhľadávanie vzdelávacích materiálov Viki."""

from __future__ import annotations

import ast
import html
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MATERIAL_FILE_CANDIDATES = ("zs_vycistene.csv", "zs_vycistne.csv")
STANDARDS_FILE = "SVPv5.csv"
OPTIMIZED_DATA_DIR = APP_DIR / "data"
OPTIMIZED_SCHEMA_VERSION = 1

MATERIAL_COLUMNS = {
    "resource_title",
    "description",
    "subject",
    "cyklus",
    "keywords",
    "standard",
    "link",
}
STANDARD_COLUMNS = {"id", "predmet", "cyklus", "typ", "definicia"}

RESOURCE_TYPE_LABELS = {
    "ziacka-lekcia": "Žiacka lekcia",
    "student-lesson": "Žiacka lekcia",
    "cvicenie": "Cvičenie",
    "interaktivne-cvicenie": "Interaktívne cvičenie",
    "video": "Video",
    "pdf": "PDF",
    "doplnkovy-material": "Doplnkový materiál",
    "audio-text": "Audio",
    "interactive-book": "Interaktívna kniha",
    "kurz": "Kurz",
    "aktivita": "Aktivita",
    "motivacny-vstup": "Motivačný vstup",
    "idp": "IDP",
}

RESOURCE_TYPE_ICONS = {
    "video": "▶",
    "audio-text": "♪",
    "pdf": "PDF",
    "cvicenie": "✓",
    "interaktivne-cvicenie": "✦",
    "interactive-book": "▤",
    "ziacka-lekcia": "▦",
    "student-lesson": "▦",
    "kurz": "▥",
}

st.set_page_config(
    page_title="Katalóg vzdelávacích materiálov",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def normalize_text(value: object) -> str:
    """Return a lowercase, accent-insensitive representation for searching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def parse_python_list(value: object) -> list[str]:
    """Parse the list-like values stored in the source CSV safely."""
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


def validate_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Súbor {filename} nemá povinné stĺpce: {', '.join(missing)}"
        )


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def join_unique(values: Iterable[object]) -> str:
    unique: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    return " · ".join(unique)


def find_material_file() -> Path:
    for filename in MATERIAL_FILE_CANDIDATES:
        path = APP_DIR / filename
        if path.exists():
            return path
    expected = " alebo ".join(MATERIAL_FILE_CANDIDATES)
    raise FileNotFoundError(f"V priečinku aplikácie chýba {expected}.")


def optimized_data_available(manifest_path: Path) -> bool:
    """Validate the self-contained optimized production data bundle."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != OPTIMIZED_SCHEMA_VERSION:
            return False
        outputs = manifest["outputs"]
        for filename in ("materials.parquet", "standards.parquet"):
            output_path = OPTIMIZED_DATA_DIR / filename
            if filename not in outputs or not output_path.is_file():
                return False
            if outputs[filename].get("size") != output_path.stat().st_size:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


@st.cache_data(show_spinner="Načítavam materiály a štandardy…")
def load_data(
    material_path: str,
    material_mtime: int,
    standards_path: str,
    standards_mtime: int,
    manifest_path: str,
    manifest_mtime: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and enrich both local datasets."""
    del material_mtime, standards_mtime, manifest_mtime  # Cache invalidation keys.

    optimized_manifest = Path(manifest_path)
    if optimized_data_available(optimized_manifest):
        materials = pd.read_parquet(OPTIMIZED_DATA_DIR / "materials.parquet")
        standards = pd.read_parquet(OPTIMIZED_DATA_DIR / "standards.parquet")
        return materials, standards

    if not material_path or not standards_path:
        raise FileNotFoundError(
            "Chýbajú platné optimalizované dáta aj zdrojové CSV súbory."
        )

    materials = pd.read_csv(material_path, low_memory=False)
    standards_raw = pd.read_csv(standards_path, low_memory=False)
    validate_columns(materials, MATERIAL_COLUMNS, Path(material_path).name)
    validate_columns(standards_raw, STANDARD_COLUMNS, Path(standards_path).name)

    materials = materials.copy()
    materials["_cycles"] = materials["cyklus"].map(parse_python_list)
    materials["_standard_ids"] = materials["standard"].map(parse_python_list)
    materials["_keyword_list"] = materials["keywords"].map(parse_keywords)
    materials["_search_text"] = (
        materials["resource_title"].fillna("").astype(str)
        + " "
        + materials["description"].fillna("").astype(str)
    ).map(normalize_text)

    link_counts = Counter(
        standard_id
        for ids in materials["_standard_ids"]
        for standard_id in ids
    )

    standards_raw = standards_raw[standards_raw["id"].isin(link_counts)].copy()
    standards_raw["cyklus"] = pd.to_numeric(standards_raw["cyklus"], errors="coerce")

    # One standard ID can occur in several components. Present it once and retain
    # every component/topic in the combined metadata.
    optional_columns = {
        "komponent": "component",
        "tematický celok": "topic",
        "typ štandardu": "standard_subtype",
    }
    aggregations: dict[str, object] = {
        "predmet": first_nonempty,
        "cyklus": "first",
        "typ": first_nonempty,
        "definicia": first_nonempty,
    }
    for source_column in optional_columns:
        if source_column in standards_raw.columns:
            aggregations[source_column] = join_unique

    standards = standards_raw.groupby("id", as_index=False, sort=False).agg(aggregations)
    standards = standards.rename(columns=optional_columns)
    for target in optional_columns.values():
        if target not in standards.columns:
            standards[target] = ""
    standards["cyklus"] = standards["cyklus"].astype("Int64")
    standards["material_count"] = standards["id"].map(link_counts).fillna(0).astype(int)
    standards["_search_text"] = standards.apply(
        lambda row: normalize_text(
            " ".join(
                str(row.get(column, ""))
                for column in (
                    "id",
                    "definicia",
                    "component",
                    "topic",
                    "typ",
                    "standard_subtype",
                )
            )
        ),
        axis=1,
    )
    standards = standards.sort_values(
        ["predmet", "cyklus", "typ", "id"], na_position="last"
    ).reset_index(drop=True)

    return materials, standards


def cycle_label(cycles: Iterable[str]) -> str:
    values = sorted({str(cycle).strip() for cycle in cycles if str(cycle).strip()})
    if not values:
        return "Cyklus neuvedený"
    return f"{', '.join(values)}. cyklus"


def shorten(text: object, limit: int = 280) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rsplit(" ", 1)[0] + "…"


def apply_material_filters(
    materials: pd.DataFrame,
    query: str,
    subjects: list[str],
    cycles: list[str],
    standard_id: str | None = None,
) -> pd.DataFrame:
    filtered = materials

    normalized_query = normalize_text(query).strip()
    if normalized_query:
        filtered = filtered[
            filtered["_search_text"].str.contains(
                normalized_query, regex=False, na=False
            )
        ]
    if subjects:
        filtered = filtered[filtered["subject"].isin(subjects)]
    if cycles:
        selected_cycles = set(cycles)
        filtered = filtered[
            filtered["_cycles"].map(lambda row_cycles: bool(selected_cycles & set(row_cycles)))
        ]
    if standard_id:
        filtered = filtered[
            filtered["_standard_ids"].map(lambda ids: standard_id in ids)
        ]

    if normalized_query and not filtered.empty:
        title_hits = filtered["resource_title"].fillna("").map(normalize_text).str.contains(
            normalized_query, regex=False, na=False
        )
        filtered = filtered.assign(_title_hit=title_hits.astype(int)).sort_values(
            ["_title_hit", "resource_title"], ascending=[False, True]
        )
    else:
        filtered = filtered.sort_values("resource_title", na_position="last")
    return filtered


def render_material_card(material: pd.Series) -> None:
    with st.container(border=True):
        resource_type = str(material.get("resource_type", "")).strip()
        type_label = RESOURCE_TYPE_LABELS.get(resource_type, resource_type.replace("-", " ").title())
        type_icon = RESOURCE_TYPE_ICONS.get(resource_type, "◆")
        subject_label = html.escape(str(material.get("subject", "Predmet neuvedený")))
        cycles_label = html.escape(cycle_label(material.get("_cycles", [])))
        st.markdown(
            f"""
            <div class="viki-card-cover viki-card-cover--{resource_type}">
                <span class="viki-card-icon">{type_icon}</span>
                <span class="viki-card-cover-label">Vzdelávací materiál</span>
                <span class="viki-card-cover-type">{html.escape(type_label or 'Materiál')}</span>
                <span class="viki-card-cover-meta">
                    {subject_label} · {cycles_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"#### {material.get('resource_title', 'Materiál bez názvu')}")
        description = shorten(material.get("description", ""))
        if description:
            st.write(description)

        visible_keywords = material.get("_keyword_list", [])[:6]
        if len(visible_keywords) > 0:
            suffix = " …" if len(material.get("_keyword_list", [])) > 6 else ""
            st.caption("Kľúčové slová: " + " · ".join(visible_keywords) + suffix)

        link = str(material.get("link", "")).strip()
        if link and link.lower() != "nan":
            st.link_button("Otvoriť materiál ↗", link, use_container_width=True)


def render_results(
    materials: pd.DataFrame,
    result_key: str,
) -> None:
    result_count = len(materials)
    st.markdown(f"### Nájdené materiály · {result_count:,}".replace(",", " "))
    if materials.empty:
        st.info("Pre zvolené podmienky sa nenašli žiadne materiály. Skúste upraviť vyhľadávanie alebo filtre.")
        return

    size_col, page_col, space_col = st.columns([1, 1, 4])
    with size_col:
        page_size = st.selectbox(
            "Na stránke", [12, 24, 48], key=f"page_size_{result_key}"
        )
    page_count = max(1, math.ceil(result_count / page_size))
    with page_col:
        page = st.selectbox(
            "Strana",
            range(1, page_count + 1),
            format_func=lambda value: f"{value} / {page_count}",
            key=f"page_{result_key}_{page_size}_{page_count}",
        )

    start = (page - 1) * page_size
    page_rows = materials.iloc[start : start + page_size]
    columns = st.columns(3)
    for position, (_, material) in enumerate(page_rows.iterrows()):
        with columns[position % 3]:
            render_material_card(material)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --viki-purple: #742f8a;
                --viki-purple-light: #967faf;
                --viki-yellow: #ffd700;
                --viki-bg: #fafafa;
                --viki-panel: #ffffff;
                --viki-muted: #6c6c76;
            }

            [data-testid="stHeader"], [data-testid="stSidebar"] {display: none;}
            .stApp {
                background: var(--viki-bg);
                color: rgba(0, 0, 0, .87);
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }
            .block-container {
                max-width: 1500px;
                padding: 2rem 3rem 4rem;
            }

            h1, h2, h3, h4 {
                font-family: "Segoe UI Semibold", "Segoe UI", sans-serif !important;
                color: rgba(0,0,0,.87) !important;
            }
            h1 {font-size: 2rem !important; font-weight: 600 !important; letter-spacing: 0 !important;}
            h3 {font-size: 1.35rem !important; font-weight: 600 !important;}
            h4 {font-size: 1rem !important; font-weight: 600 !important; line-height: 1.3 !important;}
            .catalog-kicker {
                font-size: .76rem;
                letter-spacing: .08em;
                text-transform: uppercase;
                font-weight: 700;
                color: var(--viki-purple);
            }
            .catalog-subtitle {
                font-size: 1rem;
                color: var(--viki-muted);
                max-width: 820px;
                margin-bottom: 1.7rem;
            }

            [data-baseweb="input"], [data-baseweb="select"] > div {
                background: white !important;
                border-color: #d9d8df !important;
                border-radius: 4px !important;
            }
            [data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
                border-color: var(--viki-purple) !important;
                box-shadow: 0 0 0 1px var(--viki-purple) !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--viki-panel);
                border: 0 !important;
                border-radius: 16px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,.12);
                overflow: hidden;
                transition: transform .16s ease, box-shadow .16s ease;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 16px rgba(64,31,75,.15);
            }
            .viki-card-cover {
                min-height: 104px;
                margin: -1rem -1rem .9rem;
                padding: 1rem 1.15rem;
                border-radius: 16px 16px 0 0;
                display: grid;
                grid-template-columns: 48px 1fr;
                grid-template-rows: auto auto auto;
                column-gap: .85rem;
                align-content: center;
                color: white;
                background: linear-gradient(135deg, #742f8a, #967faf);
            }
            .viki-card-cover--video {background: linear-gradient(135deg, #742f8a, #a25687);}
            .viki-card-cover--pdf {background: linear-gradient(135deg, #6c377e, #b56776);}
            .viki-card-cover--cvicenie,
            .viki-card-cover--interaktivne-cvicenie {background: linear-gradient(135deg, #593478, #7f76ae);}
            .viki-card-icon {
                grid-row: 1 / 4;
                width: 46px;
                height: 46px;
                border: 2px solid rgba(255,255,255,.72);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: .9rem;
                font-weight: 700;
            }
            .viki-card-cover-label {
                align-self: end;
                font-size: .69rem;
                letter-spacing: .06em;
                text-transform: uppercase;
                opacity: .76;
            }
            .viki-card-cover-type {font-size: 1rem; font-weight: 600;}
            .viki-card-cover-meta {
                margin-top: .15rem;
                font-size: .75rem;
                color: rgba(255,255,255,.78);
            }
            div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
                color: #77747d;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] p {font-size: .9rem; line-height: 1.45;}

            a[data-testid="stBaseLinkButton-secondary"] {
                background: var(--viki-yellow) !important;
                color: var(--viki-purple) !important;
                border: 0 !important;
                border-radius: 4px !important;
                font-family: "Segoe UI Semibold", "Segoe UI", sans-serif;
                font-weight: 600;
                box-shadow: none !important;
            }
            a[data-testid="stBaseLinkButton-secondary"]:hover {
                background: #f2cd00 !important;
                color: #572168 !important;
            }

            @media (max-width: 800px) {
                .block-container {padding: 1.25rem 1rem 3rem;}
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
def main() -> None:
    inject_styles()

    try:
        manifest_path = OPTIMIZED_DATA_DIR / "manifest.json"
        if optimized_data_available(manifest_path):
            material_path: Path | None = None
            standards_path: Path | None = None
        else:
            material_path = find_material_file()
            standards_path = APP_DIR / STANDARDS_FILE
            if not standards_path.exists():
                raise FileNotFoundError(f"V priečinku aplikácie chýba {STANDARDS_FILE}.")
        materials, standards = load_data(
            str(material_path) if material_path else "",
            material_path.stat().st_mtime_ns if material_path else 0,
            str(standards_path) if standards_path else "",
            standards_path.stat().st_mtime_ns if standards_path else 0,
            str(manifest_path),
            manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0,
        )
    except Exception as exc:
        st.error(f"Dáta sa nepodarilo načítať: {exc}")
        st.stop()

    st.markdown('<div class="catalog-kicker">Štátny vzdelávací program pre ZŠ 2023 (dodatok č.5)</div>', unsafe_allow_html=True)
    st.title("Katalóg vzdelávacích materiálov na Viki")
    st.markdown(
        '<div class="catalog-subtitle">Vyhľadajte materiál podľa názvu, opisu alebo vzdelávacieho štandardu.</div>',
        unsafe_allow_html=True,
    )

    search_col, subject_col, cycle_col = st.columns([2.2, 1.4, 1])
    with search_col:
        query = st.text_input(
            "Názov alebo text v popise",
            placeholder="Napríklad zlomky, rozprávka alebo fotosyntéza…",
            help="Vyhľadáva sa v stĺpcoch nadpis a popis.",
        )

    subject_options = sorted(materials["subject"].dropna().astype(str).unique())
    with subject_col:
        selected_subject = st.selectbox(
            "Predmet",
            subject_options,
            index=None,
            placeholder="Všetky predmety",
        )
    with cycle_col:
        selected_cycle = st.selectbox(
            "Cyklus",
            ["1", "2", "3"],
            index=None,
            placeholder="Všetky cykly",
            format_func=lambda value: f"{value}. cyklus",
        )

    candidates = standards.copy()
    if selected_subject:
        candidates = candidates[candidates["predmet"] == selected_subject]
    if selected_cycle:
        candidates = candidates[candidates["cyklus"] == int(selected_cycle)]
    candidate_lookup = candidates.set_index("id").to_dict("index")

    def standard_option_label(standard_id: str) -> str:
        row = candidate_lookup[standard_id]
        definition = shorten(row.get("definicia", ""), 125)
        return (
            f"{definition} {standard_id}"
            f"({row.get('material_count', 0)})"
        )

    selected_standard_id = st.selectbox(
        f"Vzdelávací štandard ({len(candidates)} možností)",
        candidates["id"].tolist(),
        index=None,
        placeholder="Všetky vzdelávacie štandardy",
        format_func=standard_option_label,
        key=f"standard_{selected_subject or 'all'}_{selected_cycle or 'all'}",
        help="Bez vybraného vzdelávacieho štandardu sa zobrazujú všetky materiály zodpovedajúce textu, predmetu a cyklu.",
    )

    result_key = selected_standard_id or "all"

    filtered = apply_material_filters(
        materials,
        query=query,
        subjects=[selected_subject] if selected_subject else [],
        cycles=[selected_cycle] if selected_cycle else [],
        standard_id=selected_standard_id,
    )
    render_results(filtered, result_key)


if __name__ == "__main__":
    main()
