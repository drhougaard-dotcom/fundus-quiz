# fundus_quiz.py
import os
import io
import random
from typing import List, Tuple, Dict, Optional

import requests
import pandas as pd
from PIL import Image
import streamlit as st

# ===================== CONFIG VIA SECRETS / ENV =====================
def _secret(key: str, default: str = "") -> str:
    """Prefer Streamlit secrets; fall back to environment variables."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

# Bucket / region (shared)
S3_BUCKET = _secret("S3_BUCKET", "fundus-quiz")
S3_REGION = _secret("S3_REGION", "eu-north-1")
USE_S3 = _secret("USE_S3", "1") == "1"
S3_BASE = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

# Per-dataset image prefixes (folders in bucket)
TRAIN_PREFIX = _secret("TRAIN_PREFIX", _secret("S3_PREFIX", "RFMiD/Training"))  # backward compatibility
EVAL_PREFIX  = _secret("EVAL_PREFIX", "RFMiD/Evaluation")
TEST_PREFIX  = _secret("TEST_PREFIX", "RFMiD/Test")

# Per-dataset label CSV URLs (HTTPS)
TRAIN_LABELS_CSV_URL = _secret("TRAIN_LABELS_CSV_URL", _secret("LABELS_CSV_URL", ""))
EVAL_LABELS_CSV_URL  = _secret("EVAL_LABELS_CSV_URL",  "")
TEST_LABELS_CSV_URL  = _secret("TEST_LABELS_CSV_URL",  "")

# Optional tiny local fallback (demo)
LOCAL_LABELS_FALLBACK = "RFMiD_Training_Labels.csv"

# ===================== STREAMLIT PAGE CONFIG =====================
st.set_page_config(page_title="Fundus Pathology Quiz", layout="wide")

# Disable fade/animations globally
st.markdown(
    """
    <style>
      * { transition: none !important; animation: none !important; }
      .st-emotion-cache-1avcm0n, .st-emotion-cache-1y4p8pa { animation: none !important; } /* common fade-in classes */
      .stButton > button { margin-top: 8px; }
      /* tighten logos row */
      .logo-row { display: flex; align-items: center; gap: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== HELPERS =====================
def normalize_id(any_id) -> str:
    """Drop leading zeros from numeric IDs."""
    s = str(any_id).strip()
    try:
        return str(int(s))
    except ValueError:
        return s

def resolve_image_url(prefix: str, image_id: str) -> str:
    """Build direct HTTPS URL to PNG in S3 (non-zero-padded)."""
    clean = normalize_id(image_id)
    return f"{S3_BASE}/{prefix}/{clean}.png"

@st.cache_data(show_spinner=False)
def load_labels_any(url: str, local_fallback: str) -> pd.DataFrame:
    """Load labels from remote URL; fallback to local CSV if present."""
    if url:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))
    if local_fallback and os.path.exists(local_fallback):
        return pd.read_csv(local_fallback)
    raise FileNotFoundError(
        "Could not load labels CSV. Set the dataset’s *_LABELS_CSV_URL in Secrets (S3 HTTPS), "
        "or include a small 'RFMiD_Training_Labels.csv' in the repo."
    )

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_png(url: str) -> io.BytesIO:
    """Download an image from S3 and cache it for an hour."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return io.BytesIO(r.content)

def current_filter_signature(**kwargs):
    """Stable signature from settings to reset state when they change."""
    items = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, list):
            v = tuple(sorted(v))
        items.append((k, v))
    return tuple(items)

def build_mc_options(image_key: str, correct_codes: List[str], selected_codes: List[str], num_choices: int) -> List[str]:
    """Deterministic MC options per image/settings so widgets don’t reset."""
    seed_str = f"{image_key}|{num_choices}|{','.join(sorted(selected_codes))}"
    rng = random.Random(seed_str)
    options = list(dict.fromkeys(correct_codes))  # keep order, de-dup
    distractors = [c for c in selected_codes if c not in options]
    need = max(0, num_choices - len(options))
    if need > 0 and distractors:
        options += rng.sample(distractors, min(need, len(distractors)))
    return sorted(set(options))

def render_answer_pills(codes: List[str]):
    """Show all correct answers as pill-like badges."""
    if not codes:
        st.markdown("**Correct answers:** Normal")
        return
    labels = [label_map.get(c, c) for c in codes]
    pills = " ".join([f"<span class='pill'>{l}</span>" for l in labels])
    st.markdown(
        """
        <style>
        .pill {
            display:inline-block; padding:6px 10px; margin:4px 6px 0 0;
            border-radius:999px; background:#eef2ff; border:1px solid #c7d2fe;
            font-size:0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Correct answers:** {pills}", unsafe_allow_html=True)

def get_hero_image_url() -> Optional[str]:
    """Try to show a small fundus 'hero' image from your public S3."""
    for prefix in [TRAIN_PREFIX, EVAL_PREFIX, TEST_PREFIX]:
        if prefix:
            return resolve_image_url(prefix, "1")
    return None

# ---------- Dataset utilities ----------
DATASETS: Dict[str, Tuple[str, str]] = {
    "Training":   (TRAIN_PREFIX, TRAIN_LABELS_CSV_URL),
    "Evaluation": (EVAL_PREFIX,  EVAL_LABELS_CSV_URL),
    "Test":       (TEST_PREFIX,  TEST_LABELS_CSV_URL),
}

def _prepare_df(df_local: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure numeric label columns, compute virtual NL, and num_pathologies.
    Returns (df, pathology_cols) where pathology_cols excludes ID, NL, and helper columns.
    """
    non_label_cols = {"ID", "NL", "num_pathologies", "dataset", "img_prefix"}
    path_cols = [c for c in df_local.columns if c not in non_label_cols]

    if path_cols:
        df_local[path_cols] = (
            df_local[path_cols]
            .apply(pd.to_numeric, errors="coerce")  # "0"/"1" -> 0/1; NaN -> 0
            .fillna(0)
            .astype(int)
        )

    num_path = df_local[path_cols].sum(axis=1) if path_cols else pd.Series(0, index=df_local.index)
    if "NL" not in df_local.columns:
        df_local["NL"] = (num_path == 0).astype(int)
    df_local["num_pathologies"] = num_path

    final_path_cols = [c for c in df_local.columns if c not in non_label_cols]
    return df_local, final_path_cols

def load_all_datasets() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and union Training/Evaluation/Test (only those with CSV URLs set).
    Outer-join columns across splits, fill missing labels with 0, coerce labels to numeric.
    """
    dfs = []
    for name in ["Training", "Evaluation", "Test"]:
        prefix, csv_url = DATASETS[name]
        if not csv_url:
            continue
        df_local = load_labels_any(csv_url, LOCAL_LABELS_FALLBACK if name == "Training" else "")
        df_local["dataset"] = name
        df_local["img_prefix"] = prefix
        dfs.append(df_local)

    if not dfs:
        raise FileNotFoundError("No dataset labels URLs are set in Secrets. Configure *_LABELS_CSV_URL.")

    df_all = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    non_label_cols = {"ID", "NL", "num_pathologies", "dataset", "img_prefix"}
    candidate_label_cols = [c for c in df_all.columns if c not in non_label_cols]

    if candidate_label_cols:
        df_all[candidate_label_cols] = (
            df_all[candidate_label_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    df_all, path_cols_final = _prepare_df(df_all)
    return df_all, path_cols_final

# ===================== LABEL MAPS =====================
label_map = {
    # Normal / misc
    "NL": "Normal",
    "OTHER": "Other abnormalities",

    # Retinopathy & vascular
    "DR": "Diabetic Retinopathy",
    "HR": "Hemorrhagic Retinopathy",
    "BRVO": "Branch Retinal Vein Occlusion",
    "CRVO": "Central Retinal Vein Occlusion",
    "BRAO": "Branch Retinal Artery Occlusion",
    "CRAO": "Central Retinal Artery Occlusion",
    "MCA": "Macroaneurysm",
    "TV": "Tortuous Vessels",
    "CL": "Collateral Vessels",
    "VS": "Vasculitis",
    "PLQ": "Hollenhorst Plaque",

    # AMD / degenerative / RPE–choroid
    "ARMD": "Age-related Macular Degeneration",
    "DN": "Drusen",
    "RPEC": "Retinal Pigment Epithelium Changes",
    "CRS": "Chorioretinal Scar",
    "CF": "Choroidal Folds",
    "TSLN": "Tessellation",
    "MYA": "Myopia",

    # Macular & vitreoretinal
    "CSR": "Central Serous Retinopathy",
    "ERM": "Epiretinal Membrane",
    "MS": "Macular Scar",
    "MHL": "Macular Hole",
    "CME": "Cystoid Macular Edema",
    "RT": "Retinal Traction",
    "LS": "Laser Scar",

    # Optic disc / neuro-ophthalmology
    "ODE": "Optic Disc Edema (Papilledema)",
    "ODC": "Optic Disc Cupping (Glaucomatous)",
    "ODP": "Optic Disc Pallor",
    "TD": "Tilted Disc",
    "ST": "Optociliary Shunt",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PA": "Peripapillary Atrophy",

    # Hemorrhages / exudates / PED
    "PRH": "Preretinal Hemorrhage",
    "VH": "Vitreous Hemorrhage",
    "EDN": "Exudates / Circinate / Macular Star",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",

    # Inflammatory / infectious / dystrophies / anomalies
    "RS": "Retinitis",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton Wool Spots",
    "CB": "Coloboma",
    "MNF": "Myelinated Nerve Fibers",

    # Media / vitreous / other
    "MH": "Media Haze",  # RFMiD 'MH' = Media Haze (Macular hole is 'MHL')
    "AH": "Asteroid Hyalosis",

    # Rare/misc
    "PT": "Parafoveal Telangiectasia",
    "PTCR": "Post-Traumatic Choroidal Rupture",
}

category_map = {
    "Diabetic retinopathy": ["DR", "CME", "EDN", "LS", "TV", "CL"],
    "AMD / degenerative": ["ARMD", "DN", "RPEC", "CRS", "CF", "TSLN", "MYA", "HPED", "MS"],
    "Macular & vitreoretinal": ["MHL", "ERM", "CSR", "RT", "CME", "MS"],
    "Vascular occlusions": ["BRVO", "CRVO", "BRAO", "CRAO", "MCA", "PLQ"],
    "Optic disc / neuro-ophthalmology": ["ODE", "ODC", "ODP", "AION", "ST", "TD", "ODPM", "PA"],
    "Hemorrhages / exudates": ["PRH", "VH", "EDN", "HR"],
    "Inflammatory / dystrophies": ["RS", "RP", "CWS", "CB", "MNF", "VS"],
    "Media / vitreous / other": ["MH", "AH", "PT", "PTCR"],
}

# ===================== UI PAGES =====================
def show_intro():
    st.title("Fundus Pathology Quiz")
    st.markdown(
        """
**An online interactive quiz tool for fundus photo evaluation training.**

This app lets you practice recognizing retinal pathologies on color fundus photographs using the RFMiD dataset,
an open-access set of **3,200** fundus images captured using **three** different fundus cameras with **46** conditions
annotated through adjudicated consensus of two senior retinal experts.
"""
    )

    # Row: small hero image + Start button beside it
    col_img, col_btn = st.columns([3, 1])
    with col_img:
        hero_url = get_hero_image_url()
        if hero_url:
            st.image(hero_url, width=320)
    with col_btn:
        st.write("")  # spacer
        st.write("")
        if st.button("Start Quiz", type="primary"):
            st.session_state.quiz_started = True
            st.rerun()

    st.markdown("### Credits & Reference")
    st.markdown(
        """
Developed by **Anders Hougaard, MD, PhD**, University of Copenhagen / Copenhagen University Hospital.

This quiz uses the **Retinal Fundus Multi-Disease Image Dataset (RFMiD)**.  
**Citation:** Pachade, S., Porwal, P., Thulkar, D., *et al.*  
*Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research.*  
**Data** 2021, 6(2), 14. <https://www.mdpi.com/2306-5729/6/2/14>
"""
    )

    # Logos row, kept separate, no captions, no overlap
    col_logo1, spacer, col_logo2 = st.columns([1, 0.1, 1])
    with col_logo1:
        st.image(
            "https://designguide.ku.dk/download/co-branding/ku_logo_uk_h.png",
            width=180,
        )
    with col_logo2:
        st.image(
            "https://www.regionh.dk/til-fagfolk/Om-Region-H/regionens-design/logo-og-grundelementer/logo-til-print-og-web/PublishingImages/Maerke_Hospital.jpg",
            width=120,
        )

    st.markdown("---")

def show_quiz():
    # ----- Always use ALL datasets combined -----
    df, pathology_cols = load_all_datasets()

    # ----- Sidebar: quiz setup -----
    st.sidebar.header("Quiz setup")
    pap_mode = st.sidebar.checkbox("Papilledema (yes/no) mode", value=False, help="Binary quiz: ODE vs Normal")

    default_categories = [
        "Diabetic retinopathy",
        "AMD / degenerative",
        "Vascular occlusions",
        "Optic disc / neuro-ophthalmology",
    ]
    selected_categories = st.sidebar.multiselect(
        "Select disease categories to include",
        options=list(category_map.keys()),
        default=default_categories,
        disabled=pap_mode
    )
    include_normals = st.sidebar.checkbox("Include normals (NL) in pool", value=False, disabled=pap_mode)

    st.sidebar.markdown("---")
    mc_mode = st.sidebar.checkbox("Multiple-choice mode (multi-label)", value=True, disabled=pap_mode)
    num_choices = st.sidebar.slider("Number of MC options", min_value=3, max_value=8, value=4, step=1, disabled=pap_mode)

    # ----- Build quiz pool depending on mode -----
    if pap_mode:
        if "ODE" not in df.columns:
            st.error("Label 'ODE' (papilledema) not found in labels CSV(s).")
            return

        ode_mask = df["ODE"] == 1
        nl_mask = df["NL"] == 1
        pool_mask = ode_mask | nl_mask
        df_quiz = df[pool_mask].copy()

        if df_quiz.empty:
            st.warning("No images found for Papilledema vs Normal.")
            return

        sig = current_filter_signature(mode="pap")
        if "filter_signature" not in st.session_state or st.session_state.filter_signature != sig:
            st.session_state.filter_signature = sig
            st.session_state.current_index = random.choice(df_quiz.index)
            st.session_state.score = 0
            st.session_state.attempts = 0
            st.session_state.revealed = False

        if st.session_state.current_index not in df_quiz.index:
            st.session_state.current_index = random.choice(df_quiz.index)
            st.session_state.revealed = False

        # ----- Current item -----
        row = df_quiz.loc[st.session_state.current_index]
        image_id = normalize_id(row["ID"])
        image_url = resolve_image_url(row["img_prefix"], image_id)

        try:
            buf = fetch_png(image_url)
            im = Image.open(buf)
        except Exception as e:
            st.error(f"Failed to load image from S3 URL:\n{image_url}\n\n{e}")
            return

        ds_name = row["dataset"]
        st.markdown(f"### 🖼️ Image ID: `{image_id}` • Dataset: **{ds_name}**")
        st.image(im, caption="Papilledema — Yes or No?", use_container_width=True)

        # Ground truth for pap-mode
        is_pap = bool(row.get("ODE", 0) == 1)
        row_positive = [c for c in pathology_cols if row.get(c, 0) == 1]
        correct_codes_all = sorted(row_positive)

        # Binary answer UI (side-by-side with Next)
        choice_key = f"pap_choice_{image_id}"
        user_choice = st.radio("Your answer:", ["Papilledema — Yes", "Papilledema — No"], index=None, key=choice_key)

        colA, colB, _ = st.columns([1, 1, 6])
        check_clicked = colA.button("Check", type="primary", key=f"check_pap_{image_id}")
        next_clicked  = colB.button("Next image", key=f"next_pap_{image_id}")

        if check_clicked:
            if user_choice is None:
                st.warning("Please select Yes or No.")
            else:
                st.session_state.attempts += 1
                user_is_pap = (user_choice == "Papilledema — Yes")
                if user_is_pap == is_pap:
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    st.error("Not quite.")
                st.session_state.revealed = True

        if next_clicked:
            st.session_state.current_index = random.choice(df_quiz.index)
            st.session_state.revealed = False
            st.rerun()

        if st.session_state.revealed:
            render_answer_pills(correct_codes_all)
            st.info(f"**Score:** {st.session_state.score} / {st.session_state.attempts}")
        return

    # ===== Normal multi-label / flashcard modes (pap_mode OFF) =====
    selected_codes = {code for cat in selected_categories for code in category_map.get(cat, [])}
    if include_normals:
        selected_codes.add("NL")
    selected_codes = sorted(selected_codes)

    if not selected_codes:
        st.warning("Please select at least one category or enable normals.")
        return

    present_cols = [c for c in selected_codes if c in df.columns]
    path_mask = (df[[c for c in present_cols if c != "NL"]].sum(axis=1) > 0) if present_cols else pd.Series(False, index=df.index)
    nl_mask = (df["NL"] == 1) if include_normals else pd.Series(False, index=df.index)
    pool_mask = path_mask | nl_mask
    df_quiz = df[pool_mask].copy()

    if df_quiz.empty:
        st.warning("No images match the current selection. Try adding categories or enabling normals.")
        return

    sig = current_filter_signature(
        mode="multi",
        selected_categories=selected_categories,
        include_normals=include_normals,
        mc_mode=mc_mode,
        num_choices=num_choices,
    )
    if "filter_signature" not in st.session_state or st.session_state.filter_signature != sig:
        st.session_state.filter_signature = sig
        st.session_state.current_index = random.choice(df_quiz.index)
        st.session_state.score = 0
        st.session_state.attempts = 0
        st.session_state.revealed = False

    if st.session_state.current_index not in df_quiz.index:
        st.session_state.current_index = random.choice(df_quiz.index)
        st.session_state.revealed = False

    # ----- Current item (S3 image load) -----
    row = df_quiz.loc[st.session_state.current_index]
    image_id = normalize_id(row["ID"])
    image_url = resolve_image_url(row["img_prefix"], image_id)

    try:
        buf = fetch_png(image_url)
        im = Image.open(buf)
    except Exception as e:
        st.error(f"Failed to load image from S3 URL:\n{image_url}\n\n{e}")
        return

    ds_name = row["dataset"]
    st.markdown(f"### 🖼️ Image ID: `{image_id}` • Dataset: **{ds_name}**")
    st.image(im, caption="Guess the pathology 👇", use_container_width=True)

    # Determine correct labels for this row (restricted to selected set + NL if chosen)
    row_positive = [c for c in pathology_cols if row.get(c, 0) == 1]
    correct_codes = [c for c in row_positive if c in selected_codes]
    if include_normals and row.get("NL", 0) == 1:
        correct_codes = ["NL"]

    if mc_mode:
        # Multiple-choice (multi-label)
        image_key = f"ALL:{image_id}"
        options_codes = build_mc_options(image_key, correct_codes, selected_codes, num_choices)
        label_options = [label_map.get(c, c) for c in options_codes]
        msel_key = f"msel_{image_id}_{num_choices}"
        st.write("Select **all** that apply, then press **Check**.")
        user_choice_labels = st.multiselect("Your selection:", label_options, default=[], key=msel_key)

        inv_map = {label_map.get(k, k): k for k in options_codes}
        user_codes = sorted({inv_map[lbl] for lbl in user_choice_labels if lbl in inv_map})

        # Buttons side-by-side
        colA, colB, _ = st.columns([1, 1, 6])
        check_clicked = colA.button("Check", type="primary", key=f"check_mc_{image_id}")
        next_clicked  = colB.button("Next image", key=f"next_mc_{image_id}")

        if check_clicked:
            st.session_state.attempts += 1
            is_correct = set(user_codes) == set(correct_codes)
            if is_correct:
                st.success("✅ Correct!")
                st.session_state.score += 1
            else:
                st.error("Not quite.")
            st.session_state.revealed = True

        if next_clicked:
            st.session_state.current_index = random.choice(df_quiz.index)
            st.session_state.revealed = False
            st.rerun()

        if st.session_state.revealed:
            render_answer_pills([c for c in correct_codes if c != "NL"])
            st.info(f"**Score:** {st.session_state.score} / {st.session_state.attempts}")

    else:
        # Flashcard mode
        st.write("**Your guess:** (Think before revealing)")
        colA, colB, _ = st.columns([1, 1, 6])
        reveal_clicked = colA.button("Reveal answer", type="primary", key=f"reveal_{image_id}")
        next_clicked   = colB.button("Next image", key=f"next_fc_{image_id}")

        if reveal_clicked:
            st.session_state.revealed = True

        if next_clicked:
            st.session_state.current_index = random.choice(df_quiz.index)
            st.session_state.revealed = False
            st.rerun()

        if st.session_state.revealed:
            render_answer_pills([c for c in correct_codes if c != "NL"])

# ===================== MAIN =====================
def main():
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if not st.session_state.quiz_started:
        show_intro()
    else:
        show_quiz()

if __name__ == "__main__":
    main()
