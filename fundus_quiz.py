# fundus_quiz.py
import os
import io
import random
from typing import List

import requests
import pandas as pd
from PIL import Image
import streamlit as st

# ===================== CONFIG VIA SECRETS / ENV =====================
def _secret(key: str, default: str = "") -> str:
    """Prefer Streamlit secrets; fall back to environment variables."""
    try:
        return st.secrets.get(key, default)  # Streamlit Cloud
    except Exception:
        return os.getenv(key, default)       # local dev

S3_BUCKET = _secret("S3_BUCKET", "fundus-quiz")
S3_REGION = _secret("S3_REGION", "eu-north-1")
S3_PREFIX = _secret("S3_PREFIX", "RFMiD/Training")  # path inside bucket with PNGs
USE_S3 = _secret("USE_S3", "1") == "1"

# IMPORTANT: labels CSV must be reachable over HTTPS (e.g., S3 object URL)
LABELS_CSV_URL = _secret("LABELS_CSV_URL", "").strip()
# Optional small fallback if you ship a tiny demo CSV in the repo:
LOCAL_LABELS_FALLBACK = "RFMiD_Training_Labels.csv"

S3_BASE = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

# ===================== STREAMLIT PAGE CONFIG =====================
st.set_page_config(page_title="RFMiD Fundus Quiz", layout="wide")

# ===================== HELPERS =====================
def normalize_id(any_id) -> str:
    """
    Convert '0001' -> '1' and leave non-numeric IDs unchanged.
    Ensures no zero-padding in S3 URLs (keys are '1.png', not '0001.png').
    """
    s = str(any_id).strip()
    try:
        return str(int(s))
    except ValueError:
        return s

def resolve_image_url(image_id: str) -> str:
    """Build direct HTTPS URL to PNG in S3 (non-zero-padded)."""
    clean = normalize_id(image_id)
    return f"{S3_BASE}/{S3_PREFIX}/{clean}.png"

@st.cache_data(show_spinner=False)
def load_labels_any(url: str, local_fallback: str) -> pd.DataFrame:
    """
    Load labels from remote URL (S3/HTTP). If that fails and a local CSV exists,
    use it as a fallback (for small demos).
    """
    if url:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))
    if os.path.exists(local_fallback):
        return pd.read_csv(local_fallback)
    raise FileNotFoundError(
        "Could not load labels CSV. Set LABELS_CSV_URL in Secrets to your S3 HTTPS CSV, "
        "or include a small 'RFMiD_Training_Labels.csv' in the repo."
    )

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_png(url: str) -> io.BytesIO:
    """Download an image from S3 and cache it for an hour."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return io.BytesIO(r.content)

def current_filter_signature(selected_categories, include_normals, mc_mode, num_choices):
    return (tuple(sorted(selected_categories)), include_normals, mc_mode, num_choices)

# ===================== LABEL MAPS =====================
# Full RFMiD code → human label mapping
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
    "MH": "Media Haze",  # note: RFMiD 'MH' = Media Haze (Macular hole is 'MHL')
    "AH": "Asteroid Hyalosis",

    # Rare/misc
    "PT": "Parafoveal Telangiectasia",
    "PTCR": "Post-Traumatic Choroidal Rupture",
}

# Category groupings (edit to taste)
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

# ===================== LOAD LABELS + PREP DATA =====================
df = load_labels_any(LABELS_CSV_URL, LOCAL_LABELS_FALLBACK)

# Determine label columns (assume all except ID)
all_cols = df.columns.tolist()
pathology_cols: List[str] = [c for c in all_cols if c.lower() != "id"]

# Rows with ≥1 positive label (for pathology pool)
df["num_pathologies"] = df[pathology_cols].sum(axis=1)
df_nonempty = df[df["num_pathologies"] > 0].copy()

# ===================== UI PAGES =====================
def show_intro():
    st.title("Fundus Pathology Quiz (RFMiD)")
    st.markdown(
        """
**Credits & Reference**

This quiz uses the **Retinal Fundus Multi-Disease Image Dataset (RFMiD)**.

**Citation:**  
Pachade, S., Porwal, P., Thulkar, D., et al.  
*Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research.*  
**Data** 2021, 6(2), 14.  
<https://www.mdpi.com/2306-5729/6/2/14>

RFMiD contains 3,200 fundus images annotated for **46** disease/pathology categories by expert graders.

Use the **sidebar** to select disease categories, toggle **Multiple-choice mode**, and choose whether to **include normals (NL)**.  
Then click **Start Quiz**.
"""
    )
    if st.button("Start Quiz"):
        st.session_state.quiz_started = True

def show_quiz():
    # ----- Sidebar controls -----
    st.sidebar.header("Quiz setup")
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
    )
    include_normals = st.sidebar.checkbox("Include normals (NL) in pool", value=False)

    st.sidebar.markdown("---")
    mc_mode = st.sidebar.checkbox("Multiple-choice mode (multi-label)", value=True)
    num_choices = st.sidebar.slider("Number of MC options", min_value=3, max_value=8, value=4, step=1)

    # ----- Build selected codes -----
    selected_codes = {code for cat in selected_categories for code in category_map.get(cat, [])}
    if include_normals and "NL" in df.columns:
        selected_codes.add("NL")
    selected_codes = sorted(selected_codes)

    if not selected_codes:
        st.warning("Please select at least one category or enable normals.")
        return

    present_cols = [c for c in selected_codes if c in df.columns]

    # Build quiz pool: rows with ≥1 selected label OR NL==1 if include_normals
    path_mask = (df[present_cols].sum(axis=1) > 0) if present_cols else pd.Series(False, index=df.index)
    nl_mask = (df["NL"] == 1) if (include_normals and "NL" in df.columns) else pd.Series(False, index=df.index)
    pool_mask = path_mask | nl_mask
    df_quiz = df[pool_mask].copy()

    if df_quiz.empty:
        st.warning("No images match the current selection. Try adding categories or enabling normals.")
        return

    # ----- State reset if filters change -----
    sig = current_filter_signature(selected_categories, include_normals, mc_mode, num_choices)
    if "filter_signature" not in st.session_state or st.session_state.filter_signature != sig:
        st.session_state.filter_signature = sig
        st.session_state.current_index = random.choice(df_quiz.index)
        st.session_state.score = 0
        st.session_state.attempts = 0
        st.session_state.revealed = False

    # Safety: if index not in pool, reset
    if st.session_state.current_index not in df_quiz.index:
        st.session_state.current_index = random.choice(df_quiz.index)
        st.session_state.revealed = False

    # ----- Top bar -----
    col_top1, col_top2, col_top3 = st.columns([1, 2, 1])
    with col_top1:
        if st.button("Next image"):
            st.session_state.current_index = random.choice(df_quiz.index)
            st.session_state.revealed = False
    with col_top2:
        cat_label = ', '.join(selected_categories) if selected_categories else '—'
        st.write(f"**Pool size:** {len(df_quiz)}  |  **Categories:** {cat_label}"
                 f"{'  |  + Normals' if include_normals else ''}")
    with col_top3:
        if st.button("Reset score"):
            st.session_state.score = 0
            st.session_state.attempts = 0

    # ----- Current item (S3 image load) -----
    row = df_quiz.loc[st.session_state.current_index]
    image_id = normalize_id(row["ID"])
    image_url = resolve_image_url(image_id)

    try:
        buf = fetch_png(image_url)
        im = Image.open(buf)
    except Exception as e:
        st.error(f"Failed to load image from S3 URL:\n{image_url}\n\n{e}")
        return

    st.markdown(f"### 🖼️ Image ID: `{image_id}`")
    st.image(im, caption="Guess the pathology 👇", use_container_width=True)

    # ----- Determine correct labels for this row (restricted to selected set + NL if chosen) -----
    row_positive = [c for c in pathology_cols if c in df_quiz.columns and row.get(c, 0) == 1]
    correct_codes = [c for c in row_positive if c in selected_codes]

    if include_normals and "NL" in df.columns and row.get("NL", 0) == 1 and ("NL" in selected_codes):
        # If NL included and this is a normal case, use NL as the correct label
        correct_codes = ["NL"]

    # ----- Modes -----
    if mc_mode:
        # multi-label capable MC
        option_pool = set(correct_codes)
        distractor_pool = [c for c in selected_codes if c not in correct_codes]
        need = max(0, num_choices - len(option_pool))
        if need > 0 and distractor_pool:
            option_pool.update(random.sample(distractor_pool, min(need, len(distractor_pool))))
        options = sorted(option_pool)

        st.write("Select **all** that apply, then press **Check**.")
        label_options = [label_map.get(c, c) for c in options]
        user_choice_labels = st.multiselect("Your selection:", label_options, default=[])

        # Map back to codes for scoring
        inv_map = {label_map.get(k, k): k for k in options}
        user_codes = sorted({inv_map[lbl] for lbl in user_choice_labels if lbl in inv_map})

        if st.button("Check"):
            st.session_state.attempts += 1
            if set(user_codes) == set(correct_codes):
                st.success("✅ Correct!")
                st.session_state.score += 1
                st.session_state.revealed = True
            else:
                human_correct = ", ".join([label_map.get(c, c) for c in correct_codes]) if correct_codes else "Normal"
                st.error(f"Not quite. Correct: {human_correct}")
                st.session_state.revealed = True

        if st.session_state.revealed:
            st.info(f"**Score:** {st.session_state.score} / {st.session_state.attempts}")

    else:
        # Flashcard mode
        st.write("**Your guess:** (Think before revealing)")
        if st.button("Reveal answer"):
            st.session_state.revealed = True
        if st.session_state.revealed:
            if correct_codes:
                human = ", ".join([label_map.get(c, c) for c in correct_codes])
                st.success(human)
            else:
                st.info("Normal / No selected-category labels present.")

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
