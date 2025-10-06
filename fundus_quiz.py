# app.py
import os
import random
from typing import List

import pandas as pd
from PIL import Image
import streamlit as st

# ============ CONFIG (edit these paths to your local RFMiD data) ============
IMAGE_DIR = "/Users/andershougaard/Documents/Training_Set/Training"
LABELS_CSV = "/Users/andershougaard/Documents/Training_Set/RFMiD_Training_Labels.csv"

# ============ LOAD DATA ============
@st.cache_data(show_spinner=False)
def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

st.set_page_config(page_title="RFMiD Fundus Quiz", layout="wide")
df = load_labels(LABELS_CSV)

# Identify pathology columns (assume all except ID are labels)
all_cols = df.columns.tolist()
pathology_cols: List[str] = [c for c in all_cols if c.lower() != "id"]

# Precompute rows with at least one positive label (non-empty pathology)
df["num_pathologies"] = df[pathology_cols].sum(axis=1)
df_nonempty = df[df["num_pathologies"] > 0].copy()

# ============ FULL RFMiD LABEL MAP (46 codes) ============
label_map = {
    # Normal / misc (kept for completeness; selectable via "Include normals")
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
    "MH": "Media Haze",  # Note: in RFMiD, MH = Media Haze (Macular hole is MHL)
    "AH": "Asteroid Hyalosis",

    # Misc rare
    "PT": "Parafoveal Telangiectasia",
    "PTCR": "Post-Traumatic Choroidal Rupture",
}

# ============ CATEGORY MAP (edit as you prefer) ============
category_map = {
    "Diabetic retinopathy": ["DR", "CME", "EDN", "LS", "TV", "CL"],
    "AMD / degenerative": ["ARMD", "DN", "RPEC", "CRS", "CF", "TSLN", "MYA", "HPED", "MS"],
    "Macular & vitreoretinal": ["MHL", "ERM", "CSR", "RT", "CME", "MS"],
    "Vascular occlusions": ["BRVO", "CRVO", "BRAO", "CRAO", "MCA", "PLQ"],
    "Optic disc / neuro-ophthalmology": ["ODE", "ODC", "ODP", "AION", "ST", "TD", "ODPM", "PA"],
    "Hemorrhages / exudates": ["PRH", "VH", "EDN", "HR"],
    "Inflammatory / dystrophies": ["RS", "RP", "CWS", "CB", "MNF", "VS"],
    "Media / vitreous / other": ["MH", "AH", "PT", "PTCR"],
    # Optionally: "Normals / other": ["NL", "OTHER"],  # normals controlled separately below
}

# ============ HELPERS ============
def resolve_image_path(image_dir: str, image_id: str):
    """Try common extensions to find the actual file for an ID."""
    candidates = [
        os.path.join(image_dir, f"{image_id}.png"),
        os.path.join(image_dir, f"{image_id}.jpg"),
        os.path.join(image_dir, f"{image_id}.jpeg"),
        os.path.join(image_dir, f"{image_id}.JPG"),
        os.path.join(image_dir, f"{image_id}.PNG"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def current_filter_signature(selected_categories, include_normals, mc_mode, num_choices):
    return (tuple(sorted(selected_categories)), include_normals, mc_mode, num_choices)

# ============ UI PAGES ============
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
[https://www.mdpi.com/2306-5729/6/2/14](https://www.mdpi.com/2306-5729/6/2/14)

RFMiD contains 3,200 fundus images annotated for **46** disease/pathology categories by expert graders.

Use the **sidebar** to select which disease categories to include, toggle **Multiple-choice mode**, and choose whether to **include normals (NL)**. Then click **Start Quiz**.
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
        default=default_categories
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

    # Codes present in CSV:
    present_cols = [c for c in selected_codes if c in df.columns]
    if not present_cols and not include_normals:
        st.error("Selected codes are not present in your CSV. Check your RFMiD file or category map.")
        return

    # ----- Build quiz pool (rows with ≥1 selected label; or NL==1 if normals included) -----
    path_mask = (df[present_cols].sum(axis=1) > 0) if present_cols else pd.Series(False, index=df.index)
    nl_mask = (df["NL"] == 1) if (include_normals and "NL" in df.columns) else pd.Series(False, index=df.index)
    pool_mask = path_mask | nl_mask
    df_quiz = df[pool_mask].copy()

    if df_quiz.empty:
        st.warning("No images match the current selection. Try adding categories or enabling normals.")
        return

    # ----- Reset state when filters change -----
    sig = current_filter_signature(selected_categories, include_normals, mc_mode, num_choices)
    if "filter_signature" not in st.session_state or st.session_state.filter_signature != sig:
        st.session_state.filter_signature = sig
        st.session_state.current_index = random.choice(df_quiz.index)
        st.session_state.score = 0
        st.session_state.attempts = 0
        st.session_state.revealed = False

    # Safety reset if current index not in pool
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
        st.write(f"**Pool size:** {len(df_quiz)}  |  **Categories:** {', '.join(selected_categories) or '—'}"
                 f"{'  |  + Normals' if include_normals else ''}")
    with col_top3:
        if st.button("Reset score"):
            st.session_state.score = 0
            st.session_state.attempts = 0

    # ----- Current image -----
    row = df_quiz.loc[st.session_state.current_index]
    image_id = str(row["ID"])
    img_path = resolve_image_path(IMAGE_DIR, image_id)
    if not img_path:
        st.error(f"Image file not found for ID={image_id} in {IMAGE_DIR}")
        return

    st.markdown(f"### 🖼️ Image ID: `{image_id}`")
    st.image(Image.open(img_path), caption="Guess the pathology 👇", use_container_width=True)

    # ----- Determine correct labels for this item (limited to selected set + NL if included) -----
    # Gather positives from all pathology cols present in df for this row:
    row_positive = [c for c in pathology_cols if c in df_quiz.columns and row.get(c, 0) == 1]
    # Restrict to our selected set (incl. NL if chosen):
    correct_codes = [c for c in row_positive if c in selected_codes]

    # If no selected labels present but it's NL and included_normals, define correct as NL
    if include_normals and "NL" in df.columns and row.get("NL", 0) == 1 and ("NL" in selected_codes):
        correct_codes = ["NL"]

    # ----- Modes -----
    if mc_mode:
        # Multiple-choice: multi-select (some images have multiple correct labels)
        # Build option pool from selected codes: ensure all correct present + sample distractors
        option_pool = set(correct_codes)
        distractor_pool = [c for c in selected_codes if c not in correct_codes]
        # number of distractors to add
        need = max(0, num_choices - len(option_pool))
        if need > 0 and distractor_pool:
            option_pool.update(random.sample(distractor_pool, min(need, len(distractor_pool))))
        options = sorted(option_pool)

        # Human-readable labels for UI
        label_options = [label_map.get(c, c) for c in options]
        st.write("Select **all** that apply, then press **Check**.")
        user_choice_labels = st.multiselect("Your selection:", label_options, default=[])

        # Convert back to codes for scoring
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
        # Flashcard mode (reveal)
        st.write("**Your guess:** (Think before revealing)")
        if st.button("Reveal answer"):
            st.session_state.revealed = True
        if st.session_state.revealed:
            if correct_codes:
                human = ", ".join([label_map.get(c, c) for c in correct_codes])
                st.success(human)
            else:
                st.info("Normal / No selected-category labels present.")

def main():
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False

    if not st.session_state.quiz_started:
        show_intro()
    else:
        show_quiz()

if __name__ == "__main__":
    main()
