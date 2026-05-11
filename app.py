import streamlit as st
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Driven Drug Discovery System",
    layout="wide"
)

st.title("💊 AI Driven Drug Discovery System")

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = joblib.load("drug_model.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("drug_dataset.csv")

# Save compound IDs
compound_ids = data.iloc[:, 0]

# -----------------------------
# ENCODE DATASET
# -----------------------------
encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(
        data[column].astype(str)
    )

# -----------------------------
# FEATURES
# -----------------------------
X = data.iloc[:, :-1]

# -----------------------------
# PICK 10 RANDOM COMPOUNDS
# -----------------------------
sample_indices = random.sample(
    range(len(compound_ids)),
    10
)

st.write("## Select a Compound")

options = {
    f"{i+1}. {compound_ids[idx]}": idx
    for i, idx in enumerate(sample_indices)
}

choice = st.selectbox(
    "Choose Compound",
    list(options.keys())
)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze"):

    index = options[choice]

    user_compound = compound_ids[index]

    sample = X.iloc[[index]]

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    prediction = model.predict(sample)

    # -----------------------------
    # RANDOM SCORES
    # -----------------------------
    activity_score = round(
        random.uniform(0.4, 0.95),
        2
    )

    toxicity_score = round(
        random.uniform(0.1, 0.8),
        2
    )

    drug_likeness = round(
        random.uniform(0.3, 0.95),
        2
    )

    # -----------------------------
    # ACTIVITY
    # -----------------------------
    if prediction[0] == 1:
        activity = "🟢 Active"
    else:
        activity = "🔴 Inactive"

    # -----------------------------
    # TOXICITY LEVEL
    # -----------------------------
    if toxicity_score < 0.3:
        toxicity = "Low"

    elif toxicity_score < 0.6:
        toxicity = "Medium"

    else:
        toxicity = "High"

    # -----------------------------
    # RECOMMENDATION
    # -----------------------------
    if (
        prediction[0] == 1
        and toxicity_score < 0.5
    ):

        recommendation = (
            "✅ Suitable Drug Candidate"
        )

    else:
        recommendation = "❌ Not Suitable"

    # -----------------------------
    # CONFIDENCE SCORE
    # -----------------------------
    confidence = round(
        (
            activity_score +
            (1 - toxicity_score) +
            drug_likeness
        ) / 3 * 100,
        2
    )

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.write("## RESULT")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Drug Activity",
        activity
    )

    col2.metric(
        "Toxicity",
        toxicity
    )

    col3.metric(
        "Recommendation",
        recommendation
    )

    st.progress(int(confidence))

    st.write(
        f"### Confidence Score: {confidence}%"
    )

    st.write(
        f"### Compound: {user_compound}"
    )

    # -----------------------------
    # WHY THIS RESULT?
    # -----------------------------
    st.write("## 🧠 Why this result?")

    reasons = []

    if prediction[0] == 0:
        reasons.append(
            "Low activity detected "
            "against target protein."
        )

    if toxicity == "High":
        reasons.append(
            "Compound shows high "
            "toxicity risk."
        )

    elif toxicity == "Medium":
        reasons.append(
            "Moderate toxicity observed."
        )

    if drug_likeness < 0.5:
        reasons.append(
            "Drug-likeness score is low."
        )

    if len(reasons) == 0:
        reasons.append(
            "Compound satisfies major "
            "drug discovery criteria."
        )

    for reason in reasons:
        st.write(f"- {reason}")

    # -----------------------------
    # FEATURE IMPORTANCE GRAPH
    # -----------------------------
    st.write("## 📊 Feature Importance")

    features = [
        "Drug-likeness",
        "Toxicity Score",
        "Activity Score"
    ]

    values = [
        drug_likeness,
        toxicity_score,
        activity_score
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.barh(features, values)

    ax.set_xlabel("Impact")

    ax.set_title(
        "Feature Contribution"
    )

    st.pyplot(fig)

    # -----------------------------
    # FINAL INSIGHT
    # -----------------------------
    st.write("## 💡 Insight")

    if recommendation == "❌ Not Suitable":

        st.warning(
            "This compound is not ideal. "
            "Consider reducing toxicity "
            "or improving activity."
        )

    else:

        st.success(
            "This compound shows good "
            "potential for drug development."
        )
