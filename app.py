import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Math Score Predictor",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "math_score_model.joblib"
)


# ============================================================
# PRETTY LABELS
# ============================================================

PRETTY_LABEL = {
    "gender": "Gender",
    "race_ethnicity": "Race/Ethnicity",
    "parental_level_of_education": "Parental Education",
    "lunch": "Lunch",
    "test_preparation_course": "Test Preparation",
    "reading_score": "Reading Score",
    "writing_score": "Writing Score",
}


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CAT_FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]


# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Model file was not found:

{MODEL_PATH}

Please run model_train.py first.
"""
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:

    model_data = load_model()

except Exception as exc:

    st.error("❌ Could not load the saved model.")

    st.code(str(exc))

    st.info(
        """
Expected project structure:

project/
│
├── app.py
│
├── models/
│   └── math_score_model.joblib
│
├── data/
│   └── stud.csv
│
└── src/
    └── model_train.py
"""
    )

    st.stop()


# ============================================================
# EXTRACT SAVED OBJECTS
# ============================================================

try:

    preprocessor = model_data["preprocessor"]

    final_model = model_data["model"]

    feature_names = model_data["feature_names"]

    df = model_data["df"]

except KeyError as exc:

    st.error(
        f"❌ Missing key in saved model: {exc}"
    )

    st.info(
        """
Your joblib file should contain:

- preprocessor
- model
- feature_names
- df
- results (optional)
"""
    )

    st.stop()


# ============================================================
# OPTIONAL MODEL COMPARISON RESULTS
# ============================================================

results = model_data.get("results", None)


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Student Math Score Predictor")

st.markdown(
    """
Predict a student's **math score** using the trained
Linear Regression model.

The model and preprocessing pipeline are loaded directly
from the saved `.joblib` file.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎓 Student Information")


# ============================================================
# INPUTS
# ============================================================

gender = st.sidebar.selectbox(
    "Gender",
    options=[
        "female",
        "male",
    ],
)


race_ethnicity = st.sidebar.selectbox(
    "Race/Ethnicity",
    options=[
        "group A",
        "group B",
        "group C",
        "group D",
        "group E",
    ],
)


parental_level_of_education = st.sidebar.selectbox(
    "Parental Level of Education",
    options=[
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
)


lunch = st.sidebar.selectbox(
    "Lunch",
    options=[
        "standard",
        "free/reduced",
    ],
)


test_preparation_course = st.sidebar.selectbox(
    "Test Preparation Course",
    options=[
        "none",
        "completed",
    ],
)


reading_score = st.sidebar.slider(
    "Reading Score",
    min_value=0,
    max_value=100,
    value=70,
)


writing_score = st.sidebar.slider(
    "Writing Score",
    min_value=0,
    max_value=100,
    value=70,
)


# ============================================================
# CREATE INPUT PROFILE
# ============================================================

profile = {
    "gender": gender,
    "race_ethnicity": race_ethnicity,
    "parental_level_of_education":
        parental_level_of_education,
    "lunch": lunch,
    "test_preparation_course":
        test_preparation_course,
    "reading_score": reading_score,
    "writing_score": writing_score,
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_score(profile):

    # Convert dictionary into DataFrame
    row = pd.DataFrame([profile])

    # Apply saved preprocessing
    transformed = preprocessor.transform(row)

    # Convert sparse matrix to numpy array
    if hasattr(transformed, "toarray"):

        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    # Prediction
    raw_score = float(
        final_model.predict(transformed)[0]
    )

    # Keep prediction inside realistic score range
    score = np.clip(
        raw_score,
        0,
        100
    )

    # ========================================================
    # FEATURE CONTRIBUTIONS
    # ========================================================

    terms = []

    if hasattr(final_model, "coef_"):

        coefficients = np.asarray(
            final_model.coef_
        ).flatten()

        active_values = transformed[0]

        contributions = (
            coefficients
            * active_values
        )

        for (
            name,
            value,
            active
        ) in zip(
            feature_names,
            contributions,
            active_values
        ):

            # Ignore inactive one-hot features
            if (
                not name.startswith("numeric__")
                and abs(active) < 1e-9
            ):
                continue

            # Remove transformer prefix
            clean_name = (
                name
                .replace(
                    "numeric__",
                    ""
                )
                .replace(
                    "categorical__",
                    ""
                )
            )

            # Numeric feature
            if name.startswith(
                "numeric__"
            ):

                label = PRETTY_LABEL.get(
                    clean_name,
                    clean_name
                )

            # Categorical feature
            else:

                label = clean_name

                for category in CAT_FEATURES:

                    prefix = category + "_"

                    if clean_name.startswith(
                        prefix
                    ):

                        category_value = (
                            clean_name[
                                len(prefix):
                            ]
                        )

                        label = (
                            f"{PRETTY_LABEL.get(category, category)}: "
                            f"{category_value}"
                        )

                        break

            terms.append(
                (
                    label,
                    float(value)
                )
            )

    return score, terms


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🎯 Predict Math Score",
    use_container_width=True,
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        raw_score, terms = predict_score(
            profile
        )

    except Exception as exc:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(exc))

        st.stop()


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    st.subheader(
        "🎯 Predicted Math Score"
    )

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    with col1:

        st.metric(
            "Predicted Score",
            f"{raw_score:.2f}/100"
        )


    # --------------------------------------------------------
    # PERFORMANCE CATEGORY
    # --------------------------------------------------------

    if raw_score >= 80:

        category = "Excellent 🏆"

    elif raw_score >= 70:

        category = "Good 👍"

    elif raw_score >= 60:

        category = "Average 🙂"

    elif raw_score >= 50:

        category = "Needs Improvement ⚠️"

    else:

        category = "Poor ❌"


    with col2:

        st.metric(
            "Performance",
            category
        )


    # --------------------------------------------------------
    # PERCENTILE
    # --------------------------------------------------------

    try:

        mean_score = float(
            df["math_score"].mean()
        )

        std_score = float(
            df["math_score"].std()
        )

        if std_score > 0:

            z_score = (
                raw_score - mean_score
            ) / std_score

        else:

            z_score = 0

        percentile = (
            0.5
            * (
                1
                + math.erf(
                    z_score
                    / math.sqrt(2)
                )
            )
            * 100
        )

        percentile = np.clip(
            percentile,
            0,
            100
        )

    except Exception:

        percentile = np.nan


    with col3:

        if not np.isnan(percentile):

            st.metric(
                "Estimated Percentile",
                f"{percentile:.1f}%"
            )

        else:

            st.metric(
                "Estimated Percentile",
                "N/A"
            )


    # ========================================================
    # PROGRESS BAR
    # ========================================================

    st.progress(
        int(raw_score)
    )


    # ========================================================
    # STUDENT PROFILE
    # ========================================================

    st.subheader(
        "👤 Student Profile"
    )

    profile_df = pd.DataFrame(
        {
            "Feature": [
                "Gender",
                "Race/Ethnicity",
                "Parental Education",
                "Lunch",
                "Test Preparation",
                "Reading Score",
                "Writing Score",
            ],
            "Value": [
                gender,
                race_ethnicity,
                parental_level_of_education,
                lunch,
                test_preparation_course,
                reading_score,
                writing_score,
            ],
        }
    )

    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # FEATURE CONTRIBUTIONS
    # ========================================================

    st.subheader(
        "📊 Prediction Explanation"
    )

    if terms:

        terms_df = pd.DataFrame(
            terms,
            columns=[
                "Feature",
                "Contribution"
            ],
        )

        terms_df = terms_df.sort_values(
            "Contribution",
            ascending=False
        )

        # --------------------------------------------
        # Positive / Negative
        # --------------------------------------------

        positive_terms = terms_df[
            terms_df["Contribution"] > 0
        ]

        negative_terms = terms_df[
            terms_df["Contribution"] < 0
        ]


        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                "### 📈 Positive Factors"
            )

            if len(positive_terms) > 0:

                st.dataframe(
                    positive_terms,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No positive contributions."
                )


        with col2:

            st.markdown(
                "### 📉 Negative Factors"
            )

            if len(negative_terms) > 0:

                st.dataframe(
                    negative_terms,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No negative contributions."
                )


        # ====================================================
        # WATERFALL CHART
        # ====================================================

        st.subheader(
            "🌊 Prediction Contribution Waterfall"
        )

        sorted_terms = sorted(
            terms,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Limit chart to top features
        sorted_terms = sorted_terms[:10]

        labels = [
            "Intercept"
        ]

        values = [
            float(
                final_model.intercept_
            )
        ]

        for label, value in sorted_terms:

            labels.append(label)
            values.append(value)

        labels.append(
            "Final Prediction"
        )

        values.append(0)


        measures = (
            ["absolute"]
            + ["relative"] * len(
                sorted_terms
            )
            + ["total"]
        )


        fig = go.Figure(
            go.Waterfall(
                name="Prediction",
                orientation="v",
                measure=measures,
                x=labels,
                y=values,
                connector={
                    "line": {
                        "width": 1
                    }
                },
            )
        )


        fig.update_layout(
            title="How Features Influence the Prediction",
            xaxis_title="Feature",
            yaxis_title="Score Contribution",
            height=600,
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # READING / WRITING COMPARISON
    # ========================================================

    st.subheader(
        "📚 Subject Score Comparison"
    )

    subject_df = pd.DataFrame(
        {
            "Subject": [
                "Math",
                "Reading",
                "Writing",
            ],
            "Score": [
                raw_score,
                reading_score,
                writing_score,
            ],
        }
    )


    fig_subject = px.bar(
        subject_df,
        x="Subject",
        y="Score",
        range_y=[0, 100],
        text="Score",
        title="Student Subject Scores",
    )


    fig_subject.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )


    st.plotly_chart(
        fig_subject,
        use_container_width=True
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader(
        "🤖 Machine Learning Model Comparison"
    )


    if results is not None:

        try:

            # Make a copy so we don't modify
            # the original saved object
            results_display = results.copy()

            st.dataframe(
                results_display,
                use_container_width=True,
                hide_index=True,
            )


            # --------------------------------------------
            # Detect model and R2 columns
            # --------------------------------------------

            model_column = None

            possible_model_columns = [
                "Model",
                "model",
                "Model Name",
                "model_name",
            ]

            for column in possible_model_columns:

                if column in results_display.columns:

                    model_column = column
                    break


            r2_column = None

            possible_r2_columns = [
                "R2",
                "r2",
                "R2 Score",
                "r2_score",
            ]

            for column in possible_r2_columns:

                if column in results_display.columns:

                    r2_column = column
                    break


            if (
                model_column is not None
                and r2_column is not None
            ):

                fig_models = px.bar(
                    results_display,
                    x=model_column,
                    y=r2_column,
                    text=r2_column,
                    title="R² Score Comparison",
                )


                fig_models.update_traces(
                    textposition="outside"
                )


                fig_models.update_layout(
                    xaxis_title="Model",
                    yaxis_title="R² Score",
                )


                st.plotly_chart(
                    fig_models,
                    use_container_width=True
                )


        except Exception as exc:

            st.warning(
                "Could not display model comparison."
            )

            st.code(str(exc))


    else:

        st.info(
            """
Model comparison results were not saved in the
`.joblib` file.

If you want the comparison table, save your
`model_result_df` as `results` during training.
"""
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "ℹ️ Model Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            type(final_model).__name__
        )


    with col2:

        st.metric(
            "Features",
            len(feature_names)
        )


    with col3:

        st.metric(
            "Training Samples",
            len(df)
        )


# ============================================================
# INITIAL PAGE MESSAGE
# ============================================================

else:

    st.info(
        """
👈 Enter the student's information from the sidebar
and click **Predict Math Score**.
"""
    )


    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader(
        "📊 Dataset Overview"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Students",
            len(df)
        )


    with col2:

        st.metric(
            "Features",
            len(df.columns)
        )


    with col3:

        st.metric(
            "Average Math Score",
            f"{df['math_score'].mean():.2f}"
        )


    with col4:

        st.metric(
            "Maximum Math Score",
            f"{df['math_score'].max():.0f}"
        )


    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    if "math_score" in df.columns:

        fig_distribution = px.histogram(
            df,
            x="math_score",
            nbins=20,
            title="Math Score Distribution",
        )


        fig_distribution.update_layout(
            xaxis_title="Math Score",
            yaxis_title="Number of Students",
        )


        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )