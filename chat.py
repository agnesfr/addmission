import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.success("Graduate Admissions Prediction")

# Header image
st.image('admission.jpg',
         caption="Predict your chance of admission based on your profile",
         width=400)

st.write("This app uses multiple inputs to predict the probability of admission to a graduate school.")

# ========== Sidebar inputs ==========
st.sidebar.header('**Enter your profile details**')

gre_score = st.sidebar.number_input('GRE Score', min_value=0, max_value=340, value=300, step=1)
toefl = st.sidebar.number_input('TOEFL Score', min_value=0, max_value=120, value=100, step=1)
cgpa = st.sidebar.number_input('CGPA', min_value=0.0, max_value=10.0, value=8.0, step=0.01)
research_experience = st.sidebar.selectbox('Research Experience', ('No', 'Yes'))
university_rating = st.sidebar.slider('University Rating', 1, 5, 3)
sop = st.sidebar.slider('Statement of Purpose Strength', 1, 5, 3)
lor = st.sidebar.slider('Letter of Recommendation Strength', 1, 5, 3)

confidence = st.sidebar.selectbox("Confidence level", ["90%", "95%", "99%"], index=1)
Z_LOOKUP = {"90%": 1.645, "95%": 1.96, "99%": 2.576}
z = Z_LOOKUP[confidence]

# ========== Helpers ==========
def to_yes_no_dummies(val):
    """Return tuple (Research_No, Research_Yes) from raw input."""
    if isinstance(val, str):
        v = val.strip().lower()
        is_yes = v in ("yes", "y", "1", "true", "t")
    else:
        is_yes = bool(val)
    return int(not is_yes), int(is_yes)

def extract_feature_names(m):
    """Try to pull feature_names_in_ from common wrapper locations."""
    paths = [
        "feature_names_in_",
        "estimator_.feature_names_in_",
        "single_estimator_.feature_names_in_",
        "estimator_.single_estimator_.feature_names_in_",
    ]
    for path in paths:
        try:
            obj = m
            for part in path.split("."):
                obj = getattr(obj, part)
            if obj is not None:
                return list(obj)
        except Exception:
            continue
    return None

def try_mapie_predict_with_pi(m, X, conf_label):
    """
    If the model is MAPIE (or compatible), use its conformal intervals.
    Returns (pred, lower, upper) or None if not available.
    """
    try:
        # Convert confidence (e.g., 95%) to alpha (e.g., 0.05)
        alpha = 1.0 - float(conf_label.strip("%")) / 100.0
        # MAPIE signature: y_pred, y_pis = model.predict(X, alpha=alpha)
        out = m.predict(X, alpha=alpha)
        if isinstance(out, tuple) and len(out) == 2:
            y_pred, y_pis = out
            # y_pis shape: (n_samples, 2, n_alphas)
            if y_pis.ndim == 3 and y_pis.shape[1] == 2:
                lower = y_pis[:, 0, 0]
                upper = y_pis[:, 1, 0]
                return y_pred, lower, upper
        return None
    except TypeError:
        # Some models accept alpha as list
        try:
            alpha = [1.0 - float(conf_label.strip("%")) / 100.0]
            y_pred, y_pis = m.predict(X, alpha=alpha)
            lower = y_pis[:, 0, 0]
            upper = y_pis[:, 1, 0]
            return y_pred, lower, upper
        except Exception:
            return None
    except Exception:
        return None

# ========== Predict button ==========
if st.sidebar.button('Predict'):
    # ---- Load pickle once (support plain model OR dict with model + sigma) ----
    with open('reg_admission.pickle', 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        sigma = loaded.get("sigma", None)
    else:
        model = loaded
        sigma = None  # not provided; we'll only have PI if MAPIE works

    # ---- Build input row ----
    r_no, r_yes = to_yes_no_dummies(research_experience)
    row = {
        'GRE Score': gre_score,
        'TOEFL Score': toefl,
        'University Rating': university_rating,
        'SOP': sop,
        'LOR': lor,
        'CGPA': cgpa,
        # dummies expected at train time:
        'Research_No': r_no,
        'Research_Yes': r_yes,
        # keep raw column too in case the model actually expects it
        'Research Experience': research_experience,
    }
    X = pd.DataFrame([row])

    # ---- Align columns to the model's expected features (names & order) ----
    expected = extract_feature_names(model)
    if expected is not None:
        X = X.reindex(columns=expected, fill_value=0)

    # ---- Predict ----
    try:
        # Prefer MAPIE intervals if available
        mapie_result = try_mapie_predict_with_pi(model, X, confidence)
        if mapie_result is not None:
            y_pred, lower, upper = mapie_result
            pred_val = float(y_pred[0])
            lb, ub = float(lower[0]), float(upper[0])
            used_pi = f"{confidence} conformal prediction interval (MAPIE)"
        else:
            # Fall back to plain predict
            y_pred = model.predict(X)
            pred_val = float(y_pred[0])

            # If sigma available, form symmetric interval with Z
            if sigma is not None:
                lb = pred_val - z * float(sigma)
                ub = pred_val + z * float(sigma)
                used_pi = f"{confidence} interval using residual σ"
            else:
                lb = ub = None
                used_pi = None

        # ---- Show results ----
        st.subheader('Prediction Results')
        st.write(f'Based on the provided profile details, the predicted chance of admission is: **{pred_val:.2f}**')
        if pred_val>0.8:
            st.balloons()

        if used_pi and (lb is not None) and (ub is not None):
            st.info(f'{used_pi}: **[{lb:.2f}, {ub:.2f}]**')
        elif sigma is None:
            st.warning("No interval available: your pickle did not include σ and the model does not provide intervals. "
                       "Save `sigma` during training or use MAPIE for intervals.")
        else:
            st.warning("Could not compute an interval with the current model and settings.")

        if sigma is not None:
            st.caption(f"Residual standard deviation (σ): {float(sigma):.4f}")

    except Exception as e:
        st.error(f"Prediction failed. Most common cause is feature mismatch. Details:\n\n{e}")

    # ---- Insights section (unchanged images) ----
    st.subheader("Model Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predictive vs Actual", "Coverage Plot"])
    with tab1:
        st.write("### Feature Importance")
        st.image('Feature_Importance.png')
    with tab2:
        st.write("### Histogram of Residuals")
        st.image('Residuals_Histogram.png')
    with tab3:
        st.write("### Predictive vs Actual")
        st.image('Predicted_vs_Actual.png')
    with tab4:
        st.write("### Coverage Plot")
        st.image('Prediction_Intervals.png')
