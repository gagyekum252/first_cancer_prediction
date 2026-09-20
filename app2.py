# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cancer Risk Predictor", page_icon=":guardsman:", layout="centered")

# --- Load artifacts once on start ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_xgb_class_weighted.pkl")
    le = joblib.load("label_encoder.pkl")          # LabelEncoder fitted on y_train
    feature_names = joblib.load("feature_names.pkl")
    return model, le, feature_names

model, le, feature_names = load_artifacts()

st.title("Cancer Risk Level Predictor")
st.markdown("This application predicts the 'risk_level'(Low /Medium /High) from patient features.")

# --- Upload CSV or manual input ---
option = st.radio("Choose input method:", ("Upload CSV (batch)", "Manual Input(single)"))

def preprocess_input(df):
    """"
    Ensure DF has columns in Feature_names order and numeric dtype.
    Fills missing columns with 0 and reorders.
    """
    # if uploaded csv contains extra columns, keep only the feature columns
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in input data - filling {len(missing)} missing columns with 0: {missing}")
        for c in missing:
            df[c] = 0
    # Keep only the required columns and in same order as feature_names
    df = df[feature_names].copy()
    # Convert to numeric, coerce errors to NaN, then fill NaN with 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

# Keep only the required columns and in same order as feature_names
if option == "Upload CSV (batch)":
    uploaded_file = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        X = preprocess_input(input_df)
        preds_enc = model.predict(X)
        probs = model.predict_proba(X)
        preds = le.inverse_transform(preds_enc)
        results = X.copy()
        results["predicted_risk_level"] = preds
        
        # attach probabilities for each class
        for i, cls in enumerate(le.classes_):
            results[f"prob_{cls}"] = probs[:, i]
        st.success('Prediction completed!')
        st.dataframe(results)
        st.download_button("Download Results as CSV", results.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

    else:
        st.sidebar.header("Patient features (manual)")
        input_data = {}
        # Create numeric inputs for each feature (you may want to group or change ranges)
        for feat in feature_names:
             # heuristic: use 0 as default; change for age/bmi etc. if needed
            val = st.sidebar.number_input(feat, value=float(0.0))
            input_data[feat] = val

        if st.sidebar.button("Predict Risk Level"):
            x_single = pd.DataFrame([input_data])
            x_proc = preprocess_input(x_single)
            preds_enc = model.predict(x_proc)
            probs = model.predict_proba(x_proc)
            pred = le.inverse_transform(preds_enc)[0]

            st.write("### Prediction")
            st.write(f"**Predicted Risk Level:** {pred}")
            st.write("**Class Probabilities:**")
            prob_df = pd.DataFrame({'class': list(le.classes_), 'probability': probs.flatten()}).sort_values('probability', ascending=False).reset_index(drop=True)
            st.table(prob_df)

            # optional: hight 'High' probability and thresholds advice
            high_prob = prob_df.loc[prob_df['class'] == 'High', 'probability'].values[0]
            st.info(f"Probability of High Risk Level: {high_prob:.2f}")
            if high_prob > 0.5:
                st.warning("High probability >= 0.5 - consider clinical follow-up.")
            else:
                st.success("Low probability of High Risk Level - routine monitoring advised.")


# Footer
st.markdown("---")
st.caption("Model: class-weighted XGBoost (tuned via Optuna). Ensure uploaded CSV has the same feature columns as used in training. Missing features will be filled with 0.")   
                

