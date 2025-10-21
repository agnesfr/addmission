import streamlit as st
import pickle
import pandas as pd

password_guess= st.text_input("Whats the password?", type="password")
if password_guess != st.secrets['password']:
    st.stop()
    

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

st.markdown("<h1 style='color: green;'>Graduate Admissions Prediction</h1>", unsafe_allow_html=True)

#add image text

st.image('admission.jpg',caption="Predict your chance of admission based on your profile", width = 400)

st.write("This app uses multiple inputs to predict the probability of admission to a graduate school.")
# Display an image of penguins

st.markdown("<h2 style='color: blue;'>Predicting Admission Chance...</h2>", unsafe_allow_html=True)

# Create a sidebar for input collection
st.sidebar.header('**Enter your profile details**')

gre_score= st.sidebar.number_input('GRE Score', min_value=0, max_value=340, value=300, step=1, help="Enter your GRE score (0–340).")
toefl= st.sidebar.number_input('TOEFL Score', min_value=0, max_value=120, value=100, step=1,help="Enter your TOEFL score (0–120)." )
cgpa= st.sidebar.number_input('CGPA', min_value=0.0, max_value=10.0, value=8.0, step=0.1, help="Enter your CGPA (0.0–10.0)." )
research_experience=st.sidebar.selectbox('Research Experience', ('No', 'Yes'), help="Select whether you have research experience.")
university_rating=st.sidebar.slider('University Rating', 1, 5, 3, help="Rate the university (1–5).")
sop = st.sidebar.slider('Statement of Purpose Strength', 1.0, 5.0, step=0.1, help="Rate the strength of your Statement of Purpose (1.0–5.0).")
lor = st.sidebar.slider('Letter of Recommendation Strength', 1.0, 5.0, step=0.1, help="Rate the strength of your Letter of Recommendation (1.0–5.0).")


if research_experience == 'Yes':
    Reachers_yes = 1
    Reachers_no=0
else:
    Reachers_yes = 0
    Reachers_no=1

dt_pickle = open('reg_admission.pickle', 'rb') 
model = pickle.load(dt_pickle) 
dt_pickle.close()
  # 1) Get expected names (robust across MAPIE / Pipeline / RF etc.)


if st.sidebar.button('Predict'):
    input_data = pd.DataFrame([[gre_score, toefl, university_rating, sop, lor, cgpa, Reachers_yes, Reachers_no]],
                            columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research_No', 'Research_Yes'])
    #get prediction and prediction intervals
    prediction = model.predict(input_data)
    #     display the prediction


    y_pred, y_pis = model.predict(input_data, alpha=0.1)
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]


    # --- CSS styling ---
    st.markdown("""
        <style>
        .result-card {
            background: white;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 10px;
            padding: 18px 20px;
            margin: 8px 0 20px 0;
        }
        
        .result-label {
            font-weight: 600;
            font-size: 0.95rem;
            color: #333;
            margin-bottom: 6px;
        }
        .result-value {
            font-size: 2.6rem;
            line-height: 1.1;
            margin: 0 0 8px 0;
            color: #2b2b2b;
        }
        .muted {
            color: #666;
            font-size: 0.95rem;
            margin: 6px 0 0 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- First box: Predicted Probability ---
    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Admission Probability</div>
            <div class="result-value">{prediction[0]*100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Second box: Confidence Interval ---
    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">With a {int(0.9*100)} % confidence level)</div>
            <div class="muted"><b>Prediction Interval:</b> [{lower*100}%, {upper*100}%]</div>
        </div>
    """, unsafe_allow_html=True)



# Showing additional items in tabs
st.markdown("<h3 style='color: #b30000;'>Model Insights</h3>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", " Predictive vs Actual","Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('Feature_Importance.png', caption="Feature Importance Plot"  )

with tab2:
    st.write("###  Histogram of Residuals")
    st.image('Residuals_Histogram.png', caption="Histogram of Residuals"  )

with tab3:
    st.write("### Predictive vs Actual")
    st.image('Predicted_vs_Actual.png', caption="Predictive vs Actual Plot"  )  
with tab4:
    st.write("### Coverage Plot")
    st.image('Prediction_Intervals.png', caption="Coverage Plot"    )

