import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Writing Score Prediction")
st.markdown("""
This application predicts a student's writing score based on their demographic and academic information.
Please fill out the form below and click 'Predict' to see the results.
""")

# Load the trained model
try:
    model = joblib.load('/home/mohamed/Desktop/AmitProject/Models/Model_1_Of_Writting.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the expected feature order (you may need to adjust this based on your model)
# This should match the order of features your model was trained on
expected_features = [
    'Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
    'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans',
    'WklyStudyHours', 'MathScore', 'ReadingScore'
]

# Define encoding mappings (you'll need to adjust these based on how your model was trained)
encoding_mappings = {
    'Gender': {'male': 0, 'female': 1},
    'EthnicGroup': {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4},
    'ParentEduc': {
        'some high school': 0, 
        'high school': 1, 
        'some college': 2, 
        'associate\'s degree': 3, 
        'bachelor\'s degree': 4, 
        'master\'s degree': 5
    },
    'LunchType': {'standard': 0, 'free/reduced': 1},
    'TestPrep': {'none': 0, 'completed': 1},
    'ParentMaritalStatus': {'married': 0, 'single': 1, 'divorced': 2, 'widowed': 3},
    'PracticeSport': {'never': 0, 'sometimes': 1, 'regularly': 2},
    'IsFirstChild': {'yes': 1, 'no': 0},
    'TransportMeans': {'private': 0, 'school_bus': 1, 'public': 2},
    'WklyStudyHours': {'< 5': 0, '5 - 10': 1, '> 10': 2}
}

def preprocess_input(input_dict):
    """Preprocess the input data to match the model's training format"""
    # Create a copy of the input dictionary
    processed = input_dict.copy()
    
    # Encode categorical variables
    for feature, mapping in encoding_mappings.items():
        if feature in processed:
            processed[feature] = mapping[processed[feature]]
    
    # Convert to DataFrame with the correct feature order
    processed_df = pd.DataFrame([processed])[expected_features]
    
    array_of_data = processed_df.to_numpy().reshape(1, -1)
    
    return array_of_data

def predict_writing_score(input_data):
    """Use the trained model to make predictions"""
    # Preprocess the input
    processed_data = preprocess_input(input_data)
    
    # Make prediction
    return model.predict(processed_data)[0]

# Create form for user input
with st.form("student_info_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Student Demographics")
        gender = st.selectbox("Gender", ["male", "female"])
        ethnic_group = st.selectbox("Ethnic Group", ["group A", "group B", "group C", "group D", "group E"])
        parent_educ = st.selectbox("Parent's Education", 
                                  ["some high school", "high school", "some college", 
                                   "associate's degree", "bachelor's degree", "master's degree"])
        lunch_type = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        test_prep = st.selectbox("Test Preparation", ["none", "completed"])
        
    with col2:
        st.subheader("Family & Study Information")
        parent_marital_status = st.selectbox("Parent Marital Status", ["married", "single", "divorced", "widowed"])
        practice_sport = st.selectbox("Practice Sport", ["never", "sometimes", "regularly"])
        is_first_child = st.selectbox("Is First Child", ["yes", "no"])
        transport_means = st.selectbox("Transportation Means", ["private", "school_bus", "public"])
        wkly_study_hours = st.selectbox("Weekly Study Hours", ["< 5", "5 - 10", "> 10"])
        
        st.subheader("Existing Scores")
        math_score = st.slider("Math Score", 0, 100, 70)
        reading_score = st.slider("Reading Score", 0, 100, 75)
    
    # Submit button
    submitted = st.form_submit_button("Predict Writing Score")

# When form is submitted
if submitted:
    # Create a DataFrame with the user input
    input_dict = {
        'Gender': gender,
        'EthnicGroup': ethnic_group,
        'ParentEduc': parent_educ,
        'LunchType': lunch_type,
        'TestPrep': test_prep,
        'ParentMaritalStatus': parent_marital_status,
        'PracticeSport': practice_sport,
        'IsFirstChild': is_first_child,
        'TransportMeans': transport_means,
        'WklyStudyHours': wkly_study_hours,
        'MathScore': math_score,
        'ReadingScore': reading_score
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Display the user input
    st.subheader("Student Information Summary")
    st.dataframe(input_df)
    
    # Show the encoded values (for debugging)
    with st.expander("View Encoded Values (For Debugging)"):
        encoded_df = preprocess_input(input_dict)
        st.dataframe(encoded_df)
    
    # Make prediction
    try:
        writing_score = predict_writing_score(input_dict)
        
        # Display prediction
        st.subheader("Prediction Result")
        st.metric(label="Predicted Writing Score", value=f"{writing_score:.2f}")
        
        # Additional visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Math Score", math_score)
        with col2:
            st.metric("Reading Score", reading_score)
        with col3:
            st.metric("Writing Score", f"{writing_score:.2f}")
            
        # Score comparison chart
        scores_data = pd.DataFrame({
            'Subject': ['Math', 'Reading', 'Writing'],
            'Score': [math_score, reading_score, writing_score]
        })
        
        st.bar_chart(scores_data.set_index('Subject'))
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check that the encoding mappings match how your model was trained.")

# Add some information about the model (optional)
with st.expander("About This Model"):
    st.markdown("""
    This prediction model was trained on student performance data with the following features:
    - **Demographics**: Gender, Ethnic Group
    - **Family Background**: Parent's Education, Parent Marital Status, Is First Child
    - **School Factors**: Lunch Type, Test Preparation, Transportation Means
    - **Study Habits**: Weekly Study Hours, Practice Sport
    - **Existing Scores**: Math Score, Reading Score
    
    The model predicts the writing score based on these input features.
    """)

# Footer
st.markdown("---")
st.markdown("© 2023 Student Performance Prediction Dashboard")