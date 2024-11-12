import streamlit as st
from sklearn import *
import joblib
import sys
sys.path.insert(1,r"C:\Users\LENOVO T480\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\streamlit_option_menu")
from streamlit_option_menu import option_menu
import pandas as pd
import google.generativeai as genai
# import pyttsx3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import os

clf = joblib.load(r'C:\Users\LENOVO T480\Desktop\zenith\model.joblib')
scaler = joblib.load(r'C:\Users\LENOVO T480\Desktop\zenith\scaler.joblib')
svc = joblib.load(r"C:\Users\LENOVO T480\Desktop\zenith\svc.joblib")

st.set_page_config(layout="wide")

# 'AIzaSyAooef3TMsvjrESXilW-P-CyawEsP6dhHs' 

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def pred_forms():
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=0, max_value=120, value=52, step=1)
            sex = st.selectbox('Sex (1 = male, 0 = female)', [1, 0], index=0)
            cp = st.number_input('Chest Pain Type (cp)', min_value=0, max_value=3, value=0, step=1)
            trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=125, step=5)
            chol = st.number_input('Cholesterol (chol)', min_value=0, max_value=600, value=212, step=5)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)', [1, 0], index=0)
        
        with col2:
            restecg = st.number_input('Resting Electrocardiographic Result (restecg)', min_value=0, max_value=2, value=1, step=1)
            thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=168, step=5)
            exang = st.selectbox('Exercise Induced Angina (1 = yes, 0 = no)', [1, 0], index=0)
            oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
            slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=2, step=1)
            ca = st.number_input('Number of Major Vessels (0-3) Colored by Fluoroscopy (ca)', min_value=0, max_value=3, value=0, step=1)
            thal = st.number_input('Thalassemia (thal)', min_value=0, max_value=3, value=2, step=1)

        # Submit button for the form
        submit = st.form_submit_button("Predict")

        if submit:
    # Define column names in the same order as when the scaler was trained
            columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            
            # Convert input data to a DataFrame with column names
            data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                thalach, exang, oldpeak, slope, ca, thal]], 
                                columns=columns)
            genai.configure(api_key = GOOGLE_API_KEY)
            mod = genai.GenerativeModel('gemini-1.5-pro')
            
            # transform with the scaler
            scaled_data = scaler.transform(data)
            prediction = svc.predict(scaled_data)

            if prediction[0] == 1:
                st.write("### The patient has a heart disease")
                diseased = 'how do i have to maintain my health to deal with my heart disease'
                with st.spinner("Wait for your health suggestion"):
                    response = mod.generate_content(diseased)
                st.markdown(response.text if hasattr(response, 'text') else "No response text")
                return

            elif prediction[0] == 0:
                st.write("### The patient does not have a heart disease")
                undiseased = "i don't have a heart disease but how can i improve my health"
                with st.spinner("Wait for your health suggestion"):
                    response = mod.generate_content(diseased)
                st.markdown(response.text if hasattr(response, 'text') else "No response text")
                return
        

        

def csv_chat_section():
    st.title("CSV Data Analysis Assistant")
    
    uploaded_file = os.getenv('HEART_CSV_FILE_PATH')
    
    # Initialize session state for storing the agent
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = None
    
      # Replace with your API key
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display basic information about the dataset
        # st.subheader("Dataset Overview")
        # from tabulate import tabulate
        # summary_data = {
        #     'Metric': ['Number of Rows', 'Number of Columns', 'Missing Values', 'Numeric Columns'],
        #     'Value': [
        #         len(df),
        #         len(df.columns),
        #         df.isnull().sum().sum(),
        #         len(df.select_dtypes(include=['float64', 'int64']).columns)
        #     ]
        # }
        # summary_df = pd.DataFrame(summary_data)
        # st.write(tabulate(summary_df, headers='keys', tablefmt='pretty', showindex=False))
        
        # Initialize the chat agent if not already done
        if st.session_state.chat_agent is None:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.7
                )
                
                # Create the agent without custom prompt
                st.session_state.chat_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True
                )
                
                st.success("Chat agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing chat agent: {str(e)}")
                return
        
        # Example questions
        with st.expander("See example questions you can ask"):
            st.write("""
            Statistical Analysis:
            - What is the minimum age in the dataset?
            - What is the maximum cholesterol level?
            - How many patients have heart disease (target=1)?
            - What is the average age of patients?
            
            Patient Demographics:
            - Show me the distribution of chest pain types
            - How many males and females are in the dataset?
            
            Risk Factor Analysis:
            - What's the average cholesterol level for patients with heart disease?
            - Is there a correlation between age and target?
            """)
        
        # Chat interface
        user_question = st.text_input(
            "Ask a question about your data:",
            key="csv_chat_input",
            placeholder="e.g., What is the minimum age in the dataset?"
        )
        
        if st.button("Ask", key="csv_chat_button"):
            if user_question:
                try:
                    with st.spinner("Analyzing your data..."):
                        response = st.session_state.chat_agent.run(user_question)
                    
                    # Display the response
                    st.markdown("### Analysis Results")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Try rephrasing your question or asking something simpler.")
            else:
                st.warning("Please enter a question about your data.")
    else:
        st.info("Please upload a CSV file to begin analysis.")      

def chat_section():
    st.title("Let's Talk - Medical Assistant")
    prompt = st.text_input("Tell me what's happening (medical queries only)")
    
    #genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # # Configure text-to-speech (TTS) enginE
    # engine = pyttsx3.init()
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id) 
    # engine.setProperty('rate', 200)

    # Define medical keywords for filtering
    medical_keywords = ["pain", "headache", "fever", "medicine", "doctor", "symptoms", "treatment", 
                        "diagnosis", "health", "cough", "infection", "injury", "allergy", "disease","cure",'acne','ill','sick']
    
    if st.button("Enter", key='chat_button'):
        # Check if the prompt contains any medical-related keywords
        if any(keyword in prompt.lower() for keyword in medical_keywords):
            # Generate and display response if the query is medical-related
            with st.spinner('Wait for the response'):
                response = model.generate_content(prompt)
            st.header(":blue[Response]")
            st.markdown(response.text if hasattr(response, 'text') else "No response text")
            
            # Read the response aloud
            # engine.say(response.text)
            # engine.runAndWait()
        else:
            # Display a message for non-medical queries
            st.write("Please ask a medical-related question.")
            return 

def main():
    st.title(":red[Heart Disease Prediction]")
    selected = option_menu(
        menu_title="", 
        options=["Prediction Report","Suggestions on your Health","Heart Disease Analysis"],
        orientation="horizontal",
        icons=["bar-chart", "chat-dots"],
    )
    if selected == "Prediction Report":
        pred_forms()
    elif selected == "Suggestions on your Health":
        chat_section()
    elif selected == "Heart Disease Analysis":
        csv_chat_section()
    

if __name__ == "__main__":
    main()
