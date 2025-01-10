import os
import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

def init_firebase():
    """Initialize Firebase if not already initialized"""
    try:
        if not firebase_admin._apps:
            # Try to load from streamlit secrets
            try:
                firebase_credentials = json.loads(st.secrets["firebase"]["credentials"])
                print("Successfully loaded credentials from Streamlit secrets")
            except Exception as e:
                print(f"Failed to load from Streamlit secrets: {str(e)}")
                # If that fails, try loading from local file
                try:
                    with open('.streamlit/secrets.toml', 'r') as f:
                        content = f.read()
                        # Find the credentials between triple quotes
                        start = content.find("'''") + 3
                        end = content.rfind("'''")
                        if start == -1 or end == -1:
                            raise ValueError("Could not find credentials in secrets.toml")
                        json_str = content[start:end].strip()
                        firebase_credentials = json.loads(json_str)
                        print("Successfully loaded credentials from local secrets.toml")
                except Exception as local_e:
                    print(f"Failed to load from local file: {str(local_e)}")
                    raise Exception("Failed to load Firebase credentials from both Streamlit secrets and local file")
            
            if not firebase_credentials:
                raise Exception("Firebase credentials are empty")
            
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred)
            print("Successfully initialized Firebase app")
        return firestore.client()
    except Exception as e:
        error_msg = f"Firebase initialization error: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        raise e

def save_feature_request(feature_request, email):
    try:
        db = init_firebase()
        feature_data = {
            'feature': feature_request.strip(),
            'email': email.strip() if email else '',
            'timestamp': datetime.now(),
            'created_at': firestore.SERVER_TIMESTAMP
        }
        db.collection('feature_requests').add(feature_data)
        return True
    except Exception as e:
        if "SERVICE_DISABLED" in str(e):
            st.error("Database service is currently being activated. Please try again in a few minutes.")
        else:
            st.error(f"Failed to save feature request: {str(e)}")
        return False

def render_feature_form():
    # Initialize session state variables if they don't exist
    if 'form_submit_success' not in st.session_state:
        st.session_state.form_submit_success = False
        st.session_state.feature_input = ""
        st.session_state.email_input = ""

    # Reset fields if previous submission was successful
    if st.session_state.form_submit_success:
        st.session_state.feature_input = ""
        st.session_state.email_input = ""
        st.session_state.form_submit_success = False

    # Custom CSS for styling
    st.markdown("""
    <style>
        /* Feature request section container */
        .feature-request-section {
            background: linear-gradient(180deg, rgba(76, 175, 80, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
            border: 1px solid rgba(76, 175, 80, 0.2);
            border-radius: 12px;
            padding: 2.5rem;
            margin: 3rem 0;
        }
        
        /* Title styling */
        .feature-title {
            color: white;
            font-size: 2.2em;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }
        
        /* Promise text styling */
        .promise-text {
            color: #4CAF50;
            font-size: 1.3em;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 500;
        }
    </style>
    
    <div class='feature-request-section'>
        <h2 class='feature-title'>Feature Requests</h2>
        <p class='promise-text'>I will build and deploy your ideas in 2 days.</p>
    """, unsafe_allow_html=True)
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        feature_request = st.text_input("Suggest a Feature", 
                                      placeholder="What feature would you like to see?",
                                      key="feature_input")
    with col2:
        contact_email = st.text_input("Your Email (optional)", 
                                    placeholder="Where should we notify you?",
                                    key="email_input")
    
    # Submit button
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("Submit Request", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        if not feature_request:
            st.error("Please enter a feature request")
        else:
            with st.spinner('Saving your feature request...'):
                if save_feature_request(feature_request, contact_email):
                    st.success("Thank you for your suggestion!")
                    st.session_state.form_submit_success = True
                    st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Feature Request Form", layout="centered")
    render_feature_form() 

