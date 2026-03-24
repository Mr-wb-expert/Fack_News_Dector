import streamlit as st
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Page Config ---
st.set_page_config(
    page_title="TruthSeeker AI | Fake News Detector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS to match the original gradient and glassmorphism UI ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Background blur effects */
    .stApp::before {
        content: "";
        position: fixed;
        width: 500px;
        height: 500px;
        background: #6366f1;
        filter: blur(150px);
        border-radius: 50%;
        opacity: 0.15;
        z-index: 0;
        top: -100px;
        left: -100px;
    }
    
    .stApp::after {
        content: "";
        position: fixed;
        width: 400px;
        height: 400px;
        background: #c084fc;
        filter: blur(150px);
        border-radius: 50%;
        opacity: 0.15;
        z-index: 0;
        bottom: -100px;
        right: -100px;
        pointer-events: none;
    }

    /* Card styling for the main container */
    .main .block-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-top: 3rem;
        margin-bottom: 3rem;
        z-index: 10;
        position: relative;
    }
    
    /* Headers */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #818cf8, #c084fc) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.05em !important;
        text-align: center;
        padding-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        color: #f8fafc !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        min-height: 200px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    }

    /* Hide standard label for text area */
    .stTextArea label {
        display: none;
    }

    /* Primary Button */
    .stButton button {
        width: 100% !important;
        background: #6366f1 !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 16px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4) !important;
        filter: brightness(1.1) !important;
        color: white !important;
    }
    
    /* Results Styling */
    .result-real {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        color: #22c55e;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2rem;
        animation: slideUp 0.5s ease-out;
    }
    
    .result-fake {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #ef4444;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2rem;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Remove default Streamlit top padding and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- NLTK Data and Models ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

@st.cache_resource
def load_models():
    try:
        model = joblib.load("final_model(XGBoost).pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess(text):
    text = clean_text(text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])
    return text

# --- App Logic ---
download_nltk_data()
model, vectorizer = load_models()

st.markdown("<h1>TruthSeeker AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced XGBoost News Classification System</div>", unsafe_allow_html=True)

user_input = st.text_area("News Text", placeholder="Paste full news article text here for analysis...")

if st.button("Check Credibility"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Models failed to load. Cannot parse news.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned = preprocess(user_input)
            transform = vectorizer.transform([cleaned])
            pred = model.predict(transform)[0]
            
            if pred == 1:
                st.markdown('<div class="result-real">✅ THIS IS REAL NEWS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fake">⚠️ WARNING: THIS IS FAKE NEWS</div>', unsafe_allow_html=True)
