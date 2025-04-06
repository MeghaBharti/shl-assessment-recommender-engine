import streamlit as st

# First Streamlit command
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="centered",
    page_icon="🔍"
)

# Custom CSS for numbered assessment format
st.markdown("""
<style>
    /* Dark theme for assessment blocks */
    .assessment-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 30px;
        color: white;
    }
    
    /* Styling for assessment titles */
    .assessment-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: white;
    }
    
    /* Styling for assessment details */
    .assessment-detail {
        margin-left: 15px;
        margin-bottom: 5px;
        display: flex;
    }
    
    /* Bullet point style */
    .bullet {
        color: white;
        margin-right: 8px;
    }
    
    /* Link styling */
    a {
        color: #4287f5;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Support tag styling */
    .support-tag {
        display: inline-block;
        padding: 2px 8px;
        margin-right: 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .yes-tag {
        background-color: #4CAF50;
        color: white;
    }
    .no-tag {
        background-color: #F44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

import os
from dotenv import load_dotenv
from main import SHLAssessmentAnalyzer

load_dotenv('config.env')

@st.cache_resource
def load_analyzer():
    return SHLAssessmentAnalyzer("SHL_Assignment_Data.csv")

analyzer = load_analyzer()

st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter job requirements to get matching SHL assessments")

query = st.text_area(
    "📝 Enter Job Requirements",
    height=150,
    placeholder="E.g., Mid-level .NET developers with SQL skills..."
)

if st.button("🔎 Get Recommendation"):
    if query.strip():
        with st.spinner("Analyzing..."):
            try:
                # Get response from RAG model
                response = analyzer.rag_chain.invoke(query)
                
                # Parse the response into individual assessments
                assessments = []
                current_assessment = {}
                lines = response.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('- Assessment Name:'):
                        if current_assessment:
                            assessments.append(current_assessment)
                            current_assessment = {}
                        current_assessment['name'] = line.replace('- Assessment Name:', '').strip()
                    elif line.startswith('- Test Type:'):
                        current_assessment['test_type'] = line.replace('- Test Type:', '').strip()
                    elif line.startswith('- Key Features:'):
                        current_assessment['features'] = line.replace('- Key Features:', '').strip()
                    elif line.startswith('- Duration:'):
                        current_assessment['duration'] = line.replace('- Duration:', '').strip()
                    elif line.startswith('- Remote Testing Support:'):
                        current_assessment['remote_testing'] = line.replace('- Remote Testing Support:', '').strip()
                    elif line.startswith('- Adaptive/IRT Support:'):
                        current_assessment['adaptive'] = line.replace('- Adaptive/IRT Support:', '').strip()
                    elif line.startswith('- URL:'):
                        current_assessment['url'] = line.replace('- URL:', '').strip()
                
                # Add the last assessment if any
                if current_assessment:
                    assessments.append(current_assessment)
                
                # Display results in numbered format with bullet points
                st.markdown("### 📋 Recommended Assessments")
                
                for i, assessment in enumerate(assessments, 1):
                    name = assessment.get('name', '')
                    test_type = assessment.get('test_type', '')
                    features = assessment.get('features', '')
                    duration = assessment.get('duration', '')
                    remote_testing = assessment.get('remote_testing', 'No')
                    adaptive = assessment.get('adaptive', 'No')
                    url = assessment.get('url', '')
                    
                    # Create status tags based on Yes/No values
                    remote_class = "yes-tag" if remote_testing.strip().lower() == "yes" else "no-tag"
                    adaptive_class = "yes-tag" if adaptive.strip().lower() == "yes" else "no-tag"
                    
                    # Format with numbers and bullet points
                    html = f"""
                    <div class="assessment-container">
                        <div class="assessment-title">{i}. {name}</div>
                        <div class="assessment-detail"><span class="bullet">•</span> <strong>Test Type:</strong> {test_type}</div>
                        <div class="assessment-detail"><span class="bullet">•</span> <strong>Key Features:</strong> {features}</div>
                        <div class="assessment-detail"><span class="bullet">•</span> <strong>Duration:</strong> {duration}min</div>
                        <div class="assessment-detail"><span class="bullet">•</span> <strong>Support:</strong> 
                            <span class="support-tag {remote_class}">Remote Testing: {remote_testing}</span>
                            <span class="support-tag {adaptive_class}">Adaptive/IRT: {adaptive}</span>
                        </div>
                        <div class="assessment-detail"><span class="bullet">•</span> <strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                
                # Fallback for unparsed output
                if not assessments:
                    st.markdown(response)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query")
