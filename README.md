# SHL Assessment Recommender Engine

## **Project Overview**
The SHL Assessment Recommender System simplifies the process of finding relevant assessments for hiring managers. It leverages advanced AI techniques to recommend the most suitable SHL assessments based on natural language queries or job descriptions. The system provides key details about each assessment, including its name, URL, duration, test type, and support for remote testing and adaptive/IRT features.

---

## **Features**
1. **Dynamic Recommendations**:
   - Accepts natural language queries or job descriptions.
   - Returns up to 10 relevant SHL assessments with key attributes.
2. **Key Attributes for Each Recommendation**:
   - Assessment Name (linked to SHL catalog).
   - Remote Testing Support (Yes/No).
   - Adaptive/IRT Support (Yes/No).
   - Duration and Test Type.
3. **Interactive Web Application**:
   - Built with Streamlit for a user-friendly interface.
4. **API Endpoint**:
   - REST API using FastAPI for JSON-based recommendations.

---

## **Architecture**
1. **Data Collection**:
   - Scraped data from SHL's product catalog using `Selenium` and `BeautifulSoup`.
   - Extracted ~400 assessments by iterating 32 times (12 entries per page) due to dynamic loading constraints.
   - Merged all scraped data into a single CSV file (`SHL_Assignment_Data.csv`).

2. **Data Processing**:
   - Cleaned and preprocessed data using Pandas.
   - Standardized missing values (e.g., "No" for Remote Testing).
   - Combined multiple columns into a single text field for embedding generation.

3. **RAG Pipeline**:
   - **Embeddings**: Used `sentence-transformers/all-MiniLM-L6-v2` for efficient vectorization of assessment data.
   - **Vector Store**: Implemented FAISS for similarity search.
   - **LLM**: Integrated Hugging Face's `google/flan-t5-small` model for generating recommendations.

4. **Deployment**:
   - Hosted the web app on Streamlit Cloud.
   - Deployed the API endpoint on Render.

---

## **Tools and Libraries**
- **Frontend**: Streamlit (for UI).
- **Backend**: FastAPI (for API endpoint).
- **Data Processing**: Pandas, BeautifulSoup, Selenium.
- **Embeddings & LLMs**: Hugging Face Transformers, Sentence Transformers.
- **Vector Search**: FAISS.
- **Deployment**: Streamlit Cloud, Render.

---

## **How It Works**
1. **Input Query**:
   Users provide a natural language query or job description in the Streamlit app or via the API endpoint.

2. **Recommendation Generation**:
   - The query is processed using a Retrieval-Augmented Generation (RAG) pipeline.
   - Relevant assessments are retrieved from the FAISS vector store based on semantic similarity.
   - The LLM generates detailed recommendations in a structured format.

3. **Output**:
   Recommendations are displayed in a tabular format with attributes like assessment name, URL, duration, test type, and support features.

---

## **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shl-assessment-recommender.git
   cd shl-assessment-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Hugging Face API token to `config.env`:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Run the FastAPI server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

---

## **Example Queries**
1. *"I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."*
2. *"Looking to hire mid-level professionals who are proficient in Python, SQL, and JavaScript. Need an assessment package that can test all skills with a max duration of 60 minutes."*

---

## **Deployment Links**
1. [Streamlit Demo](https://shl-assessment-recommender-engine-ue5x9muxspqrhxpxhwisxh.streamlit.app/)
2. [Api endpoint](https://shl-assessment-recommender-engine.onrender.com)
3. [GitHub Repository](https://github.com/MeghaBharti/shl-assessment-recommender-engine)

---

## **Future Enhancements**
- Add support for parsing job descriptions from URLs.
- Implement caching for frequently queried results.
- Expand dataset to include additional SHL modules.

This project demonstrates how modern AI techniques can streamline hiring processes by providing intelligent recommendations tailored to specific job requirements!
