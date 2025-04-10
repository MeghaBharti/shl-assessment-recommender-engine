import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv('config.env')

class SHLAssessmentAnalyzer:
    def __init__(self, data_path):
        self.df = self.load_and_preprocess(data_path)
        self.documents = self.create_documents()
        self.vector_store = self.create_vector_store()
        self.rag_chain = self.create_rag_chain()
    
    def load_and_preprocess(self, file_path):
        df = pd.read_csv(file_path)
        
        # Enhanced data cleaning
        df['Job Levels'] = df['Job Levels'].fillna('General').apply(
            lambda x: x.split(', ') if isinstance(x, str) else ['General']
        )
        df['Test Type'] = df['Test Type'].fillna('General Assessment')
        df['Assessment Length'] = df['Assessment Length'].astype(str)
        df['Remote Testing'] = df['Remote Testing'].fillna('No')
        df['Adaptive / IRT'] = df['Adaptive / IRT'].fillna('No')
        
        df['combined_text'] = df.apply(lambda row: f"""
        Assessment: {row['Assessment Name']}
        Description: {row['Description']}
        Suitable For: {', '.join(row['Job Levels'])}
        Test Type: {row['Test Type']}
        Duration: {row['Assessment Length']}
        Remote Testing: {row['Remote Testing']}
        Adaptive/IRT: {row['Adaptive / IRT']}
        URL: {row['URL']}
        """, axis=1)
        
        return df

    def create_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.create_documents(self.df['combined_text'].tolist())


    def create_vector_store(self):
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-mpnet-base-v2"
        # )
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
        model_kwargs={'device': 'cpu'},  # Force CPU usage
        cache_folder="./model_cache"  # Enable model caching
        )
        return FAISS.from_documents(self.documents, embeddings)

    def create_rag_chain(self):
        prompt_template = """You are an SHL assessment expert. Use this context:
        {context}
        
        Question: {question}
        Answer in this format:
        - Assessment Name: [name]
        - Test Type: [type]
        - Description: [detailed description]
        - Duration: [length]
        - Remote Testing Support: [Yes/No]
        - Adaptive/IRT Support: [Yes/No]
        - URL: [link]
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.3,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        return (
            {"context": self.vector_store.as_retriever(search_kwargs={'k': 5}), 
             "question": RunnablePassthrough()}
            | prompt
            | llm
        )

if __name__ == "__main__":
    analyzer = SHLAssessmentAnalyzer("SHL_Assignment_Data.csv")
    query = "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
    print(analyzer.rag_chain.invoke(query))
