from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from main import SHLAssessmentAnalyzer
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv('config.env')

app = FastAPI(title="SHL Assessment Recommender API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer once at startup
analyzer = SHLAssessmentAnalyzer("SHL_Assignment_Data.csv")

# Response models
class AssessmentResponseItem(BaseModel):
    url: str
    adaptive_support: str
    description: str 
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentResponseItem]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(query: str = Body(..., embed=True)):
    try:
        # Get raw response from RAG
        response = analyzer.rag_chain.invoke(query)
        
        # Parse the response into structured format
        assessments = []
        current = {}
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('- Assessment Name:'):
                if current:
                    assessments.append(current)
                    current = {}
                current["name"] = line.replace('- Assessment Name:', '').strip()
            elif line.startswith('- URL:'):
                current["url"] = line.replace('- URL:', '').strip()
            elif line.startswith('- Adaptive/IRT Support:'):
                current["adaptive_support"] = line.replace('- Adaptive/IRT Support:', '').strip()
            elif line.startswith('- Description:'):
                current["description"] = line.replace('- Description:', '').strip()
            elif line.startswith('- Duration:'):
                try:
                    current["duration"] = int(line.replace('- Duration:', '').strip().split()[0])
                except:
                    current["duration"] = 0
            elif line.startswith('- Remote Testing Support:'):
                current["remote_support"] = line.replace('- Remote Testing Support:', '').strip()
            elif line.startswith('- Test Type:'):
                current["test_type"] = [t.strip() for t in line.replace('- Test Type:', '').split(',')]
        
        if current:
            assessments.append(current)

        # Transform to required API format
        formatted_assessments = []
        for item in assessments:
            formatted_assessments.append(AssessmentResponseItem(
                url=item.get("url", ""),
                adaptive_support=item.get("adaptive_support", "No"),
                description=item.get("description", ""),
                duration=item.get("duration", 0),
                remote_support=item.get("remote_support", "No"),
                test_type=item.get("test_type", [])
            ))
        
        return {"recommended_assessments": formatted_assessments[:10]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
