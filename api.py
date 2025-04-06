from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
from main import SHLAssessmentAnalyzer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SHL Assessment Recommender API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = SHLAssessmentAnalyzer("SHL_Assignment_Data.csv")

class AssessmentResponse(BaseModel):
    assessments: list

@app.get("/recommend", response_model=AssessmentResponse)
async def recommend(query: str = Query(..., description="Job requirements query")):
    response = analyzer.rag_chain.invoke(query)
    
    # Parse the response
    assessments = []
    current = {}
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.startswith('- Assessment Name:'):
            if current:
                assessments.append(current)
                current = {}
            current['name'] = line.replace('- Assessment Name:', '').strip()
        elif line.startswith('- Test Type:'):
            current['test_type'] = line.replace('- Test Type:', '').strip()
        elif line.startswith('- Key Features:'):
            current['features'] = line.replace('- Key Features:', '').strip()
        elif line.startswith('- Duration:'):
            current['duration'] = line.replace('- Duration:', '').strip()
        elif line.startswith('- Remote Testing Support:'):
            current['remote_testing'] = line.replace('- Remote Testing Support:', '').strip()
        elif line.startswith('- Adaptive/IRT Support:'):
            current['adaptive_support'] = line.replace('- Adaptive/IRT Support:', '').strip()
        elif line.startswith('- URL:'):
            current['url'] = line.replace('- URL:', '').strip()
    
    if current:
        assessments.append(current)
    
    return {"assessments": assessments}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
