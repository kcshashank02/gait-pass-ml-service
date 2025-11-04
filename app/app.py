from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import cv2
import numpy as np
from typing import List, Dict


from app.ml_service import MLService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ML service
ml_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Initializing ML Service...")
    global ml_service
    ml_service = MLService()
    await ml_service.initialize()
    logger.info("✅ ML Service ready")
    
    yield
    
    # Shutdown
    if ml_service:
        await ml_service.cleanup()
    logger.info("🛑 ML Service shutdown")

app = FastAPI(
    title="Gait-Pass ML Service",
    description="Face Recognition ML Service for Gait-Pass",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Gait-Pass ML Service",
        "status": "running",
        "endpoints": [
            "/extract-embedding",
            "/compare-embeddings",
            "/batch-recognize",
            "/health"
        ]
    }

@app.post("/extract-embedding")
async def extract_embedding(image: UploadFile = File(...)):
    """Extract face embedding from image"""
    try:
        # Read image
        content = await image.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Extract embedding
        result = await ml_service.extract_face_embedding_from_array(img)
        return result
        
    except Exception as e:
        logger.error(f"Extract embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 


@app.post("/compare-embeddings")
async def compare_embeddings(
    embedding1: List[float],
    embedding2: List[float],
    threshold: float = 0.4
):
    """Compare two embeddings"""
    try:
        result = await ml_service.compare_embeddings(embedding1, embedding2, threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-recognize")
async def batch_recognize(request: dict):
    """Batch face recognition"""
    try:
        query_embedding = request.get("query_embedding")
        known_faces = request.get("known_faces", {})
        threshold = request.get("threshold", 0.4)
        
        if not query_embedding:
            raise HTTPException(status_code=400, detail="Query embedding required")
        
        if not known_faces:
            return {
                "recognized": False,
                "message": "No known faces provided"
            }
        
        # Convert to numpy arrays
        query_emb = np.array(query_embedding)
        
        best_match_user_id = None
        min_distance = float('inf')
        
        # Compare with all known faces
        for user_id, embeddings_list in known_faces.items():
            for embedding in embeddings_list:
                known_emb = np.array(embedding)
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(query_emb - known_emb)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_user_id = user_id
        
        # Check if best match is within threshold
        if min_distance <= threshold:
            return {
                "recognized": True,
                "user_id": best_match_user_id,
                "confidence": float(1 - (min_distance / threshold)),
                "min_distance": float(min_distance)
            }
        else:
            return {
                "recognized": False,
                "message": "No match found within threshold",
                "min_distance": float(min_distance),
                "threshold": threshold
            }
            
    except Exception as e:
        logger.error(f"Batch recognize failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check"""
    try:
        health_status = await ml_service.health_check()
        return {
            "status": "healthy",
            "ml_service": health_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)  # Hugging Face uses port 7860
