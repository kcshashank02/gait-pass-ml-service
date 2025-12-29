# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# import logging
# import cv2
# import numpy as np
# from typing import List, Dict
# import os
# from app.ml_service import MLService


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # Global ML service
# mlservice = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("Initializing ML Service...")
#     global mlservice
#     mlservice = MLService()
#     await mlservice.initialize()
#     logger.info("ML Service ready")
#     yield
#     # Shutdown
#     if mlservice:
#         await mlservice.cleanup()
#     logger.info("ML Service shutdown")


# app = FastAPI(
#     title="Gait-Pass ML Service",
#     description="Face Recognition ML Service for Gait-Pass",
#     version="1.0.0",
#     lifespan=lifespan
# )


# # Get CORS origins from environment variable
# CORS_ORIGINS = os.getenv(
#     "CORS_ORIGINS",
#     "http://localhost:3000,http://localhost:5173,http://localhost:8000"
# )

# origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]
# logger.info(f"ML Service CORS Origins: {origins}")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/")
# async def root():
#     return {
#         "service": "Gait-Pass ML Service",
#         "status": "running",
#         "endpoints": [
#             "/extract-embedding",
#             "/compare-embeddings",
#             "/batch-recognize",
#             "/health"
#         ]
#     }


# @app.post("/extract-embedding")
# async def extract_embedding(image: UploadFile = File(...)):
#     """Extract face embedding from image"""
#     try:
#         # Read image
#         content = await image.read()
#         nparr = np.frombuffer(content, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             raise HTTPException(status_code=400, detail="Invalid image format")
        
#         # Extract embedding
#         result = await mlservice.extract_face_embedding_from_array(img)
        
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Extract embedding failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/compare-embeddings")
# async def compare_embeddings(request: Dict):
#     """Compare two embeddings"""
#     try:
#         embedding1 = request.get("embedding1")
#         embedding2 = request.get("embedding2")
#         threshold = request.get("threshold", 0.4)
        
#         if not embedding1 or not embedding2:
#             raise HTTPException(status_code=400, detail="Both embeddings required")
        
#         result = await mlservice.compare_embeddings(embedding1, embedding2, threshold)
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Compare embeddings failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/batch-recognize")
# async def batch_recognize(request: Dict):
#     """Batch face recognition"""
#     try:
#         query_embedding = request.get("query_embedding")
#         known_faces = request.get("known_faces", {})
#         threshold = request.get("threshold", 0.4)
        
#         if not query_embedding:
#             raise HTTPException(status_code=400, detail="Query embedding required")
        
#         if not known_faces:
#             return {
#                 "recognized": False,
#                 "message": "No known faces provided"
#             }
        
#         result = await mlservice.batch_recognize(
#             query_embedding,
#             known_faces,
#             threshold
#         )
        
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Batch recognize failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/health")
# async def health():
#     """Health check"""
#     try:
#         health_status = await mlservice.health_check()
#         return {
#             "status": "healthy",
#             "ml_service": health_status
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)
































from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import cv2
import numpy as np
from typing import List, Dict
import os
from app.ml_service import MLService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ML service
mlservice = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing ML Service...")
    global mlservice
    mlservice = MLService()
    await mlservice.initialize()
    logger.info("ML Service ready")
    yield
    # Shutdown
    if mlservice:
        await mlservice.cleanup()
    logger.info("ML Service shutdown")

app = FastAPI(
    title="Gait-Pass ML Service",
    description="Face Recognition ML Service for Gait-Pass",
    version="1.0.0",
    lifespan=lifespan
)

# Get CORS origins from environment variable
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8000"
)

origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]
logger.info(f"ML Service CORS Origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.1.39:3000",
        "http://192.168.*.*:3000",
        "http://192.168.1.6:3000","http://172.26.151.109:3000/" ,"http://172.16.3.180:3000",
          # Allow any device on local network
        "*",  # Or use this for development
        origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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
        logger.info(f"Received image: {image.filename}")
        
        # Read image
        content = await image.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        logger.info(f"Image decoded: shape {img.shape}")
        
        # Extract embedding
        result = await mlservice.extract_face_embedding_from_array(img)
        
        logger.info(f"Extraction result: success={result.get('success')}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extract embedding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-embeddings")
async def compare_embeddings(
    request: Dict
):
    """Compare two embeddings"""
    try:
        embedding1 = request.get("embedding1")
        embedding2 = request.get("embedding2")
        threshold = request.get("threshold", 0.4)
        
        if not embedding1 or not embedding2:
            raise HTTPException(status_code=400, detail="Both embeddings required")
        
        result = await mlservice.compare_embeddings(embedding1, embedding2, threshold)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare embeddings failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-recognize")
async def batch_recognize(request: Dict):
    """Batch face recognition"""
    try:
        query_embedding = request.get("query_embedding")
        known_faces = request.get("known_faces", {})
        threshold = request.get("threshold", 0.4)
        
        logger.info(f"Batch recognize: {len(known_faces)} known faces, threshold={threshold}")
        
        if not query_embedding:
            raise HTTPException(status_code=400, detail="Query embedding required")
        
        if not known_faces:
            return {
                "recognized": False,
                "message": "No known faces provided"
            }
        
        result = await mlservice.batch_recognize(
            query_embedding,
            known_faces,
            threshold
        )
        
        logger.info(f"Recognition result: recognized={result.get('recognized')}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch recognize failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    try:
        health_status = await mlservice.health_check()
        return {
            "status": "healthy",
            "ml_service": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)































# # from fastapi import FastAPI, File, UploadFile, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from contextlib import asynccontextmanager
# # import logging
# # import cv2
# # import numpy as np
# # from typing import List, Dict
# # import os
# # from app.ml_service import MLService  

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Global ML service
# # mlservice = None

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     # Startup
# #     logger.info("Initializing ML Service...")
# #     global mlservice
# #     mlservice = MLService()
# #     await mlservice.initialize()
# #     logger.info("ML Service ready")
# #     yield
# #     # Shutdown
# #     if mlservice:
# #         await mlservice.cleanup()
# #     logger.info("ML Service shutdown")

# # app = FastAPI(
# #     title="Gait-Pass ML Service",
# #     description="Face Recognition ML Service for Gait-Pass",
# #     version="1.0.0",
# #     lifespan=lifespan
# # )

# # # ✅ Get CORS origins from environment variable
# # CORS_ORIGINS = os.getenv(
# #     "CORS_ORIGINS",
# #     "http://localhost:3000,http://localhost:5173,https://gait-pass-backend.onrender.com"
# # )

# # # ✅ Parse comma-separated origins
# # origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]
# # logger.info(f"ML Service CORS Origins: {origins}")

# # # ✅ Add CORS middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=['http://localhost:3000', 'http://localhost:5173', {CORS_ORIGINS}], 
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # @app.get("/")
# # async def root():
# #     return {
# #         "service": "Gait-Pass ML Service",
# #         "status": "running",
# #         "endpoints": [
# #             "/extract-embedding",
# #             "/compare-embeddings",
# #             "/batch-recognize",
# #             "/health"
# #         ]
# #     }

# # @app.post("/extract-embedding")
# # async def extract_embedding(image: UploadFile = File(...)):
# #     """Extract face embedding from image"""
# #     try:
# #         content = await image.read()
# #         nparr = np.frombuffer(content, np.uint8)
# #         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
# #         if img is None:
# #             raise HTTPException(status_code=400, detail="Invalid image")
        
# #         result = await mlservice.extract_face_embedding_from_array(img)
# #         return result
# #     except Exception as e:
# #         logger.error(f"Extract embedding failed: {e}")
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/compare-embeddings")
# # async def compare_embeddings(
# #     embedding1: List[float],
# #     embedding2: List[float],
# #     threshold: float = 0.4
# # ):
# #     """Compare two embeddings"""
# #     try:
# #         result = await mlservice.compare_embeddings(embedding1, embedding2, threshold)
# #         return result
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/batch-recognize")
# # async def batch_recognize(request: dict):
# #     """Batch face recognition"""
# #     try:
# #         query_embedding = request.get("query_embedding")
# #         known_faces = request.get("known_faces", {})
# #         threshold = request.get("threshold", 0.4)
        
# #         if not query_embedding:
# #             raise HTTPException(status_code=400, detail="Query embedding required")
        
# #         if not known_faces:
# #             return {
# #                 "recognized": False,
# #                 "message": "No known faces provided"
# #             }
        
# #         result = await mlservice.batch_recognize(
# #             query_embedding,
# #             known_faces,
# #             threshold
# #         )
# #         return result
# #     except Exception as e:
# #         logger.error(f"Batch recognize failed: {e}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.get("/health")
# # async def health():
# #     """Health check"""
# #     try:
# #         health_status = await mlservice.health_check()
# #         return {
# #             "status": "healthy",
# #             "ml_service": health_status
# #         }
# #     except Exception as e:
# #         return {
# #             "status": "unhealthy",
# #             "error": str(e)
# #         }

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=7860)
