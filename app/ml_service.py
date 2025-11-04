import cv2
import numpy as np
from insightface.app import FaceAnalysis
import logging
from typing import List, Dict, Optional
import onnxruntime

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.face_app: Optional[FaceAnalysis] = None
        self.current_provider: Optional[str] = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize with auto GPU/CPU detection"""
        try:
            # ✅ Auto-detect available providers
            available_providers = onnxruntime.get_available_providers()
            logger.info(f"Available providers: {available_providers}")
            
            # Priority: CUDA > CPU
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.current_provider = 'CUDAExecutionProvider'
                logger.info("🚀 GPU (CUDA) detected - Using GPU acceleration")
            else:
                providers = ['CPUExecutionProvider']
                self.current_provider = 'CPUExecutionProvider'
                logger.info("💻 GPU not available - Using CPU")
            
            # Initialize with buffalo_sc (lightweight)
            self.face_app = FaceAnalysis(
                name='buffalo_sc',
                providers=providers
            )
            
            # Prepare model
            ctx_id = 0 if self.current_provider == 'CUDAExecutionProvider' else -1
            self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            
            self.is_initialized = True
            logger.info(f"✅ ML Service initialized (buffalo_sc, {self.current_provider}, 320x320)")
            
        except Exception as e:
            logger.error(f"❌ ML Service initialization failed: {e}")
            raise
    
    async def extract_face_embedding_from_array(self, image_array: np.ndarray) -> Dict:
        """Extract embedding from numpy array"""
        if not self.is_initialized:
            raise RuntimeError("ML Service not initialized")
        
        try:
            faces = self.face_app.get(image_array)
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "message": "No face detected",
                    "faces_detected": 0
                }
            
            # Get the face with highest confidence
            face = max(faces, key=lambda f: f.det_score)
            
            return {
                "success": True,
                "embedding": face.normed_embedding.tolist(),
                "bbox": face.bbox.astype(int).tolist(),
                "confidence": float(face.det_score),
                "faces_detected": len(faces),
                "provider": self.current_provider
            }
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise
    
    async def compare_embeddings(
        self, 
        embedding1: List[float], 
        embedding2: List[float], 
        threshold: float = 0.4
    ) -> Dict:
        """Compare two embeddings using cosine similarity"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(emb1, emb2))
            
            return {
                "success": True,
                "similarity": similarity,
                "match": similarity > threshold,
                "threshold": threshold
            }
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise
    
    async def batch_recognize(
        self, 
        query_embedding: List[float], 
        known_faces: Dict[str, List[float]], 
        threshold: float = 0.4
    ) -> Dict:
        """Batch recognition against known faces database"""
        try:
            query_emb = np.array(query_embedding)
            best_match = None
            best_similarity = 0.0
            all_scores = {}
            
            for user_id, known_embedding in known_faces.items():
                known_emb = np.array(known_embedding)
                similarity = float(np.dot(query_emb, known_emb))
                all_scores[user_id] = similarity
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_id
            
            return {
                "success": True,
                "match": best_match if best_similarity > threshold else None,
                "similarity": best_similarity,
                "threshold": threshold,
                "all_scores": all_scores,
                "provider": self.current_provider
            }
        except Exception as e:
            logger.error(f"Batch recognition failed: {e}")
            raise
    
    async def health_check(self) -> Dict:
        """Health check"""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "provider": self.current_provider,
            "model": "buffalo_sc" if self.is_initialized else None
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_initialized = False
        self.face_app = None
        logger.info("ML Service cleaned up")
