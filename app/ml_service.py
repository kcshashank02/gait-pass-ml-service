import logging
import numpy as np
from typing import Dict, List, Optional
import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class MLService:
    """
    Face Recognition ML Service using InsightFace
    """
    
    def __init__(self):
        self.app = None
        self.model_name = 'buffalo_l'
        self.initialized = False
        
    async def initialize(self):
        """Initialize the face analysis model"""
        try:
            logger.info("Initializing InsightFace model...")
            self.app = FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 for CPU, 0 for GPU
            self.initialized = True
            logger.info("✅ InsightFace model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def extract_face_embedding_from_array(self, img_array: np.ndarray) -> Dict:
        """
        Extract face embedding from image array
        
        Args:
            img_array: Image as numpy array (BGR format)
            
        Returns:
            Dictionary with success status, embedding, and bbox
        """
        try:
            if not self.initialized:
                raise RuntimeError("ML Service not initialized")
            
            # Detect faces
            faces = self.app.get(img_array)
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "message": "No face detected",
                    "embedding": None,
                    "bbox": None
                }
            
            # Get the first (largest) face
            face = faces[0]
            embedding = face.embedding.tolist()
            bbox = face.bbox.tolist()
            
            return {
                "success": True,
                "message": "Face embedding extracted",
                "embedding": embedding,
                "bbox": bbox,
                "num_faces": len(faces)
            }
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": str(e),
                "embedding": None,
                "bbox": None
            }
    
    async def compare_embeddings(
        self, 
        embedding1: List[float], 
        embedding2: List[float], 
        threshold: float = 0.4
    ) -> Dict:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (default: 0.4)
            
        Returns:
            Dictionary with match result and similarity score
        """
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Check if match
            is_match = float(similarity) > threshold
            
            return {
                "success": True,
                "match": is_match,
                "similarity": float(similarity),
                "threshold": threshold
            }
        except Exception as e:
            logger.error(f"Embedding comparison failed: {e}", exc_info=True)
            return {
                "success": False,
                "match": False,
                "similarity": 0.0,
                "error": str(e)
            }
    async def batch_recognize(
        self,
        query_embedding: List[float],
        known_faces: Dict[str, List[float]],
        threshold: float = 0.4
    ) -> Dict:
        """
        Batch face recognition against known faces
        """
        try:
            if not known_faces:
                return {
                    "recognized": False,
                    "message": "No known faces to compare"
                }
            
            logger.info(f"Batch recognize: query embedding length: {len(query_embedding)}")
            logger.info(f"Number of known faces: {len(known_faces)}")
            
            query_emb = np.array(query_embedding, dtype=np.float32)
            best_match = None
            best_similarity = 0.0
            
            # Compare with all known faces
            for user_id, embedding in known_faces.items():
                try:
                    logger.info(f"Comparing with user {user_id}, embedding length: {len(embedding)}")
                    
                    known_emb = np.array(embedding, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(query_emb, known_emb)
                    query_norm = np.linalg.norm(query_emb)
                    known_norm = np.linalg.norm(known_emb)
                    
                    similarity = dot_product / (query_norm * known_norm)
                    
                    logger.info(f"User {user_id}: similarity = {similarity:.4f}")
                    
                    if similarity > best_similarity:
                        best_similarity = float(similarity)
                        best_match = user_id
                        logger.info(f"New best match: {user_id} with similarity {similarity:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error comparing with user {user_id}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Best match: {best_match}, similarity: {best_similarity:.4f}, threshold: {threshold}")
            
            # Check if best match exceeds threshold
            if best_similarity > threshold:
                return {
                    "recognized": True,
                    "user_id": best_match,
                    "similarity": best_similarity,
                    "threshold": threshold,
                    "message": f"Face recognized as user {best_match}"
                }
            else:
                return {
                    "recognized": False,
                    "user_id": None,
                    "similarity": best_similarity,
                    "threshold": threshold,
                    "message": f"No matching face found. Best similarity: {best_similarity:.4f}"
                }
        except Exception as e:
            logger.error(f"Batch recognition failed: {e}", exc_info=True)
            return {
                "recognized": False,
                "error": str(e),
                "message": f"Recognition error: {str(e)}"
            }

    
    async def health_check(self) -> Dict:
        """Check ML service health"""
        return {
            "initialized": self.initialized,
            "model": self.model_name if self.initialized else None
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.app = None
        self.initialized = False
        logger.info("ML Service cleaned up")








# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# import logging
# from typing import List, Dict, Optional
# import onnxruntime

# logger = logging.getLogger(__name__)

# class MLService:
#     def __init__(self):
#         self.face_app: Optional[FaceAnalysis] = None
#         self.current_provider: Optional[str] = None
#         self.is_initialized = False
        
#     async def initialize(self):
#         """Initialize with auto GPU/CPU detection"""
#         try:
#             # ✅ Auto-detect available providers
#             available_providers = onnxruntime.get_available_providers()
#             logger.info(f"Available providers: {available_providers}")
            
#             # Priority: CUDA > CPU
#             if 'CUDAExecutionProvider' in available_providers:
#                 providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#                 self.current_provider = 'CUDAExecutionProvider'
#                 logger.info("🚀 GPU (CUDA) detected - Using GPU acceleration")
#             else:
#                 providers = ['CPUExecutionProvider']
#                 self.current_provider = 'CPUExecutionProvider'
#                 logger.info("💻 GPU not available - Using CPU")
            
#             # Initialize with buffalo_sc (lightweight)
#             self.face_app = FaceAnalysis(
#                 name='buffalo_sc',
#                 providers=providers
#             )
            
#             # Prepare model
#             ctx_id = 0 if self.current_provider == 'CUDAExecutionProvider' else -1
#             self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            
#             self.is_initialized = True
#             logger.info(f"✅ ML Service initialized (buffalo_sc, {self.current_provider}, 320x320)")
            
#         except Exception as e:
#             logger.error(f"❌ ML Service initialization failed: {e}")
#             raise
    
#     async def extract_face_embedding_from_array(self, image_array: np.ndarray) -> Dict:
#         """Extract embedding from numpy array"""
#         if not self.is_initialized:
#             raise RuntimeError("ML Service not initialized")
        
#         try:
#             faces = self.face_app.get(image_array)
            
#             if len(faces) == 0:
#                 return {
#                     "success": False,
#                     "message": "No face detected",
#                     "faces_detected": 0
#                 }
            
#             # Get the face with highest confidence
#             face = max(faces, key=lambda f: f.det_score)
            
#             return {
#                 "success": True,
#                 "embedding": face.normed_embedding.tolist(),
#                 "bbox": face.bbox.astype(int).tolist(),
#                 "confidence": float(face.det_score),
#                 "faces_detected": len(faces),
#                 "provider": self.current_provider
#             }
            
#         except Exception as e:
#             logger.error(f"Embedding extraction failed: {e}")
#             raise
    
#     async def compare_embeddings(
#         self, 
#         embedding1: List[float], 
#         embedding2: List[float], 
#         threshold: float = 0.4
#     ) -> Dict:
#         """Compare two embeddings using cosine similarity"""
#         try:
#             emb1 = np.array(embedding1)
#             emb2 = np.array(embedding2)
            
#             # Cosine similarity (embeddings are already normalized)
#             similarity = float(np.dot(emb1, emb2))
            
#             return {
#                 "success": True,
#                 "similarity": similarity,
#                 "match": similarity > threshold,
#                 "threshold": threshold
#             }
#         except Exception as e:
#             logger.error(f"Comparison failed: {e}")
#             raise
    
#     async def batch_recognize(
#         self, 
#         query_embedding: List[float], 
#         known_faces: Dict[str, List[float]], 
#         threshold: float = 0.4
#     ) -> Dict:
#         """Batch recognition against known faces database"""
#         try:
#             query_emb = np.array(query_embedding)
#             best_match = None
#             best_similarity = 0.0
#             all_scores = {}
            
#             for user_id, known_embedding in known_faces.items():
#                 known_emb = np.array(known_embedding)
#                 similarity = float(np.dot(query_emb, known_emb))
#                 all_scores[user_id] = similarity
                
#                 if similarity > best_similarity:
#                     best_similarity = similarity
#                     best_match = user_id
            
#             return {
#                 "success": True,
#                 "match": best_match if best_similarity > threshold else None,
#                 "similarity": best_similarity,
#                 "threshold": threshold,
#                 "all_scores": all_scores,
#                 "provider": self.current_provider
#             }
#         except Exception as e:
#             logger.error(f"Batch recognition failed: {e}")
#             raise
    
#     async def health_check(self) -> Dict:
#         """Health check"""
#         return {
#             "status": "healthy" if self.is_initialized else "not_initialized",
#             "provider": self.current_provider,
#             "model": "buffalo_sc" if self.is_initialized else None
#         }
    
#     async def cleanup(self):
#         """Cleanup resources"""
#         self.is_initialized = False
#         self.face_app = None
#         logger.info("ML Service cleaned up")
