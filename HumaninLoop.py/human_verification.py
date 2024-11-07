"""
human_verification.py: Manages human verification process and feedback
"""
import time
from typing import Dict, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from .human_checkpoint import HumanCheckpoint, CheckpointRequest, CheckpointResponse

class VerificationRequest(BaseModel):
    content: Dict
    verification_type: str
    metadata: Optional[Dict] = None
    priority: int = 1
    required_expertise: Optional[List[str]] = None

class VerificationResult(BaseModel):
    verified: bool
    feedback: Optional[str] = None
    corrections: Optional[Dict] = None
    confidence_score: float
    reviewer_expertise: Optional[List[str]] = None

class HumanVerification:
    def __init__(self):
        self.checkpoint_manager = HumanCheckpoint()
        self.verification_history: Dict[str, List[VerificationResult]] = {}
        
    async def verify_content(self, request: VerificationRequest) -> VerificationResult:
        """Submit content for human verification"""
        # Create checkpoint request
        checkpoint_request = CheckpointRequest(
            task_id=f"verify_{int(time.time())}",
            checkpoint_type=request.verification_type,
            data={
                "content": request.content,
                "metadata": request.metadata,
                "required_expertise": request.required_expertise
            },
            priority=request.priority
        )
        
        # Create checkpoint and await verification
        checkpoint_id = await self.checkpoint_manager.create_checkpoint(checkpoint_request)
        checkpoint_response = await self.checkpoint_manager.await_verification(checkpoint_id)
        
        # Convert checkpoint response to verification result
        result = VerificationResult(
            verified=checkpoint_response.approved,
            feedback=checkpoint_response.feedback,
            corrections=checkpoint_response.modifications,
            confidence_score=1.0 if checkpoint_response.approved else 0.0,
            reviewer_expertise=request.required_expertise
        )
        
        # Store in history
        if checkpoint_request.task_id not in self.verification_history:
            self.verification_history[checkpoint_request.task_id] = []
        self.verification_history[checkpoint_request.task_id].append(result)
        
        return result
        
    async def get_verification_status(self, task_id: str) -> List[VerificationResult]:
        """Get verification history for a task"""
        return self.verification_history.get(task_id, [])
        
    def get_pending_verifications(self, expertise: Optional[List[str]] = None) -> List[Dict]:
        """Get pending verifications filtered by expertise"""
        pending = self.checkpoint_manager.get_pending_checkpoints()
        
        if expertise is None:
            return pending
            
        return [
            p for p in pending
            if not p["data"].get("required_expertise") or
            any(exp in expertise for exp in p["data"]["required_expertise"])
        ]

# FastAPI endpoints
app = FastAPI()
verifier = HumanVerification()

@app.post("/verify")
async def create_verification(request: VerificationRequest):
    result = await verifier.verify_content(request)
    return result

@app.get("/verify/{task_id}")
async def get_verification(task_id: str):
    results = await verifier.get_verification_status(task_id)
    if not results:
        raise HTTPException(status_code=404, detail="Verification task not found")
    return results

@app.get("/verify/pending")
async def get_pending_verifications(expertise: Optional[List[str]] = None):
    return verifier.get_pending_verifications(expertise)






# # Example HITL Implementation
# class HumanInTheLoop:
#     def __init__(self):
#         self.pending_reviews = Queue()
#         self.feedback_history = []
        
#     async def request_human_review(self, content, importance_level):
#         if importance_level >= THRESHOLD:
#             return await self.get_human_feedback(content)
#         return None
        
#     async def get_human_feedback(self, content):
#         # Send to human interface
#         feedback = await self.human_interface.get_feedback(content)
#         self.feedback_history.append({
#             'content': content,
#             'feedback': feedback,
#             'timestamp': datetime.now()
#         })
#         return feedback

# # Example Agent with HITL and Memory
# class EnhancedAgent:
#     def __init__(self, stm, hitl, vector_db):
#         self.short_term_memory = stm
#         self.hitl = hitl
#         self.vector_db = vector_db
        
#     async def process_task(self, task):
#         # Get context from both memories
#         stm_context = self.short_term_memory.get_recent_context()
#         ltm_context = self.vector_db.similar_search(task)
        
#         # Process with context
#         result = await self.process_with_context(task, stm_context, ltm_context)
        
#         # Request human review if needed
#         importance = self.assess_importance(result)
#         human_feedback = await self.hitl.request_human_review(result, importance)
        
#         if human_feedback:
#             result = self.incorporate_feedback(result, human_feedback)
            
#         # Store in appropriate memory
#         self.short_term_memory.add_memory(result, self.id, importance)
#         if importance >= PERSISTENCE_THRESHOLD:
#             self.vector_db.add(result)
            
#         return result

# from typing import Dict, List, Optional, Any
# from pydantic import BaseModel
# from enum import Enum
# from datetime import datetime
# import asyncio
# from fastapi import HTTPException, WebSocket
# from .human_checkpoint import CheckpointType, CheckpointStatus, HumanCheckpoint

# class VerificationType(str, Enum):
#     FACTUAL_ACCURACY = "factual_accuracy"
#     SAFETY_CHECK = "safety_check"
#     ETHICAL_REVIEW = "ethical_review"
#     QUALITY_CHECK = "quality_check"
#     COMPLETENESS = "completeness"

# class VerificationRequest(BaseModel):
#     request_id: str
#     task_id: str
#     verification_type: VerificationType
#     content: Dict[str, Any]
#     priority: int = 1
#     deadline: Optional[datetime] = None

# class VerificationResult(BaseModel):
#     request_id: str
#     status: CheckpointStatus