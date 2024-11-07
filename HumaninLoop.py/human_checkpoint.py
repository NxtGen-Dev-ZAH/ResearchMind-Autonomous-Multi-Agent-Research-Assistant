"""
human_checkpoint.py: Manages checkpoints requiring human verification
"""
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi import HTTPException
import asyncio
import time
from datetime import datetime

class CheckpointRequest(BaseModel):
    task_id: str
    checkpoint_type: str
    data: Dict
    priority: int = 1
    timeout: int = 3600  # 1 hour default timeout

class CheckpointResponse(BaseModel):
    approved: bool
    feedback: Optional[str] = None
    modifications: Optional[Dict] = None
    reviewer: Optional[str] = None

class HumanCheckpoint:
    def __init__(self):
        self.pending_checkpoints: Dict[str, CheckpointRequest] = {}
        self.completed_checkpoints: Dict[str, CheckpointResponse] = {}
        self.checkpoint_locks: Dict[str, asyncio.Lock] = {}
        
    async def create_checkpoint(self, request: CheckpointRequest) -> str:
        """Create a new checkpoint for human verification"""
        checkpoint_id = f"cp_{int(time.time())}_{request.task_id}"
        
        self.pending_checkpoints[checkpoint_id] = request
        self.checkpoint_locks[checkpoint_id] = asyncio.Lock()
        
        # Start timeout monitoring
        asyncio.create_task(self._monitor_timeout(checkpoint_id, request.timeout))
        
        return checkpoint_id
        
    async def await_verification(self, checkpoint_id: str) -> CheckpointResponse:
        """Wait for human verification of a checkpoint"""
        if checkpoint_id not in self.pending_checkpoints:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
            
        async with self.checkpoint_locks[checkpoint_id]:
            while checkpoint_id not in self.completed_checkpoints:
                await asyncio.sleep(1)
                
            response = self.completed_checkpoints[checkpoint_id]
            del self.pending_checkpoints[checkpoint_id]
            return response
            
    async def submit_verification(self, checkpoint_id: str, response: CheckpointResponse):
        """Submit human verification result"""
        if checkpoint_id not in self.pending_checkpoints:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
            
        async with self.checkpoint_locks[checkpoint_id]:
            self.completed_checkpoints[checkpoint_id] = response
            
    async def _monitor_timeout(self, checkpoint_id: str, timeout: int):
        """Monitor checkpoint for timeout"""
        await asyncio.sleep(timeout)
        
        if checkpoint_id in self.pending_checkpoints:
            async with self.checkpoint_locks[checkpoint_id]:
                self.completed_checkpoints[checkpoint_id] = CheckpointResponse(
                    approved=False,
                    feedback="Checkpoint verification timed out"
                )
                
    def get_pending_checkpoints(self, priority: Optional[int] = None) -> List[Dict]:
        """Get list of pending checkpoints"""
        checkpoints = []
        for cp_id, request in self.pending_checkpoints.items():
            if priority is None or request.priority >= priority:
                checkpoints.append({
                    "checkpoint_id": cp_id,
                    "task_id": request.task_id,
                    "type": request.checkpoint_type,
                    "priority": request.priority,
                    "created_at": datetime.fromtimestamp(int(cp_id.split('_')[1])),
                    "data": request.data
                })
        return sorted(checkpoints, key=lambda x: (-x["priority"], x["created_at"]))
    



# from typing import Dict, List, Optional, Any
# from pydantic import BaseModel
# from enum import Enum
# from datetime import datetime
# import asyncio
# from fastapi import HTTPException

# class CheckpointType(str, Enum):
#     INITIAL_REVIEW = "initial_review"
#     INTERMEDIATE = "intermediate"
#     FINAL_REVIEW = "final_review"
#     ERROR_CHECK = "error_check"
#     SAFETY_CHECK = "safety_check"

# class CheckpointStatus(str, Enum):
#     PENDING = "pending"
#     APPROVED = "approved"
#     REJECTED = "rejected"
#     NEEDS_REVISION = "needs_revision"

# class CheckpointData(BaseModel):
#     checkpoint_id: str
#     type: CheckpointType
#     task_id: str
#     data: Dict[str, Any]
#     status: CheckpointStatus
#     feedback: Optional[str] = None
#     created_at: datetime
#     reviewed_at: Optional[datetime] = None
#     reviewer_id: Optional[str] = None

# class HumanCheckpoint:
#     def __init__(self):
#         self.checkpoints: Dict[str, CheckpointData] = {}
#         self.pending_reviews: Dict[str, asyncio.Event] = {}
        
#     async def create_checkpoint(
#         self,
#         task_id: str,
#         checkpoint_type: CheckpointType,
#         data: Dict[str, Any]
#     ) -> str:
#         """
#         Create a new checkpoint for human review
#         """
#         checkpoint_id = f"checkpoint_{task_id}_{len(self.checkpoints)}"
        
#         checkpoint = CheckpointData(
#             checkpoint_id=checkpoint_id,
#             type=checkpoint_type,
#             task_id=task_id,
#             data=data,
#             status=CheckpointStatus.PENDING,
#             created_at=datetime.now()
#         )
        
#         self.checkpoints[checkpoint_id] = checkpoint
#         self.pending_reviews[checkpoint_id] = asyncio.Event()
        
#         return checkpoint_id

#     async def await_review(
#         self,
#         checkpoint_id: str,
#         timeout: Optional[float] = None
#     ) -> CheckpointData:
#         """
#         Wait for human review completion
#         """
#         if checkpoint_id not in self.checkpoints:
#             raise HTTPException(status_code=404, detail="Checkpoint not found")
            
#         event = self.pending_reviews[checkpoint_id]
#         try:
#             await asyncio.wait_for(event.wait(), timeout=timeout)
#             return self.checkpoints[checkpoint_id]
#         except asyncio.TimeoutError:
#             raise HTTPException(status_code=408, detail="Review timeout")

#     async def submit_review(
#         self,
#         checkpoint_id: str,
#         status: CheckpointStatus,
#         feedback: Optional[str] = None,
#         reviewer_id: Optional[str] = None
#     ) -> CheckpointData:
#         """
#         Submit human review for a checkpoint
#         """
#         if checkpoint_id not in self.checkpoints:
#             raise HTTPException(status_code=404, detail="Checkpoint not found")
            
#         checkpoint = self.checkpoints[checkpoint_id]
#         checkpoint.status = status
#         checkpoint.feedback = feedback
#         checkpoint.reviewed_at = datetime.now()
#         checkpoint.reviewer_id = reviewer_id
        
#         # Signal that review is complete
#         self.pending_reviews[checkpoint_id].set()
        
#         return checkpoint

#     async def get_checkpoint_status(
#         self,
#         checkpoint_id: str
#     ) -> CheckpointData:
#         """
#         Get current status of a checkpoint
#         """
#         if checkpoint_id not in self.checkpoints:
#             raise HTTPException(status_code=404, detail="Checkpoint not found")
            
#         return self.checkpoints[checkpoint_id]

#     async def get_pending_checkpoints(
#         self,
#         reviewer_id: Optional[str] = None
#     ) -> List[CheckpointData]:
#         """
#         Get list of pending checkpoints
#         """
#         pending = [
#             checkpoint for checkpoint in self.checkpoints.values()
#             if checkpoint.status == CheckpointStatus.PENDING
#             and (not reviewer_id or checkpoint.reviewer_id == reviewer_id)
#         ]
#         return pending

#     async def cancel_checkpoint(
#         self,
#         checkpoint_id: str
#     ):
#         """
#         Cancel a pending checkpoint
#         """
#         if checkpoint_id not in self.checkpoints:
#             raise HTTPException(status_code=404, detail="Checkpoint not found")
            
#         checkpoint = self.checkpoints[checkpoint_id]
#         if checkpoint.status != CheckpointStatus.PENDING:
#             raise HTTPException(status_code=400, detail="Checkpoint is not pending")
            
#         # Clean up
#         del self.checkpoints[checkpoint_id]
#         del self.pending_reviews[checkpoint_id]