import collections
import time
import uuid
from typing import Any, Dict, List, Optional

class ShortTermMemory:
    def __init__(self, max_size: int = 1000):
        """
        Initialize Short-Term Memory with a maximum size
        
        Args:
            max_size (int): Maximum number of memory items to store
        """
        self._memory = collections.deque(maxlen=max_size)
        self._index = {}
    
    def add(self, 
            content: Any, 
            agent_id: str, 
            category: str = 'default',
            priority: int = 5) -> str:
        """
        Add a memory item
        
        Args:
            content (Any): Content to be stored
            agent_id (str): ID of the agent creating the memory
            category (str): Categorization of the memory
            priority (int): Priority of the memory item (1-10)
        
        Returns:
            str: Unique identifier for the memory item
        """
        memory_id = str(uuid.uuid4())
        memory_item = {
            'id': memory_id,
            'content': content,
            'agent_id': agent_id,
            'category': category,
            'priority': priority,
            'timestamp': time.time()
        }
        
        self._memory.append(memory_item)
        
        # Update index for quick retrieval
        if category not in self._index:
            self._index[category] = []
        self._index[category].append(memory_id)
        
        return memory_id
    
    def get(self, 
            category: Optional[str] = None, 
            max_age: Optional[float] = None,
            min_priority: int = 1) -> List[Dict]:
        """
        Retrieve memory items
        
        Args:
            category (str, optional): Specific category to retrieve
            max_age (float, optional): Maximum age of memories in seconds
            min_priority (int): Minimum priority of memories to retrieve
        
        Returns:
            List of memory items matching criteria
        """
        current_time = time.time()
        filtered_memories = []
        
        for memory in self._memory:
            # Check category
            if category and memory['category'] != category:
                continue
            
            # Check age
            if max_age and (current_time - memory['timestamp']) > max_age:
                continue
            
            # Check priority
            if memory['priority'] < min_priority:
                continue
            
            filtered_memories.append(memory)
        
        return filtered_memories
    
    def remove(self, memory_id: str) -> bool:
        """
        Remove a specific memory item
        
        Args:
            memory_id (str): Unique identifier of the memory item
        
        Returns:
            bool: Whether the item was successfully removed
        """
        for memory in self._memory:
            if memory['id'] == memory_id:
                self._memory.remove(memory)
                
                # Remove from index
                for category, ids in self._index.items():
                    if memory_id in ids:
                        ids.remove(memory_id)
                
                return True
        return False
    
    def clear(self, category: Optional[str] = None):
        """
        Clear memories, optionally by category
        
        Args:
            category (str, optional): Specific category to clear
        """
        if category:
            # Remove memories of specific category
            self._memory = collections.deque(
                [m for m in self._memory if m['category'] != category],
                maxlen=self._memory.maxlen
            )
            # Clear index for this category
            if category in self._index:
                del self._index[category]
        else:
            # Clear all memories
            self._memory.clear()
            self._index.clear()
    
    def __len__(self):
        """Return the number of memory items"""
        return len(self._memory)
    







    
# Example memory structure
# class ShortTermMemory:
#     def __init__(self, max_size=1000):
#         self.memory = deque(maxlen=max_size)
#         self.timestamp = time.time()
        
#     def add_memory(self, content, agent_id, priority):
#         memory_item = {
#             'content': content,
#             'agent_id': agent_id,
#             'timestamp': time.time(),
#             'priority': priority
#         }
#         self.memory.append(memory_item)
        
#     def get_recent_context(self, timeframe_minutes=30):
#         current_time = time.time()
#         return [m for m in self.memory 
#                if (current_time - m['timestamp']) < timeframe_minutes * 60]