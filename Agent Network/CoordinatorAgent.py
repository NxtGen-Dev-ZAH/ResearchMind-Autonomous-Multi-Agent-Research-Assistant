import asyncio
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


class CoordinatorAgent:
    """
    Centralized coordinator for managing multi-agent research workflow
    """

    def __init__(self, ollama_interface, short_term_memory, vector_database):
        self.ollama = ollama_interface
        self.memory = short_term_memory
        self.vector_db = vector_database

    async def route_task(self, state: Dict) -> str:
        """
        Intelligent task routing based on current state
        """
        routing_prompt = f"""
        Analyze the current research state and determine the most appropriate 
        next agent to handle the task:

        Current State:
        Input Query: {state.get('input', '')}
        Current Progress: {state.get('intermediate_steps', [])}
        Research Findings: {len(state.get('research_findings', []))}

        Possible Routes:
        1. research_agent: Detailed information gathering
        2. analysis_agent: In-depth analysis of collected data
        3. synthesis_agent: Combining insights
        4. qa_agent: Quality assurance and validation
        5. END: Workflow completion

        Provide the most appropriate next route.
        """

        route = await self.ollama.generate(prompt=routing_prompt, max_tokens=100)

        # Normalize routing decision
        route = route.strip().lower()
        valid_routes = [
            "research_agent",
            "analysis_agent",
            "synthesis_agent",
            "qa_agent",
            "end",
        ]

        # Fallback to default routing if uncertain
        if not any(r in route for r in valid_routes):
            return "research_agent"

        for valid_route in valid_routes:
            if valid_route in route:
                return valid_route

        return "research_agent"

    async def evaluate_progress(self, state: Dict) -> Dict:
        """
        Evaluate overall research progress and determine next steps
        """
        evaluation_prompt = f"""
        Comprehensive Evaluation of Research Progress:

        Original Query: {state.get('input', '')}
        Current Findings: {len(state.get('research_findings', []))}
        Intermediate Steps: {state.get('intermediate_steps', [])}

        Evaluate:
        1. Depth of research findings
        2. Relevance to original query
        3. Completeness of information
        4. Need for additional research
        5. Potential knowledge gaps

        Provide a detailed assessment with recommendations.
        """

        evaluation = await self.ollama.generate(
            prompt=evaluation_prompt, max_tokens=500
        )

        # Parse evaluation for decision-making
        return {
            "evaluation": evaluation,
            "needs_more_research": len(state.get("research_findings", [])) < 5,
            "quality_score": len(state.get("research_findings", [])) * 2,
        }


# # from typing import Dict, List, Optional
# from pydantic import BaseModel
# from fastapi import FastAPI
# from langchain_core.runnables import Runnable, RunnableConfig
# from langchain_community.chat_models import ChatMistralAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from langgraph.graph import Graph, StateGraph
# from chromadb import PersistentClient
# import asyncio

# class ResearchTask(BaseModel):
#     task_id: str
#     query: str
#     status: str
#     assigned_agent: Optional[str]
#     results: Optional[Dict]
#     priority: int = 1

# class CoordinatorState(BaseModel):
#     task: ResearchTask
#     research_results: Optional[Dict] = None
#     analysis_results: Optional[Dict] = None
#     qa_results: Optional[Dict] = None
#     final_results: Optional[Dict] = None
#     needs_human_verification: bool = False

# class CoordinatorAgent:
#     def __init__(self):
#         self.tasks: Dict[str, ResearchTask] = {}
#         self.llm = ChatMistralAI(
#             mistral_api_key="your_mistral_key",
#             model="mistral-medium",
#             temperature=0.1
#         )
#         # Initialize ChromaDB
#         self.vector_db = PersistentClient(path="./chroma_db")
#         self.collection = self.vector_db.create_collection(
#             name="research_results",
#             metadata={"hnsw:space": "cosine"}
#         )

#         # Initialize the research workflow graph
#         self.workflow = self._create_workflow_graph()

#     def _create_workflow_graph(self) -> StateGraph:
#         """Create the research workflow using LangGraph"""
#         # Define the graph
#         workflow = StateGraph(CoordinatorState)

#         # Add nodes for each agent
#         workflow.add_node("research", self._research_node())
#         workflow.add_node("analysis", self._analysis_node())
#         workflow.add_node("qa", self._qa_node())
#         workflow.add_node("synthesis", self._synthesis_node())
#         workflow.add_node("human_verification", self._human_verification_node())

#         # Define the edges
#         workflow.add_edge("research", "analysis")
#         workflow.add_edge("analysis", "qa")

#         # Conditional edges based on QA results
#         workflow.add_conditional_edges(
#             "qa",
#             self._should_verify_human,
#             {
#                 True: "human_verification",
#                 False: "synthesis"
#             }
#         )

#         workflow.add_edge("human_verification", "synthesis")

#         # Set the entry point
#         workflow.set_entry_point("research")

#         return workflow.compile()

#     def _research_node(self) -> Runnable:
#         """Create research node logic"""
#         async def research_task(state: CoordinatorState, config: RunnableConfig) -> CoordinatorState:
#             from .research_agent import ResearchAgent
#             agent = ResearchAgent()
#             state.research_results = await agent.conduct_research(state.task.query)
#             return state

#         return Runnable(research_task)

#     def _analysis_node(self) -> Runnable:
#         """Create analysis node logic"""
#         async def analysis_task(state: CoordinatorState, config: RunnableConfig) -> CoordinatorState:
#             from .analysis_agent import AnalysisAgent
#             agent = AnalysisAgent()
#             state.analysis_results = await agent.analyze_information(state.research_results)
#             return state

#         return Runnable(analysis_task)

#     def _qa_node(self) -> Runnable:
#         """Create QA node logic"""
#         async def qa_task(state: CoordinatorState, config: RunnableConfig) -> CoordinatorState:
#             from .qa_agent import QAAgent
#             agent = QAAgent()
#             qa_results = await agent.verify_results(state.analysis_results)
#             state.qa_results = qa_results
#             state.needs_human_verification = qa_results.get('needs_human_verification', False)
#             return state

#         return Runnable(qa_task)

#     def _synthesis_node(self) -> Runnable:
#         """Create synthesis node logic"""
#         async def synthesis_task(state: CoordinatorState, config: RunnableConfig) -> CoordinatorState:
#             from .synthesis_agent import SynthesisAgent
#             agent = SynthesisAgent()
#             state.final_results = await agent.synthesize_information(
#                 state.research_results,
#                 state.analysis_results,
#                 state.qa_results
#             )
#             return state

#         return Runnable(synthesis_task)

#     def _should_verify_human(self, state: CoordinatorState) -> bool:
#         """Determine if human verification is needed"""
#         return state.needs_human_verification

#     async def create_research_task(self, query: str) -> str:
#         """Create a new research task"""
#         task_id = f"task_{len(self.tasks) + 1}"
#         task = ResearchTask(
#             task_id=task_id,
#             query=query,
#             status="initialized"
#         )
#         self.tasks[task_id] = task
#         return task_id

#     async def execute_research(self, task_id: str):
#         """Execute the research workflow"""
#         task = self.tasks[task_id]
#         initial_state = CoordinatorState(task=task)

#         # Execute the workflow
#         final_state = await self.workflow.invoke(initial_state)

#         # Store results in vector database
#         self._store_in_vector_db(task_id, final_state.final_results)

#         # Update task status
#         self.tasks[task_id].status = "completed"
#         self.tasks[task_id].results = final_state.final_results

#         return final_state.final_results

#     def _store_in_vector_db(self, task_id: str, results: Dict):
#         """Store results in ChromaDB"""
#         # Convert results to embeddings and store
#         self.collection.add(
#             documents=[str(results)],
#             metadatas=[{"task_id": task_id}],
#             ids=[task_id]
#         )

# app = FastAPI()

# @app.post("/research/create")
# async def create_research(query: str):
#     coordinator = CoordinatorAgent()
#     task_id = await coordinator.create_research_task(query)
#     return {"task_id": task_id}

# @app.get("/research/{task_id}")
# async def get_research_status(task_id: str):
#     coordinator = CoordinatorAgent()
#     return await coordinator.get_task_status(task_id)

# @app.post("/research/{task_id}/execute")
# async def execute_research(task_id: str):
#     coordinator = CoordinatorAgent()
#     results = await coordinator.execute_research(task_id)
#     return results


# LangChain for LLM operations
# LangGraph for workflow orchestration
# FastAPI for the REST API
# Mistral AI as the LLM provider
# Pinecone as the vector database
# Tavily for web search

# Key features:

# Graph-based workflow management
# Asynchronous operations
# Vector storage for efficient retrieval
# REST API endpoints
# Human-in-the-loop verification
# Parallel search execution
