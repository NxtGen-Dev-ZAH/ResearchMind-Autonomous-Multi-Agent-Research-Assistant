from typing import Dict
import asyncio
class ResearchAgent:
    """
    Specialized agent for comprehensive information gathering
    """
    def __init__(self, 
                 ollama_interface, 
                 short_term_memory, 
                 vector_database,
                 web_search):
        self.ollama = ollama_interface
        self.memory = short_term_memory
        self.vector_db = vector_database
        self.web_search = web_search
    
    async def gather_information(self, state: Dict) -> Dict:
        """
        Advanced information gathering across multiple sources
        """
        query_refinement_prompt = f"""
        Refine and expand research query for comprehensive information gathering:
        
        Original Query: {state.get('input', '')}
        
        Tasks:
        1. Generate 5-7 precise sub-queries
        2. Create semantic variations
        3. Identify potential research angles
        4. Suggest diverse information sources
        """
        
        # Generate refined queries
        refined_queries = await self.ollama.generate(
            prompt=query_refinement_prompt,
            max_tokens=300
        )
        
        # Parse refined queries
        queries = [q.strip() for q in refined_queries.split('\n') if q.strip()]
        
        # Parallel web search
        search_tasks = [self.web_search.search(q, num_results=3) for q in queries]
        search_results = await asyncio.gather(*search_tasks)
        
        # Consolidate and analyze results
        comprehensive_findings = []
        for query, results in zip(queries, search_results):
            for result in results:
                # Extract and summarize content
                content = await self.web_search.extract_content(result['link'])
                
                summary_prompt = f"""
                Summarize and assess relevance:
                Query: {query}
                Title: {result['title']}
                Content: {content[:1000]}  # Limit content length
                
                Provide:
                1. Key insights
                2. Relevance score (0-100)
                3. Critical information extracts
                """
                
                summary = await self.ollama.generate(
                    prompt=summary_prompt,
                    max_tokens=300
                )
                
                comprehensive_findings.append({
                    'query': query,
                    'title': result['title'],
                    'link': result['link'],
                    'summary': summary
                })
        
        # Optional: Store embeddings in vector database
        if self.vector_db:
            embeddings = await self.ollama.embed(
                [f['summary'] for f in comprehensive_findings]
            )
            # Store embeddings (implementation depends on vector DB)
        
        return {
            'research_findings': comprehensive_findings,
            'total_sources': len(comprehensive_findings)
        }
# from typing import Dict, List, Optional
# from pydantic import BaseModel
# import asyncio
# from tavily import TavilyClient
# from langchain_community.chat_models import ChatMistralAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from chromadb import PersistentClient

# class SearchResult(BaseModel):
#     source: str
#     content: str
#     relevance_score: float
#     timestamp: float

# class ResearchAgent:
#     def __init__(self):
#         self.tavily_client = TavilyClient(api_key="your_tavily_api_key")
#         self.llm = ChatMistralAI(
#             mistral_api_key="your_mistral_key",
#             model="mistral-medium",
#             temperature=0.1
#         )
#         self.vector_db = PersistentClient(path="./chroma_db")
#         self.collection = self.vector_db.get_collection("research_results")
#         self.search_history: List[SearchResult] = []

#     async def conduct_research(self, query: str) -> Dict:
#         """
#         Conduct initial research on the given query
#         """
#         # Step 1: Break down query into search components
#         search_components = await self._decompose_query(query)
        
#         # Step 2: Execute parallel searches using Tavily
#         search_tasks = [
#             self._execute_search(component)
#             for component in search_components
#         ]
#         search_results = await asyncio.gather(*search_tasks)
        
#         # Step 3: Filter and rank results
#         filtered_results = self._filter_results(search_results)
        
#         # Step 4: Store in vector database
#         self._store_results(filtered_results, query)
        
#         return {
#             'query': query,
#             'results': filtered_results,
#             'sources': [r.source for r in filtered_results]
#         }

#     async def _decompose_query(self, query: str) -> List[str]:
#         """Break down complex query into searchable components"""
#         messages = [
#             SystemMessage(content="Break down this research query into specific searchable components. Return as a list of strings."),
#             HumanMessage(content=query)
#         ]
#         response = await self.llm.ainvoke(messages)
#         components = self._parse_components(response.content)
#         return components

#     async def _execute_search(self, query_component: str) -> List[SearchResult]:
#         """Execute search using Tavily API"""
#         try:
#             # Use Tavily's search API
#             search_response = await self.tavily_client.search(
#                 query_component,
#                 search_depth="advanced",
#                 max_results=5
#             )
            
#             results = []
#             for result in search_response['results']:
#                 results.append(SearchResult(
#                     source=result['url'],
#                     content=result['content'],
#                     relevance_score=result['relevance_score'],
#                     timestamp=result['timestamp']
#                 ))
            
#             return results
            
#         except Exception as e:
#             print(f"Search error: {e}")
#             return []

#     def _filter_results(self, search_results: List[List[SearchResult]]) -> List[SearchResult]:
#         """Filter and rank search results"""
#         # Flatten results
#         all_results = [result for sublist in search_results for result in sublist]
        
#         # Sort by relevance score
#         sorted_results = sorted(
#             all_results,
#             key=lambda x: x.relevance_score,
#             reverse=True
#         )
        
#         # Take top N results
#         return sorted_results[:10]

#     def _store_results(self, results: List[SearchResult], query: str):
#         """Store results in ChromaDB"""
#         documents = [result.content for result in results]
#         metadatas = [{"source": result.source, "query": query} for result in results]
#         ids = [f"result_{i}" for i in range(len(results))]
        
#         self.collection.add(
#             documents=documents,
#             metadatas=metadatas,
#             ids=ids
#         )

#     def _parse_components(self, llm_response: str) -> List[str]:
#         """Parse LLM response into list of search components"""
#         # Remove any markdown formatting
#         clean_response = llm_response.replace("*", "").replace("-", "").strip()
        
#         # Split into components
#         components = [
#             component.strip()
#             for component in clean_response.split("\n")
#             if component.strip()
#         ]
        
#         return components

#     async def get_similar_research(self, query: str, n_results: int = 5) -> List[Dict]:
#         """Retrieve similar past research results"""
#         results = self.collection.query(
#             query_texts=[query],
#             n_results=n_results
#         )
        
#         return [{
#             "content": result.document,
#             "metadata": result.metadata
#         } for result in results]