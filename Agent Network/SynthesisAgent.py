from typing import Dict

class SynthesisAgent:
    """
    Final synthesis and report generation
    """
    def __init__(self, 
                 ollama_interface, 
                 short_term_memory, 
                 vector_database):
        self.ollama = ollama_interface
        self.memory = short_term_memory
        self.vector_db = vector_database
    
    async def synthesize_report(self, state: Dict) -> Dict:
        """
        Generate comprehensive final report
        """
        synthesis_prompt = f"""
        Synthesize a comprehensive research report:
        
        Original Query: {state.get('input', '')}
        Analysis Findings: {state.get('comprehensive_analysis', '')}
        Key Insights: {state.get('key_insights', '')}
        
        Report Structure:
        1. Executive Summary
        2. Detailed Findings
        3. Analytical Insights
        4. Recommendations
        5. Future Research Directions
        
        Ensure cohesive, structured, and insightful presentation
        """
        
        final_report = await self.ollama.generate(
            prompt=synthesis_prompt,
            max_tokens=2000
        )
        
        # Generate executive summary
        summary_prompt = f"""
        Condense the following report into a concise executive summary:
        {final_report}
        
        Focus on:
        - Core findings
        - Key implications
        - Strategic recommendations
        """
        
        executive_summary = await self.ollama.generate(
            prompt=summary_prompt,
            max_tokens=300
        )
        
        return {
            'final_report': final_report,
            'executive_summary': executive_summary,
            'report_complexity': len(final_report)
        }