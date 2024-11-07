from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel
from langchain_community.chat_models import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

class LLMProvider(str, Enum):
    MISTRAL = "mistral"
    GEMINI = "gemini"

class LLMResponse(BaseModel):
    content: str
    provider: LLMProvider
    metadata: Dict = {}

class LLMSet:
    """Manages multiple LLM providers and handles fallback logic"""
    
    def __init__(self):
        self.llms = {}
        self._initialize_llms()
        
    def _initialize_llms(self):
        """Initialize available LLM providers"""
        # Initialize Mistral
        try:
            self.llms[LLMProvider.MISTRAL] = ChatMistralAI(
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                model="mistral-medium",
                temperature=0.1,
                max_tokens=4096
            )
        except Exception as e:
            print(f"Failed to initialize Mistral: {e}")

        # Initialize Gemini
        try:
            self.llms[LLMProvider.GEMINI] = ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model="gemini-pro",
                temperature=0.1,
                max_output_tokens=4096
            )
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """
        Generate response using available LLMs with fallback logic
        """
        providers = [preferred_provider] if preferred_provider else list(self.llms.keys())
        
        for provider in providers:
            try:
                llm = self.llms.get(provider)
                if not llm:
                    continue

                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))

                response = await llm.ainvoke(messages)
                
                return LLMResponse(
                    content=response.content,
                    provider=provider,
                    metadata={
                        "model": llm.model if hasattr(llm, 'model') else "unknown",
                        "success": True
                    }
                )
            except Exception as e:
                print(f"Error with {provider}: {e}")
                continue

        raise Exception("All LLM providers failed")

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[LLMProvider] = None
    ):
        """
        Stream responses from LLM
        """
        providers = [preferred_provider] if preferred_provider else list(self.llms.keys())
        
        for provider in providers:
            try:
                llm = self.llms.get(provider)
                if not llm:
                    continue

                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))

                async for chunk in llm.astream(messages):
                    yield {
                        "content": chunk.content,
                        "provider": provider,
                        "finished": False
                    }
                
                yield {
                    "content": "",
                    "provider": provider,
                    "finished": True
                }
                return
                
            except Exception as e:
                print(f"Error streaming with {provider}: {e}")
                continue

        raise Exception("All LLM providers failed")

    def get_provider_status(self) -> Dict[str, bool]:
        """
        Get status of all LLM providers
        """
        return {
            provider: provider in self.llms
            for provider in LLMProvider
        }

    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[LLMProvider] = None
    ) -> List[LLMResponse]:
        """
        Generate multiple responses in parallel
        """
        import asyncio
        
        tasks = [
            self.generate(
                prompt,
                system_prompt,
                preferred_provider
            )
            for prompt in prompts
        ]
        
        return await asyncio.gather(*tasks)