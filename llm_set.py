"""
llm_set.py: Manages different LLM configurations and providers
"""
from typing import Dict, Optional, Any
from langchain_community.chat_models import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.1
    streaming: bool = False
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None

class LLMSet:
    """Manages multiple LLM providers and configurations"""
    
    def __init__(self):
        self.active_provider = "mistral"  # default provider
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
    def get_llm(self, config: Optional[LLMConfig] = None) -> Any:
        """Get LLM instance based on configuration"""
        if config is None:
            config = self._get_default_config()
            
        if config.provider == "mistral":
            return ChatMistralAI(
                mistral_api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                streaming=config.streaming,
                callback_manager=self.callback_manager if config.streaming else None
            )
            
        elif config.provider == "gemini":
            return ChatGoogleGenerativeAI(
                google_api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                streaming=config.streaming,
                callback_manager=self.callback_manager if config.streaming else None
            )
            
        elif config.provider == "openai":
            return ChatOpenAI(
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                streaming=config.streaming,
                callback_manager=self.callback_manager if config.streaming else None
            )
            
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def _get_default_config(self) -> LLMConfig:
        """Get default LLM configuration"""
        return LLMConfig(
            provider="mistral",
            model="mistral-medium",
            temperature=0.1,
            streaming=False
        )

    def get_provider_configs(self) -> Dict[str, Dict]:
        """Get available provider configurations"""
        return {
            "mistral": {
                "models": ["mistral-tiny", "mistral-small", "mistral-medium"],
                "max_tokens": 4096,
                "streaming_support": True
            },
            "gemini": {
                "models": ["gemini-pro"],
                "max_tokens": 8192,
                "streaming_support": True
            },
            "openai": {
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "max_tokens": {
                    "gpt-3.5-turbo": 4096,
                    "gpt-4": 8192
                },
                "streaming_support": True
            }
        }

    async def get_model_performance(self, query: str) -> Dict[str, float]:
        """Test performance across different models"""
        results = {}
        test_configs = [
            LLMConfig(provider="mistral", model="mistral-medium"),
            LLMConfig(provider="gemini", model="gemini-pro"),
            LLMConfig(provider="openai", model="gpt-3.5-turbo")
        ]
        
        for config in test_configs:
            try:
                llm = self.get_llm(config)
                start_time = time.time()
                await llm.ainvoke(query)
                elapsed = time.time() - start_time
                results[f"{config.provider}-{config.model}"] = elapsed
            except Exception as e:
                results[f"{config.provider}-{config.model}"] = None
                print(f"Error testing {config.provider}-{config.model}: {e}")
                
        return results