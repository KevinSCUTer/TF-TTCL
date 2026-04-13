"""
LLM Client - SiliconFlow API Abstraction
Responsible for communicating with vLLM services, retrieving model responses and logprobs.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM Response Data Structure"""
    content: str                      # Generated text content
    logprobs: List[float]            # Logprob for each token
    tokens: List[str]                # Generated tokens
    total_tokens: int                # Total tokens
    prompt_tokens: int               # Prompt tokens
    completion_tokens: int           # Completion tokens


class LLMClient:
    """
    LLM Client Wrapper
    - Wraps SiliconFlow/vLLM API calls.
    - Configuration must set logprobs=True to retrieve token probabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        model_name: str = "",
        default_max_tokens: int = 512
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key, defaults to OPENAI_API_KEY environment variable if None.
            base_url: API service URL.
            model_name: Model name.
            default_max_tokens: Default max tokens for generation.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder")
        self.base_url = base_url
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        logprobs: bool = True
    ) -> LLMResponse:
        """
        Generate response.
        
        Args:
            messages: Message list (OpenAI format).
            temperature: Sampling temperature.
            top_p: Top-P sampling parameter.
            top_k: Top-K sampling parameter (vLLM supported).
            max_tokens: Maximum tokens to generate.
            logprobs: Whether to return logprobs (required for PPL calculation).
            
        Returns:
            LLMResponse object.
        """
        max_tokens = max_tokens or self.default_max_tokens
        
        # Build request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # Add logprobs parameter
        if logprobs:
            request_params["logprobs"] = True
            # Attempt to set top_logprobs=1; some APIs require this for logprob responses.
            request_params["top_logprobs"] = 1
        
        # Add optional sampling parameters
        if top_p is not None:
            request_params["top_p"] = top_p
        
        # vLLM supports top_k via extra_body
        if top_k is not None:
            request_params["extra_body"] = {"top_k": top_k}
        
        # Call API
        response = self.client.chat.completions.create(**request_params)
        
        # Parse response
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Parse logprobs
        logprobs_list = []
        tokens_list = []
        
        if logprobs:
            if not choice.logprobs:
                logger.warning(f"LLMClient: logprobs requested but not returned by API (model: {self.model_name}). This will cause PPL to be inf.")
            elif not choice.logprobs.content:
                logger.warning(f"LLMClient: logprobs.content is empty (model: {self.model_name})")
            else:
                for token_info in choice.logprobs.content:
                    logprobs_list.append(token_info.logprob)
                    tokens_list.append(token_info.token)
        
        # Get token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else len(tokens_list)
        total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens
        
        return LLMResponse(
            content=content,
            logprobs=logprobs_list,
            tokens=tokens_list,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Simple estimate; consider tiktoken for precise requirements.
        
        Args:
            text: Text to estimate.
            
        Returns:
            Estimated token count.
        """
        # Rough estimate: ~3 characters per token.
        return max(1, len(text) // 3)
    
    def build_messages(
        self,
        system_prompt: str,
        user_content: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build message list.
        
        Args:
            system_prompt: System prompt.
            user_content: User input content.
            history: Optional conversation history.
            
        Returns:
            Formatted message list.
        """
        messages = []
        
        # Add system message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_content})
        
        return messages