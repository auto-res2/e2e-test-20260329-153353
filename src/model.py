"""
LLM wrapper for making API calls to OpenAI, Anthropic, or local models.
Handles prompt formatting and response parsing.
"""

import os
from typing import Optional, Dict, Any
import time


class LLMWrapper:
    """
    Unified wrapper for LLM API calls.
    Supports OpenAI, Anthropic, and extensible to local models.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 450,
        temperature: float = 0.7,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 30,
    ):
        """
        Initialize LLM wrapper.

        Args:
            provider: 'openai', 'anthropic', or 'local'
            model_name: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            api_key_env: Environment variable name for API key
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Initialize API client based on provider
        if provider == "openai":
            import openai

            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found in environment variable: {api_key_env}"
                )
            self.client = openai.OpenAI(api_key=api_key)

        elif provider == "anthropic":
            import anthropic

            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found in environment variable: {api_key_env}"
                )
            self.client = anthropic.Anthropic(api_key=api_key)

        elif provider == "local":
            # For local models, could use vLLM, llama.cpp, etc.
            # Placeholder for now
            self.client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.call_count = 0
        self.total_tokens = 0

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, max_tok, temp, system_prompt)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, max_tok, temp, system_prompt)
            elif self.provider == "local":
                return self._generate_local(prompt, max_tok, temp, system_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            raise

    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout,
        )

        self.call_count += 1
        self.total_tokens += response.usage.total_tokens

        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate using Anthropic API."""
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        self.call_count += 1
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        return response.content[0].text

    def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate using local model (placeholder)."""
        raise NotImplementedError("Local model generation not yet implemented")

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "average_tokens_per_call": self.total_tokens / max(1, self.call_count),
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.call_count = 0
        self.total_tokens = 0
