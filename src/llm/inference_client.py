import json
import logging
import re
from typing import Dict, Any

from openai import AsyncOpenAI
from src.config import RunConfig

logger = logging.getLogger(__name__)

def clean_json_string(raw_text: str) -> str:
    """
    Strip markdown fences (e.g. ```json ) if they exist to extract raw JSON.
    """
    text = raw_text.strip()
    
    # Simple regex to extract content within json code blocks if present
    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback manual stripping if regex missed
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline+1:]
        if text.endswith("```"):
            text = text[:-3]
    
    return text.strip()

class InferenceClient:
    def __init__(self, config: RunConfig):
        self.api_key = config.inference_api_key
        self.base_url = config.inference_base_url
        
        if not self.api_key or self.api_key == "***":
            # Just log it; some local mock endpoints might not need keys
            logger.warning("No API_KEY provided to InferenceClient.")

        self.client = AsyncOpenAI(
            api_key=self.api_key if self.api_key else "dummy-key",
            base_url=self.base_url,
            # Add timeout limits to prevent hanging connections
            timeout=60.0,
            max_retries=2
        )

    async def generate_json_async(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        seed: int = 42,
        stage: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute an LLM call enforcing valid JSON output via strict prompt adherence and retry loops.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return await self._call_and_parse(model, messages, temperature, max_tokens, seed)
        except Exception as e:
            logger.debug(f"JSON parsing failed on first attempt for stage '{stage}': {e}. Retrying.")
            # Append strict manual JSON constraints and retry exactly once
            retry_msg = "Return valid JSON only. No extra text. No markdown fences. No trailing commas."
            messages.append({"role": "user", "content": retry_msg})
            
            try:
                return await self._call_and_parse(model, messages, temperature, max_tokens, seed)
            except Exception as e2:
                logger.error(f"Inference failed after retry for stage '{stage}': {e2}")
                return {"error": "inference_failed", "stage": stage}

    async def _call_and_parse(self, model: str, messages: list, temperature: float, max_tokens: int, seed: int) -> Dict[str, Any]:
        """
        Private wrapper to execute the HTTP request and decode JSON safely.
        """
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
            # Omitted response_format={"type": "json_object"} because some external
            # endpoints (nscale, openrouter depending on model) don't strictly support it reliably.
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from LLM.")
            
        cleaned = clean_json_string(content)
        return json.loads(cleaned)
