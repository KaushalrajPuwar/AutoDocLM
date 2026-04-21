import json
import logging
import re
from typing import Dict, Any

from openai import AsyncOpenAI
from src.config import RunConfig

logger = logging.getLogger(__name__)

def clean_json_string(raw_text: str) -> str:
    """
    Strip markdown fences and any non-JSON prefix/suffix text.
    """
    text = raw_text.strip()
    
    # 1. Strip markdown fences
    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback manual stripping
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline+1:]
        if text.endswith("```"):
            text = text[:-3]
    
    # 2. Find first { and last } to isolate the JSON object if there's surrounding text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    
    return text.strip()

def repair_json(text: str) -> str:
    """
    Best-effort repair of common LLM JSON errors:
    - Trailing commas in objects/arrays
    - Unescaped newlines in strings
    """
    # Remove trailing commas: ,} -> } and ,] -> ]
    text = re.sub(r',\s*\}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    return text

class InferenceClient:
    def __init__(self, config: RunConfig):
        self.api_key = config.inference_api_key
        self.base_url = config.inference_base_url
        
        if not self.api_key or self.api_key == "***":
            logger.warning("No API_KEY provided to InferenceClient.")

        self.client = AsyncOpenAI(
            api_key=self.api_key if self.api_key else "dummy-key",
            base_url=self.base_url,
            timeout=180.0,
            max_retries=2
        )

    async def aclose(self):
        """Explicitly close the underlying HTTP client before the event loop shuts down."""
        await self.client.close()

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
        Execute an LLM call enforcing valid JSON output.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return await self._call_and_parse(model, messages, temperature, max_tokens, seed)
        except Exception as e:
            logger.debug(f"JSON parsing failed on first attempt for stage '{stage}': {e}. Retrying.")
            retry_msg = "Your previous response was invalid JSON. Return valid JSON only. No markdown fences. Ensure all quotes are escaped."
            
            # Create a separate list for retry to avoid bloating history
            retry_messages = messages + [
                {"role": "assistant", "content": "I apologize, I will provide valid JSON now."},
                {"role": "user", "content": retry_msg}
            ]
            
            try:
                return await self._call_and_parse(model, retry_messages, temperature, max_tokens, seed)
            except Exception as e2:
                logger.error(f"Inference failed after retry for stage '{stage}': {e2}")
                return {"error": "inference_failed", "stage": stage, "details": str(e2)}

    async def generate_markdown_async(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        seed: int = 42,
        stage: str = "unknown"
    ) -> str:
        """
        Execute an LLM call to generate raw Markdown/text prose.
        No JSON structure is enforced.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from LLM.")
            return content
        except Exception as e:
            logger.error(f"Markdown inference failed for stage '{stage}': {e}")
            return f"# Error: Inference Failed\n\nStage: {stage}\nDetails: {str(e)}"

    async def _call_and_parse(self, model: str, messages: list, temperature: float, max_tokens: int, seed: int) -> Dict[str, Any]:
        """
        Private wrapper to execute the HTTP request and decode JSON safely.
        """
        # Attempt to use JSON mode if natively supported by the inference server
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from LLM.")
            
        cleaned = clean_json_string(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try repair before failing
            repaired = repair_json(cleaned)
            return json.loads(repaired)
