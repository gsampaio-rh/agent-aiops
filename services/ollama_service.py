"""
Service layer for interacting with Ollama API.
"""

import time
import requests
from typing import Dict, Any, List, Optional, Iterator
import json
import streamlit as st

from config.settings import OLLAMA_BASE_URL, DEFAULT_MODEL
from utils.logger import get_logger, log_performance, log_ollama_request, log_ollama_response


class OllamaService:
    """Service class for Ollama API interactions."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.logger = get_logger(__name__)
    
    @log_performance("get_available_models")
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        self.logger.info("Fetching available models from Ollama", base_url=self.base_url)
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            
            self.logger.info(f"Retrieved {len(models)} models", models=models)
            
            # Ensure default model is available
            if DEFAULT_MODEL not in models and models:
                self.logger.warning(f"Default model '{DEFAULT_MODEL}' not in available models", 
                                  default_model=DEFAULT_MODEL, available_models=models)
                return models
            elif DEFAULT_MODEL not in models:
                self.logger.error(f"No models available and default model not found", 
                                default_model=DEFAULT_MODEL)
                st.error(f"Default model '{DEFAULT_MODEL}' not found. Please ensure it's installed in Ollama.")
                return []
            
            return models
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama API", error=str(e), base_url=self.base_url)
            st.error(f"Failed to connect to Ollama: {e}")
            return []
        except Exception as e:
            self.logger.exception(f"Unexpected error fetching models", error=str(e))
            st.error(f"Error fetching models: {e}")
            return []
    
    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion from Ollama.
        
        Args:
            model: Model name to use
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            
        Yields:
            Dict containing response chunks and metadata
        """
        start_time = time.time()
        
        # Log the request
        log_ollama_request(
            model=model,
            message_count=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        self.logger.info("Starting Ollama chat stream", 
                        model=model, 
                        message_count=len(messages),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p
            }
        }
        
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                
                full_response = ""
                input_tokens = 0
                output_tokens = 0
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                full_response += content
                                
                                # Prepare comprehensive metadata
                                metadata = {
                                    "model": model,
                                    "latency_ms": round((time.time() - start_time) * 1000),
                                    "done": chunk.get('done', False),
                                    "done_reason": chunk.get('done_reason', ''),
                                    "total_duration": chunk.get('total_duration', 0),
                                    "load_duration": chunk.get('load_duration', 0),
                                    "prompt_eval_count": chunk.get('prompt_eval_count', 0),
                                    "prompt_eval_duration": chunk.get('prompt_eval_duration', 0),
                                    "eval_count": chunk.get('eval_count', 0),
                                    "eval_duration": chunk.get('eval_duration', 0)
                                }
                                
                                yield {
                                    "content": content,
                                    "full_response": full_response,
                                    "done": chunk.get('done', False),
                                    "metadata": metadata
                                }
                                
                                if chunk.get('done'):
                                    input_tokens = chunk.get('prompt_eval_count', 0)
                                    output_tokens = chunk.get('eval_count', 0)
                                    
                                    # Log the final response metrics
                                    log_ollama_response(
                                        model=model,
                                        tokens_generated=output_tokens,
                                        duration_ms=round((time.time() - start_time) * 1000),
                                        input_tokens=input_tokens,
                                        total_duration=chunk.get('total_duration', 0),
                                        eval_duration=chunk.get('eval_duration', 0)
                                    )
                                    
                                    self.logger.info("Ollama chat stream completed",
                                                    model=model,
                                                    input_tokens=input_tokens,
                                                    output_tokens=output_tokens,
                                                    total_duration_ms=round((time.time() - start_time) * 1000))
                                    
                        except json.JSONDecodeError as e:
                            self.logger.warning("Failed to parse JSON chunk", error=str(e))
                            continue
                            
        except requests.RequestException as e:
            self.logger.error("Ollama request failed", error=str(e), model=model)
            yield {
                "error": f"Request failed: {e}",
                "content": "",
                "full_response": "",
                "done": True,
                "metadata": {
                    "model": model,
                    "latency_ms": round((time.time() - start_time) * 1000),
                    "done": True,
                    "done_reason": "error",
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0
                }
            }
        except Exception as e:
            self.logger.exception("Unexpected error in Ollama chat stream", error=str(e), model=model)
            yield {
                "error": f"Unexpected error: {e}",
                "content": "",
                "full_response": "",
                "done": True,
                "metadata": {
                    "model": model,
                    "latency_ms": round((time.time() - start_time) * 1000),
                    "done": True,
                    "done_reason": "error",
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0
                }
            }
    
    @log_performance("health_check")
    def health_check(self) -> bool:
        """Check if Ollama service is available."""
        self.logger.debug("Checking Ollama service health", base_url=self.base_url)
        
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            is_healthy = response.status_code == 200
            
            if is_healthy:
                self.logger.debug("Ollama service is healthy")
            else:
                self.logger.warning("Ollama service health check failed", status_code=response.status_code)
                
            return is_healthy
        except Exception as e:
            self.logger.error("Ollama service health check failed", error=str(e))
            return False
