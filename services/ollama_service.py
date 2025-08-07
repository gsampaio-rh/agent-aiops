"""
Service layer for interacting with Ollama API.
"""

import time
import requests
from typing import Dict, Any, List, Optional, Iterator
import json
import streamlit as st

from config.settings import OLLAMA_BASE_URL, DEFAULT_MODEL


class OllamaService:
    """Service class for Ollama API interactions."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            
            # Ensure default model is available
            if DEFAULT_MODEL not in models and models:
                return models
            elif DEFAULT_MODEL not in models:
                st.error(f"Default model '{DEFAULT_MODEL}' not found. Please ensure it's installed in Ollama.")
                return []
            
            return models
            
        except requests.RequestException as e:
            st.error(f"Failed to connect to Ollama: {e}")
            return []
        except Exception as e:
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
                                    
                        except json.JSONDecodeError:
                            continue
                            
        except requests.RequestException as e:
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
    
    def health_check(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
