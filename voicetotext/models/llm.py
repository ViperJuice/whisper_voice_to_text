"""
LLM enhancement functionality for voice transcription.

Implementation Notes:
- Claude/Anthropic API implementation uses direct REST API calls instead of the anthropic library
- Google API uses models/gemini-2.0-flash-lite model
- We're prioritizing successful API connectivity over enhancement quality for now
"""

import os
import traceback
import time
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

def get_api_key(service_name: str) -> Optional[str]:
    """Get API key for the specified service, checking both .env and environment variables"""
    # First try to get from environment variables
    env_key = os.environ.get(f"{service_name.upper()}_API_KEY")
    if env_key and not env_key.startswith('your_'):
        return env_key
    
    # If not found in environment variables, try .env file
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent.absolute()
        env_path = project_root / '.env'
        
        if env_path.exists():
            # Load .env file
            load_dotenv(env_path, override=True)
            
            # Try to get the key again
            env_key = os.environ.get(f"{service_name.upper()}_API_KEY")
            if env_key and not env_key.startswith('your_'):
                return env_key
    except Exception as e:
        print(f"Error loading .env file: {e}")
    
    return None

def is_openai_available():
    """Check if OpenAI API key is available"""
    api_key = get_api_key("OPENAI")
    return api_key is not None

def is_claude_available():
    """Check if Claude API key is available"""
    api_key = get_api_key("ANTHROPIC")
    return api_key is not None

def is_google_available():
    """Check if Google API key is available"""
    api_key = get_api_key("GOOGLE")
    return api_key is not None

def is_deepseek_available():
    """Check if DeepSeek API key is available"""
    api_key = get_api_key("DEEPSEEK")
    return api_key is not None

def enhance_with_llm(text, llm_model="OLLAMA"):
    """
    Single entry point to enhance text with various LLM providers.
    
    Args:
        text (str): The text to enhance
        llm_model (str): The model to use (OLLAMA, OPENAI, CLAUDE, GOOGLE, DEEPSEEK)
        
    Returns:
        str: The enhanced text or original text if enhancement fails
    """
    print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: ENTER with model parameter: {llm_model}")
    
    if not text or not isinstance(text, str):
        print("⚠️ Invalid input text for enhancement")
        print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: EXIT early - invalid text input")
        return text
        
    try:
        model = llm_model.upper() if llm_model else "OLLAMA"
        print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: After normalizing, model: {model}")
        
        # First check if the specified model's API key is available (except for OLLAMA which is local)
        if model == "OPENAI" and not is_openai_available():
            print("⚠️ OpenAI API key not available, falling back to original text")
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: OpenAI API check failed")
            return text
        elif model == "CLAUDE" and not is_claude_available():
            print("⚠️ Claude API key not available, falling back to original text")
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Claude API check failed")
            return text
        elif model == "GEMINI" and not is_google_available():
            print("⚠️ Google API key not available, falling back to original text")
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Google API check failed")
            return text
        elif model == "DEEPSEEK" and not is_deepseek_available():
            print("⚠️ DeepSeek API key not available, falling back to original text")
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: DeepSeek API check failed")
            return text
        
        # If we get this far, proceed with the model (or OLLAMA as default)
        print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: API key check passed, proceeding with model: {model}")
        
        result = None
        if model == "OPENAI":
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Calling enhance_with_openai")
            result = enhance_with_openai(text)
        elif model == "CLAUDE":
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Calling enhance_with_claude")
            result = enhance_with_claude(text)
        elif model == "GEMINI":
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Calling enhance_with_gemini")
            result = enhance_with_gemini(text)
        elif model == "DEEPSEEK":
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Calling enhance_with_deepseek")
            result = enhance_with_deepseek(text)
        else:  # Default to OLLAMA
            print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Calling enhance_with_ollama")
            result = enhance_with_ollama(text)
            
        print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Enhancement completed, text changed: {result != text}")
        return result
    except Exception as e:
        print(f"⚠️ Error in enhance_with_llm: {e}")
        print(f"🔎 MODEL-TRACE [llm.enhance_with_llm]: Exception occurred: {str(e)}")
        return text

def enhance_with_openai(text):
    """Enhance text using OpenAI model"""
    try:
        # Import the necessary modules
        try:
            from openai import OpenAI
        except ImportError:
            print("⚠️ OpenAI module not installed or not found")
            return text
        
        # Ensure text is properly formatted
        text = text.strip()
        
        # Get API key using the get_api_key function
        api_key = get_api_key("OPENAI")
        if not api_key:
            print("⚠️ OpenAI API key not found or empty in environment variables")
            return text
        
        # Configure the API client
        client = OpenAI(api_key=api_key)
        
        # Create a system message for better instruction
        system_message = "You are a professional transcription editor that improves text from speech recognition."
        
        # Create a prompt that's focused on the specific task
        user_message = f"""Please improve this speech-to-text transcription by fixing grammar, punctuation, capitalization, obvious recognition errors, and make the overall meaning clear and concise:

{text}

Return only the corrected text without any explanations, comments, or additional formatting."""
        
        # Make the API call
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract and return the enhanced text
            enhanced_text = response.choices[0].message.content.strip()
            return enhanced_text
            
        except Exception as api_error:
            print(f"⚠️ Error during OpenAI API call: {api_error}")
            return text
            
    except Exception as e:
        print(f"⚠️ Error in enhance_with_openai: {e}")
        return text

def enhance_with_claude(text):
    """Enhance text using Claude model"""
    try:
        # Ensure text is properly formatted
        text = text.strip()
        
        # Skip enhancement for very short inputs
        if len(text) < 5:
            print("Text too short to enhance, returning original")
            return text
        
        # Get API key using the get_api_key function
        api_key = get_api_key("ANTHROPIC")
        if not api_key:
            print("⚠️ Claude API key not found or empty in environment variables")
            return text
        
        # Use direct API requests with the requests library
        try:
            import requests
            import json
            
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Create a prompt for Claude
            prompt = f"Please improve this speech-to-text transcription by fixing grammar, punctuation, capitalization, obvious recognition errors, and make the overall meaning clear and concise:\n\n{text}"
            
            body = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "temperature": 0.2,
                "system": "You are a professional transcription editor who improves text from speech recognition systems. Only output the improved text.",
                "messages": [{"role": "user", "content": prompt}]
            }
            
            print("Sending request to Claude API...")
            response = requests.post(url, headers=headers, json=body, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                enhanced_text = result["content"][0]["text"].strip()
                return enhanced_text
            else:
                print(f"⚠️ Claude API error: {response.status_code} - {response.text}")
                return text
                
        except Exception as api_error:
            print(f"⚠️ Error during Claude API call: {api_error}")
            return text
            
    except Exception as e:
        print(f"⚠️ Error in enhance_with_claude: {e}")
        return text

def enhance_with_gemini(text):
    """Enhance text using Google model (Gemini)"""
    try:
        # Ensure google-generativeai is installed
        try:
            import google.generativeai as genai
        except ImportError:
            print("⚠️ google-generativeai not installed, returning original text")
            print("Install with: pip install google-generativeai")
            return text
        
        # Ensure the text is properly formatted
        text = text.strip()
        
        # Skip if text is too short
        if len(text) < 5:
            print("Text too short to enhance, returning original")
            return text
        
        # Get API key using the get_api_key function
        api_key = get_api_key("GOOGLE")
        if not api_key:
            print("⚠️ Google API key not found or empty in environment variables")
            return text
        
        # Configure the API 
        try:
            genai.configure(api_key=api_key)
        except Exception as config_err:
            print(f"⚠️ Error configuring Google Generative AI: {config_err}")
            return text
            
        # Define a specific prompt for enhancement
        prompt = f"""As a professional transcription editor, improve this speech-to-text transcription:

Text: {text}

Tasks:
- Fix grammar, punctuation, and capitalization
- Correct obvious recognition errors
- Remove filler words
- Clean up sentence structure
- Make the overall meaning clear and concise

Return only the improved text with no additional comments or formatting."""
        
        # Generate the enhanced text
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            response = model.generate_content(prompt)
            enhanced_text = response.text.strip()
            return enhanced_text
        except Exception as api_error:
            print(f"⚠️ Error during Google API call: {api_error}")
            return text
            
    except Exception as e:
        print(f"⚠️ Error in enhance_with_google: {e}")
        return text

def enhance_with_deepseek(text):
    """Enhance text using DeepSeek API."""
    # Get API key using the get_api_key function
    api_key = get_api_key("DEEPSEEK")
    if not api_key:
        print("⚠️ DeepSeek API key not found in environment variables")
        return text
    
    try:
        # Check if deepseek-chat module is available
        try:
            import requests
            import json
        except ImportError:
            print("⚠️ requests module not installed, returning original text")
            print("Install with: pip install requests")
            return text
        
        # Prepare API call to DeepSeek
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Create a system message for better instruction
        system_message = "You are a professional transcription editor that improves text from speech recognition."
        
        # Create a prompt that's focused on the specific task
        user_message = f"""Please improve this speech-to-text transcription by fixing grammar, punctuation, capitalization, obvious recognition errors, and make the overall meaning clear and concise:

{text}

Return only the corrected text without any explanations, comments, or additional formatting."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        # Make the API call
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                enhanced_text = result["choices"][0]["message"]["content"].strip()
                return enhanced_text
            else:
                print(f"⚠️ DeepSeek API error: {response.status_code} - {response.text}")
                return text
                
        except Exception as api_error:
            print(f"⚠️ Error during DeepSeek API call: {api_error}")
            return text
            
    except Exception as e:
        print(f"⚠️ Error in enhance_with_deepseek: {e}")
        return text

def enhance_with_ollama(text):
    """Enhance text using Ollama model"""
    print("\n" + "="*50)
    print("OLLAMA ENHANCEMENT STARTED")
    print("="*50 + "\n")
    
    # Check text quality
    if not text or len(text.strip()) < 5:
        print("Text too short to enhance with Ollama")
        return text
        
    # Import required modules
    try:
        import requests
        import json
        from requests.exceptions import RequestException
    except ImportError:
        print("requests module not installed")
        return text
    
    # Will be determined dynamically from available models
    model_name = None
    api_endpoint = "http://localhost:11434/api/generate"
    
    print(f"Connecting to Ollama at {api_endpoint}")
    
    # First check if Ollama server is running
    try:
        # Get Ollama version
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✓ Ollama server running: version {version_info.get('version', 'unknown')}")
        else:
            print(f"⚠️ Ollama server returned unexpected status: {response.status_code}")
            return text
    except RequestException as e:
        print(f"⚠️ Could not connect to Ollama server: {e}")
        print("Make sure Ollama is running (run the Ollama application)")
        return text
    
    # Dynamically get available models and select the most appropriate one
    try:
        models_response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if models_response.status_code == 200:
            models_data = models_response.json()
            if "models" in models_data and models_data["models"]:
                available_models = models_data["models"]
                print(f"Found {len(available_models)} available Ollama models:")
                
                # Print available models for debugging
                for m in available_models:
                    model_size = "unknown size"
                    if "details" in m and "parameter_size" in m["details"]:
                        model_size = m["details"]["parameter_size"]
                    print(f" - {m['name']} ({model_size})")
                
                # Preferred model priority: smaller models first as they're faster and use less memory
                # This order ensures we'll get the best model that fits in memory
                model_preferences = [
                    "llama3.1", "llama3.2", "llama3.3",  # Llama3 size variants (smallest to largest)
                    "tinyllama", "llama2", "gemma",      # Alternative models
                    "mistral", "mixtral", "phi"          # More alternatives
                ]
                
                # First try exact matches with :latest tag
                for prefix in model_preferences:
                    exact_model = f"{prefix}:latest"
                    if any(m["name"] == exact_model for m in available_models):
                        model_name = exact_model
                        print(f"Selected exact model: {model_name}")
                        break
                
                # If no exact match, try prefix matches (any version/variant)
                if not model_name:
                    for prefix in model_preferences:
                        matching_models = [m for m in available_models if m["name"].startswith(prefix)]
                        if matching_models:
                            # Sort by name to get the most basic/stable version
                            matching_models.sort(key=lambda x: x["name"])
                            model_name = matching_models[0]["name"]
                            print(f"Selected prefix-matched model: {model_name}")
                            break
                
                # Last resort: use the first available model
                if not model_name and available_models:
                    model_name = available_models[0]["name"]
                    print(f"No preferred models found. Using first available: {model_name}")
            else:
                print("No models found in Ollama response")
        else:
            print(f"Failed to get models: {models_response.status_code}")
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
    
    # If we still don't have a model, use a safe default
    if not model_name:
        model_name = "llama3.2:latest"  # Fallback to a reasonable default
        print(f"Using fallback model: {model_name}")
    
    # Create a prompt for the enhancement
    prompt = f"""Please improve this speech-to-text transcription by fixing grammar, punctuation, and other errors:

Original text: {text.strip()}

Return only the improved text without any explanations or comments."""

    # Set up the API request with the selected model
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending request to Ollama API with model: {model_name}")
    
    # Try to get a response from the API
    try:
        # Send the API request
        response = requests.post(api_endpoint, json=payload, timeout=30)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: Ollama API returned status code {response.status_code}")
            
            # If model not found (404) or other error, try with a different model
            if response.status_code == 404 or "no such model" in response.text.lower():
                print("Model not found, trying with generic 'llama2'")
                payload["model"] = "llama2"
                
                try:
                    response = requests.post(api_endpoint, json=payload, timeout=30)
                    if response.status_code != 200:
                        print(f"Second attempt failed: {response.status_code}")
                        return text
                except Exception as retry_err:
                    print(f"Error in retry attempt: {retry_err}")
                    return text
            else:
                print(f"Response: {response.text}")
                return text
        
        # Parse the response
        result = response.json()
        if "response" not in result:
            print("Error: Unexpected response format")
            print(f"Response: {result}")
            return text
            
        enhanced_text = result["response"].strip()
        
        # If the result is empty or identical, use the original
        if not enhanced_text or enhanced_text.lower() == text.lower():
            print("Ollama returned no enhancements, using original text")
            return text
            
        print("\n" + "="*50)
        print("OLLAMA ENHANCEMENT SUCCESSFUL")
        print("="*50 + "\n")
        
        return enhanced_text
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        import traceback
        traceback.print_exc()
        return text 