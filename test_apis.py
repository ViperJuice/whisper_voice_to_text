import os
import requests
import json

def test_openai():
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.status_code == 200
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return False

def test_anthropic():
    try:
        headers = {
            "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Anthropic Error: {str(e)}")
        return False

def test_google():
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
        )
        if response.status_code == 200:
            return True
        else:
            print(f"Google API Response: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Google Error: {str(e)}")
        return False

def test_deepseek():
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Deepseek Error: {str(e)}")
        return False

def main():
    print("Testing API Access...")
    print("-" * 50)
    
    print("OpenAI API:", "✓ Success" if test_openai() else "✗ Failed")
    print("Anthropic API:", "✓ Success" if test_anthropic() else "✗ Failed")
    print("Google API:", "✓ Success" if test_google() else "✗ Failed")
    print("Deepseek API:", "✓ Success" if test_deepseek() else "✗ Failed")

if __name__ == "__main__":
    main() 