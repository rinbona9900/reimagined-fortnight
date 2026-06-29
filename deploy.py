#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import re
import urllib.request
import urllib.error

# --- Configuration ---
CLOUD_MODELS = [
    "minimax-m3:cloud",
    "kimi-k2.7-code:cloud",
    "glm-5.2:cloud",
    "gemma4:12b", # Using a specific size for stability
    "gemma4:31b",
    "qwen3.6:27b",
    "nemotron-3-ultra:cloud"
]

LOCAL_MODELS = [
    "lfm2.5:8b",
    "north-mini-code-1.0:30b",
    "ornith:9b",
    "phi4:14b"
]

OLLAMA_HOST = "127.0.0.1:11434"
LITELLM_PORT = 4000
NUM_CTX = 32768

def log(msg):
    print(f"[DEPLOY] {msg}", flush=True)

def run_cmd(cmd, check=True):
    log(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        log(f"ERROR: {result.stderr}")
        # Don't exit immediately for some commands like apt-get updates which might have warnings
        if "apt-get" not in cmd:
            pass # Continue to next step for resilience, or handle specifically
    return result

def wait_healthy(url, retries=30):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def pull_model(model_tag):
    log(f"Pulling {model_tag}...")
    # Ollama pull can take a while, set longer timeout implicitly by not checking return immediately
    run_cmd(f"ollama pull {model_tag}", check=False) 

def verify_tools(model_tag):
    log(f"Verifying tool support for {model_tag}...")
    probe = {
        "model": model_tag,
        "messages": [{"role": "user", "content": "Use the get_weather tool for Paris."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            }
        }],
        "stream": False
    }
    try:
        data = json.dumps(probe).encode()
        req = urllib.request.Request(f"http://{OLLAMA_HOST}/api/chat", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read().decode())
            if res.get("message", {}).get("tool_calls"):
                log(f"✓ {model_tag} supports tools.")
                return True
            else:
                log(f"✗ {model_tag} did not return tool_calls.")
                return False
    except Exception as e:
        log(f"✗ Error verifying {model_tag}: {e}")
        return False

def main():
    log("Starting deployment script...")

    # 1. Install System Deps & Ollama
    log("Installing system dependencies...")
    run_cmd("sudo apt-get update -y")
    run_cmd("sudo apt-get install -y curl jq net-tools")

    log("Installing Ollama...")
    run_cmd("curl -fsSL https://ollama.com/install.sh | sh")
    
    # Set Env Vars
    os.environ["OLLAMA_HOST"] = f"http://{OLLAMA_HOST}"
    os.environ["OLLAMA_NUM_CTX"] = str(NUM_CTX)
    
    # Start Ollama
    log("Starting Ollama service...")
    run_cmd("sudo systemctl start ollama")
    time.sleep(5)
    
    if not wait_healthy(f"http://{OLLAMA_HOST}/api/tags"):
        log("FATAL: Ollama failed to start.")
        sys.exit(1)
    log("Ollama is healthy.")

    # 2. Pull Cloud Models (Non-blocking, fire and forget mostly, but ensure they are registered)
    log("Registering cloud models...")
    for model in CLOUD_MODELS:
        pull_model(model)
        time.sleep(1) # Small delay between pulls

    # 3. Pull & Verify Local Models (Find best working one)
    best_local = None
    log("Attempting to setup local fallback models...")
    for model in LOCAL_MODELS:
        pull_model(model)
        if verify_tools(model):
            best_local = model
            break
        else:
            log(f"Model {model} failed verification, trying next...")
            # Optional: Unpull to save space? No, keep for user choice if they want to try manually
    
    if not best_local:
        log("WARNING: No local model passed tool verification. Falling back to lfm2.5 anyway.")
        best_local = "lfm2.5:8b" # Default fallback even if verify failed
    
    log(f"Selected best local model: {best_local}")

    # 4. Install LiteLLM
    log("Installing LiteLLM...")
    run_cmd("pip install litellm[proxy]")

    # 5. Generate Config
    log("Generating LiteLLM configuration...")
    
    model_list = []
    
    # Add Cloud Models
    for m in CLOUD_MODELS:
        model_list.append({
            "model_name": m.split(":")[0], # Simple name
            "litellm_params": {
                "model": f"ollama_chat/{m}",
                "api_base": f"http://{OLLAMA_HOST}",
                "num_ctx": NUM_CTX
            }
        })
    
    # Add Best Local Model
    model_list.append({
        "model_name": best_local.split(":")[0],
        "litellm_params": {
            "model": f"ollama_chat/{best_local}",
            "api_base": f"http://{OLLAMA_HOST}",
            "num_ctx": NUM_CTX
        }
    })

    config = {
        "model_list": model_list,
        "general_settings": {
            "master_key": "sk-opencode-local-key"
        },
        "litellm_settings": {
            "drop_params": True,
            "request_timeout": 600
        }
    }

    with open("litellm_config.yaml", "w") as f:
        import yaml
        # Fallback if pyyaml not installed, use simple string formatting
        try:
            yaml.dump(config, f)
        except ImportError:
            # Simple manual dump if yaml lib missing (unlikely in litellm env but safe)
            f.write("model_list:\n")
            for m in model_list:
                f.write(f"  - model_name: {m['model_name']}\n")
                f.write("    litellm_params:\n")
                for k, v in m['litellm_params'].items():
                    f.write(f"      {k}: {v}\n")
            f.write("general_settings:\n  master_key: sk-opencode-local-key\n")
            f.write("litellm_settings:\n  drop_params: true\n  request_timeout: 600\n")

    log("LiteLLM config written.")

    # 6. Start LiteLLM
    log("Starting LiteLLM proxy...")
    # Run in background
    with open("litellm.log", "w") as f:
        subprocess.Popen(["litellm", "--config", "litellm_config.yaml", "--port", str(LITELLM_PORT)], stdout=f, stderr=subprocess.STDOUT)
    
    time.sleep(10)
    if not wait_healthy(f"http://127.0.0.1:{LITELLM_PORT}/health/liveliness"):
        log("FATAL: LiteLLM failed to start.")
        sys.exit(1)
    log("LiteLLM is healthy.")

    # 7. Start Cloudflared
    log("Downloading and starting Cloudflared...")
    run_cmd("curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
    run_cmd("chmod +x cloudflared")
    
    with open("cloudflare.log", "w") as f:
        subprocess.Popen(["./cloudflared", "tunnel", "--url", f"http://127.0.0.1:{LITELLM_PORT}"], stdout=f, stderr=subprocess.STDOUT)
    
    time.sleep(10)
    
    tunnel_url = None
    with open("cloudflare.log", "r") as f:
        content = f.read()
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', content)
        if match:
            tunnel_url = match.group(0)
    
    if not tunnel_url:
        log("FATAL: Could not detect Cloudflare tunnel URL.")
        sys.exit(1)
    
    log(f"Tunnel URL: {tunnel_url}")

    # 8. Generate OpenCode Config
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "gh-actions-backend": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "GH-Actions-Ollama-Proxy",
                "options": {
                    "baseURL": f"{tunnel_url}/v1",
                    "apiKey": "sk-opencode-local-key",
                    "timeout": 600000
                },
                "models": {}
            }
        },
        "model": "gh-actions-backend/" + best_local.split(":")[0]
    }

    # List all available models in config
    for m in model_list:
        name = m['model_name']
        opencode_config["provider"]["gh-actions-backend"]["models"][name] = {
            "id": name,
            "name": name,
            "tool_call": True,
            "temperature": True,
            "limit": {"context": NUM_CTX, "output": 8192}
        }

    with open("opencode_config.json", "w") as f:
        json.dump(opencode_config, f, indent=2)
    
    log("Deployment Complete!")
    log(f"Base URL: {tunnel_url}/v1")
    log(f"API Key: sk-opencode-local-key")
    log(f"Available Models: {[m['model_name'] for m in model_list]}")
    log("Config saved to opencode_config.json")

if __name__ == "__main__":
    main()
