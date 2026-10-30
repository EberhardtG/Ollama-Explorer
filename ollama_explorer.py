import requests
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Use whatever model you pulled

#---intiate deployment of ollama---
def check_ollama():
    """Verify Ollama is running."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama is running. Available models: {models}")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return False


def generate(prompt, system="You are a helpful assistant.", temperature=None):
    """
    Send a prompt to Ollama's /api/chat endpoint, measure response time,
    and return the model's response text.
    """

    # Start timing
    start = time.time()

    # Build request payload
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    # Add temperature only if provided
    if temperature is not None:
        payload["temperature"] = temperature

    # Send request
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)

    # Measure elapsed time
    elapsed = time.time() - start

    # Basic error handling
    if response.status_code != 200:
        print(f"[Error {response.status_code}] {response.text}")
        return None

    # Extract model response
    data = response.json()
    try:
        text = data["message"]["content"]
    except KeyError:
        print("Unexpected response format:", data)
        return None

    print(f"[Generated in {elapsed:.2f}s]")
    return text


# --- Run the demo ---
if not check_ollama():
    exit(1)

print("\n" + "=" * 60)
print("TEST 1: Same question, different system prompts")
print("=" * 60)

print("\n--- Experiment 1: No system prompt ---")
resp_default = generate("What is an API?", system=None)
print(resp_default)

print("\n--- Experiment 1: Explain like I'm 5 ---")
resp_eLI5 = generate("What is an API?", system="Explain like I’m 5 years old.")
print(resp_eLI5)

print("\n--- Experiment 1: Senior software architect ---")
resp_arch = generate("What is an API?", system="You are a senior software architect. Be technical and precise.")
print(resp_arch)

# Observations:
# - No system prompt: neutral, generic definition
# - Explain like I'm 5: simple, analogy-based explanation
# - Senior architect: technical, precise, uses jargon



print("\n" + "=" * 60)
print("TEST 2: With system prompt (RAG-style)")
print("=" * 60)

context = ("An API is a set of rules that allows one software system to communicate with another. It defines how requests should be structured and how responses are returned. APIs make it possible for applications to share data and functionality without needing to understand each other's internal code.")

context = (
    "An API is a set of rules that allows one software system to communicate "
    "with another. It defines how requests should be structured and how "
    "responses are returned. APIs make it possible for applications to share "
    "data and functionality without needing to understand each other's internal code."
)

print("\n--- Experiment 2: Answerable question ---")
resp1 = generate(
    "How do APIs help software communicate?",
    system=f"Use ONLY this context:\n{context}\nIf the answer is not in the context, say 'I don't know.'"
)
print(resp1)

print("\n--- Experiment 2: Unanswerable question ---")
resp2 = generate(
    "Who invented APIs?",
    system=f"Use ONLY this context:\n{context}\nIf the answer is not in the context, say 'I don't know.'"
)
print(resp2)



print("\n" + "=" * 60)
print("TEST 3:  Response timing")
print("=" * 60)


print("\n--- Experiment 3: Short prompt ---")
resp_short = generate("What is gravity?")
print(resp_short)


print("\n--- Experiment 3: Medium prompt ---")
resp_medium = generate(
    "Can you explain how gravity affects objects that are moving through space in a simple and clear way?"
)
print(resp_medium)

print("\n--- Experiment 3: Long prompt ---")
resp_long = generate(
    "Provide a detailed explanation of how gravitational forces influence the motion of planets, moons, and other celestial bodies, "
    "including how these forces interact over long distances and how they maintain orbital stability."
)
print(resp_long)

print("\n" + "=" * 60)
print("TEST 4:  Temperature")
print("=" * 60)

print("\n--- Low temperature (0.1) ---")
low_temp = generate("Write a poem about gravity.", temperature=0.1)
print(low_temp)


print("\n---High temperature (1.0) ---")
high_temp = generate("Write a poem about gravity.", temperature=1.0)
print(high_temp)

# Observations:
# - Temperature 0.1 produced a simple, predictable poem with clear structure.
# - Temperature 1.0 produced a more creative and varied poem with unusual imagery.


