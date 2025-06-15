**🪨 STONE**

Self-Taught Ollama Node Engine — a compact, memory-persistent, function-calling AI assistant kernel.
Real-time streaming. RAG memory. No BS.

🚀 Features

🌐 WebSocket streaming (fast token delivery)

🧠 Context + RAG Memory (SQLite-backed)

🔧 Function calling with auto-response handling

🤖 Ollama model support (like llama3, mistral, etc)

💾 Session persistence

🌙 Background maintenance tasks

💬 Minimal frontend (HTML/JS only — no React circus)

🧩 All in one file for maximum flow and minimal complexity

**📦 Dependencies**
You’ll need:
Python 3.10+
Ollama running locally (ollama serve)
Node not required (despite "Node Engine" — the front-end is pure JS)

The following Python libraries:
pip install flask flask-socketio requests eventlet
⚠️ SQLite comes built-in with Python. No setup required.

**🛠️ Setup Instructions**
Start Ollama
If you haven’t yet:

ollama serve
ollama run llama3
Clone this repo

git clone https://github.com/your-name/stone-ai.git
cd stone-ai
Run the STONE server

python stone.py
Open the UI
Visit: http://localhost:11434
(Or whatever port you’ve set in the script)

**💾 Memory Storage**
context.db: stores message history by session

rag_memory: stores important/referenced information using a simple keyword-matching RAG system

**🧠 Function Calling**
STONE can detect and run predefined functions dynamically via chat messages.
You define tools like:

TOOLS = {
    "get_weather": {
        "function": get_weather_function,
        "description": "Fetches current weather for a location."
    },
    ...
}


**STONE will:**
Detect calls like run get_weather("Tokyo")
Execute them
Send results back to the LLM in real-time

**🔁 Background Tasks**
Runs a cleanup task every hour to:
Trim message history per session (last 100 only)
Trim RAG memory (last 1000 entries)

**⚙️ Customization**
You can:
Add new function tools in the TOOLS dictionary
Modify the context handling (e.g., change from SQLite to Postgres)
Replace the memory strategy with vector search if needed
Plug in other LLMs via API by replacing the OLLAMA_BASE_URL logic

📁 Project Structure

stone.py           # The main app (Flask + WebSocket + RAG)
templates/
└── index.html      # Minimal front-end (HTML + JS)
context.db         # SQLite memory store (auto-created)

**❗Troubleshooting**

🔌 Ollama not responding?
Make sure ollama serve is running
Try hitting http://localhost:11434/api/tags in your browser

⌛ Timeouts or no response?
Your model may be cold or too large. Try smaller ones like mistral or gemma.

💬 No models listed?
Run ollama run llama3 to make sure a model is available

🤘 Philosophy
“One file. Real-time. Human-centered.”
STONE isn’t a framework. It’s a forge.
Build your AI assistant like you mean it.

 🧠 Coming Soon (Ideas / TODOs)
 Web-based memory inspector
 Built-in function calling templates
 Front-end enhancements (Markdown, avatars)
 Local vector store integration
 Offline mode (LLM + RAG fully local)

📜 License
MIT.
Use it. Fork it. Break it. Just don’t make it boring.
