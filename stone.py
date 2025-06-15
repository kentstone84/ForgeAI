# STONE AI - Enhanced Ollama Server with RAG Memory and Function Calling
# STONE Enhanced Server with Function Calling, RAG Memory, and Context Storage


from flask import Flask, render_template_string, request, jsonify, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests
import json
import os
import uuid
import sqlite3
from datetime import datetime
import subprocess
import re
import threading
import time
from collections import defaultdict
import hashlib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stone-secret-key-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
PORT = 5000
CONTEXT_DB = "stone_context.db"

# Initialize SQLite database for context storage and RAG memory
def init_db():
    conn = sqlite3.connect(CONTEXT_DB)
    c = conn.cursor()
    
    # Context table
    c.execute('''CREATE TABLE IF NOT EXISTS context 
                (session_id TEXT, timestamp TEXT, message TEXT, role TEXT)''')
    
    # RAG Memory table for semantic storage
    c.execute('''CREATE TABLE IF NOT EXISTS rag_memory 
                (id TEXT PRIMARY KEY, session_id TEXT, content TEXT, 
                 keywords TEXT, timestamp TEXT, importance INTEGER DEFAULT 1)''')
    
    # Knowledge base for persistent facts
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_base 
                (topic TEXT, content TEXT, source TEXT, timestamp TEXT,
                 PRIMARY KEY (topic, source))''')
    
    conn.commit()
    conn.close()

init_db()

# RAG Memory System
class RAGMemory:
    def __init__(self):
        self.keyword_index = defaultdict(list)
        self.load_memory_index()
    
    def extract_keywords(self, text):
        """Extract keywords from text using simple regex"""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'with', 'for', 'be', 'have', 'this', 'that', 'will', 'you', 'they', 'of', 'it', 'in', 'or', 'an', 'what', 'when', 'where', 'how', 'why', 'who'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [word for word in words if word not in common_words]
    
    def store_memory(self, session_id, content, importance=1):
        """Store content in RAG memory with keyword indexing"""
        memory_id = hashlib.md5(f"{session_id}_{content}_{datetime.now()}".encode()).hexdigest()
        keywords = self.extract_keywords(content)
        
        conn = sqlite3.connect(CONTEXT_DB)
        c = conn.cursor()
        
        c.execute("""INSERT OR REPLACE INTO rag_memory 
                    (id, session_id, content, keywords, timestamp, importance) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (memory_id, session_id, content, ' '.join(keywords), 
                  datetime.now().isoformat(), importance))
        
        conn.commit()
        conn.close()
        
        # Update in-memory index
        for keyword in keywords:
            self.keyword_index[keyword].append({
                'id': memory_id,
                'content': content,
                'importance': importance,
                'timestamp': datetime.now().isoformat()
            })
    
    def search_memory(self, query, session_id=None, limit=5):
        """Search memory using keyword matching"""
        query_keywords = self.extract_keywords(query)
        
        conn = sqlite3.connect(CONTEXT_DB)
        c = conn.cursor()
        
        if session_id:
            c.execute("""SELECT content, importance, timestamp FROM rag_memory 
                        WHERE session_id = ? AND keywords LIKE ?
                        ORDER BY importance DESC, timestamp DESC LIMIT ?""",
                     (session_id, f"%{' '.join(query_keywords)}%", limit))
        else:
            c.execute("""SELECT content, importance, timestamp FROM rag_memory 
                        WHERE keywords LIKE ?
                        ORDER BY importance DESC, timestamp DESC LIMIT ?""",
                     (f"%{' '.join(query_keywords)}%", limit))
        
        results = c.fetchall()
        conn.close()
        
        return [{'content': r[0], 'importance': r[1], 'timestamp': r[2]} for r in results]
    
    def store_knowledge(self, topic, content, source="user"):
        """Store persistent knowledge"""
        conn = sqlite3.connect(CONTEXT_DB)
        c = conn.cursor()
        
        c.execute("""INSERT OR REPLACE INTO knowledge_base 
                    (topic, content, source, timestamp) VALUES (?, ?, ?, ?)""",
                 (topic, content, source, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_knowledge(self, topic):
        """Retrieve knowledge about a topic"""
        conn = sqlite3.connect(CONTEXT_DB)
        c = conn.cursor()
        
        c.execute("SELECT content, source, timestamp FROM knowledge_base WHERE topic LIKE ?",
                 (f"%{topic}%",))
        
        results = c.fetchall()
        conn.close()
        
        return [{'content': r[0], 'source': r[1], 'timestamp': r[2]} for r in results]
    
    def load_memory_index(self):
        """Load memory index on startup"""
        try:
            conn = sqlite3.connect(CONTEXT_DB)
            c = conn.cursor()
            c.execute("SELECT id, content, keywords, importance, timestamp FROM rag_memory")
            
            for row in c.fetchall():
                memory_id, content, keywords_str, importance, timestamp = row
                keywords = keywords_str.split() if keywords_str else []
                
                for keyword in keywords:
                    self.keyword_index[keyword].append({
                        'id': memory_id,
                        'content': content,
                        'importance': importance,
                        'timestamp': timestamp
                    })
            
            conn.close()
        except Exception as e:
            print(f"Warning: Could not load memory index: {e}")

# Initialize RAG Memory
rag_memory = RAGMemory()

# Function calling tools
TOOLS = {
    "weather": {
        "function": lambda city: get_weather(city),
        "description": "Get current weather for a specified city",
        "pattern": r"weather\s+(?:in\s+|for\s+)?(.+)",
        "example": "weather in New York"
    },
    "python": {
        "function": lambda code: run_python_code(code),
        "description": "Execute Python code and return output",
        "pattern": r"run\s+python\s+(.+)",
        "example": "run python print('hello world')"
    },
    "calculate": {
        "function": lambda expr: calculate_expression(expr),
        "description": "Calculate mathematical expressions",
        "pattern": r"calculate\s+(.+)",
        "example": "calculate 2 + 2 * 3"
    },
    "remember": {
        "function": lambda info: remember_info(info),
        "description": "Store information in memory",
        "pattern": r"remember\s+(.+)",
        "example": "remember I like pizza"
    },
    "recall": {
        "function": lambda query: recall_info(query),
        "description": "Search stored memories",
        "pattern": r"(?:recall|what do you know about)\s+(.+)",
        "example": "recall pizza"
    }
}

def get_weather(city):
    """Get weather information for a city"""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=%C+%t+%h+%w", timeout=5)
        if response.status_code == 200:
            return f"Weather in {city}: {response.text.strip()}"
        else:
            return f"Could not get weather for {city}"
    except Exception as e:
        return f"Weather service error: {str(e)}"

def run_python_code(code):
    """Execute Python code safely"""
    try:
        # Basic security: restrict dangerous imports and operations
        dangerous_patterns = ['import os', 'import sys', 'import subprocess', '__import__', 'eval', 'exec']
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                return "Security Error: Dangerous operation not allowed"
        
        result = subprocess.run(
            ["python", "-c", code], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            return result.stdout if result.stdout else "Code executed successfully (no output)"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Execution error: {str(e)}"

def calculate_expression(expr):
    """Safely calculate mathematical expressions"""
    try:
        # Remove any dangerous characters/functions
        safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expr)
        result = eval(safe_expr)
        return f"{expr} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

def remember_info(info):
    """Store information in RAG memory"""
    # This will be set in the context of the session
    session_id = getattr(remember_info, 'current_session', 'default')
    rag_memory.store_memory(session_id, info, importance=2)
    return f"Remembered: {info}"

def recall_info(query):
    """Search stored memories"""
    session_id = getattr(recall_info, 'current_session', 'default')
    memories = rag_memory.search_memory(query, session_id)
    
    if memories:
        results = []
        for mem in memories:
            results.append(f"• {mem['content']}")
        return f"I recall:\n" + "\n".join(results)
    else:
        return f"I don't have any memories about '{query}'"

def detect_function_call(message):
    """Detect and extract function calls from user message"""
    message_lower = message.lower().strip()
    
    for tool_name, tool in TOOLS.items():
        match = re.search(tool['pattern'], message_lower)
        if match:
            parameter = match.group(1).strip()
            return tool_name, parameter
    
    return None, None

# HTML Template with fixed WebSocket implementation
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STONE AI</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #00d4ff;
            min-height: 100vh;
            overflow: hidden;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: relative;
        }

        .header {
            padding: 20px;
            text-align: center;
            background: rgba(0, 212, 255, 0.1);
            border-bottom: 2px solid rgba(0, 212, 255, 0.3);
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 0.2em;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
            to { text-shadow: 0 0 30px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.3); }
        }

        .status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
            animation: pulse 2s infinite;
        }

        .status-indicator.connected {
            background: #00ff44;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow: hidden;
        }

        .model-selector {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .model-selector select {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #00d4ff;
            padding: 10px;
            border-radius: 8px;
            font-size: 14px;
        }

        .chat-container {
            flex: 1;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            backdrop-filter: blur(5px);
        }

        .messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 20px;
            position: relative;
            animation: slideIn 0.3s ease-out;
            white-space: pre-wrap;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
        }

        .message.assistant {
            align-self: flex-start;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }

        .message.system {
            align-self: center;
            background: rgba(255, 196, 0, 0.1);
            border: 1px solid rgba(255, 196, 0, 0.3);
            color: #ffc400;
            font-style: italic;
            max-width: 60%;
        }

        .message.function {
            align-self: center;
            background: rgba(0, 255, 100, 0.1);
            border: 1px solid rgba(0, 255, 100, 0.3);
            color: #00ff64;
            font-family: 'Courier New', monospace;
            max-width: 90%;
        }

        .input-container {
            padding: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid rgba(0, 212, 255, 0.2);
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        .message-input {
            width: 100%;
            padding: 15px 60px 15px 20px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 25px;
            color: #00d4ff;
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
        }

        .message-input:focus {
            border-color: #00d4ff;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }

        .send-btn {
            position: absolute;
            right: 5px;
            top: 50%;
            transform: translateY(-50%);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            font-size: 18px;
        }

        .send-btn:hover {
            transform: translateY(-50%) scale(1.1);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 5px;
            color: #00d4ff;
            font-style: italic;
            padding: 10px 20px;
        }

        .typing-indicator.show {
            display: flex;
        }

        .typing-dots {
            display: flex;
            gap: 3px;
        }

        .typing-dots span {
            width: 6px;
            height: 6px;
            background: #00d4ff;
            border-radius: 50%;
            animation: typingBounce 1.4s infinite ease-in-out;
        }

        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typingBounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .background-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }

        .circuit-line {
            position: absolute;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
            animation: circuitFlow 4s linear infinite;
        }

        @keyframes circuitFlow {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100vw); }
        }

        .messages::-webkit-scrollbar {
            width: 6px;
        }

        .messages::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }

        .messages::-webkit-scrollbar-thumb {
            background: rgba(0, 212, 255, 0.3);
            border-radius: 3px;
        }

        .messages::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 212, 255, 0.5);
        }

        .help-panel {
            background: rgba(0, 212, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            font-size: 14px;
        }

        .help-panel h3 {
            margin-bottom: 10px;
            color: #00d4ff;
        }

        .help-panel ul {
            list-style: none;
            padding-left: 0;
        }

        .help-panel li {
            margin: 5px 0;
            padding: 5px 10px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="background-animation"></div>

    <div class="container">
        <div class="header">
            <h1>STONE AI</h1>
            <div class="status">
                <div class="status-indicator" id="statusIndicator"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </div>

        <div class="main">
            <div class="model-selector">
                <label>Model:</label>
                <select id="modelSelect">
                    <option value="">Loading models...</option>
                </select>
                <button onclick="loadModels()">🔄</button>
            </div>

            <div class="help-panel">
                <h3>Available Commands:</h3>
                <ul>
                    <li>weather in [city] - Get weather information</li>
                    <li>run python [code] - Execute Python code</li>
                    <li>calculate [expression] - Mathematical calculations</li>
                    <li>remember [info] - Store information in memory</li>
                    <li>recall [query] - Search stored memories</li>
                </ul>
            </div>

            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="message system">
                        STONE Enhanced Server online. Ready for commands with function calling, RAG memory, and context storage.
                    </div>
                </div>

                <div class="typing-indicator" id="typingIndicator">
                    <span>STONE is thinking</span>
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>

                <div class="input-container">
                    <div class="input-wrapper">
                        <input type="text" class="message-input" id="messageInput" 
                               placeholder="Ask STONE anything... (try 'weather in London' or 'remember I like coffee')" 
                               onkeypress="handleKeyPress(event)">
                        <button class="send-btn" id="sendBtn" onclick="sendMessage()">▶</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentModel = '';
        let isGenerating = false;
        let sessionId = localStorage.getItem('stoneSessionId') || generateUUID();
        localStorage.setItem('stoneSessionId', sessionId);
        let socket = null;

        function generateUUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        document.addEventListener('DOMContentLoaded', function() {
            initializeSocket();
            loadModels();
            createBackgroundAnimation();
            loadContext();
        });

        function initializeSocket() {
            socket = io();
            
            socket.on('connect', function() {
                document.getElementById('statusIndicator').classList.add('connected');
                document.getElementById('statusText').textContent = 'Connected to Ollama';
            });

            socket.on('disconnect', function() {
                document.getElementById('statusIndicator').classList.remove('connected');
                document.getElementById('statusText').textContent = 'Disconnected';
            });

            socket.on('response_token', function(data) {
                updateMessage(data.token);
            });

            socket.on('response_complete', function(data) {
                hideTypingIndicator();
                isGenerating = false;
                document.getElementById('sendBtn').disabled = false;
                saveContext('assistant', data.full_response);
            });

            socket.on('function_result', function(data) {
                addMessage('function', `🔧 ${data.function}(${data.parameter})\\n\\n${data.result}`);
            });

            socket.on('error', function(data) {
                hideTypingIndicator();
                addMessage('system', `Error: ${data.message}`);
                isGenerating = false;
                document.getElementById('sendBtn').disabled = false;
            });
        }

        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                
                const modelSelect = document.getElementById('modelSelect');
                modelSelect.innerHTML = '';
                
                if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = model.name;
                        modelSelect.appendChild(option);
                    });
                    
                    currentModel = data.models[0].name;
                    modelSelect.value = currentModel;
                } else {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No models found';
                    modelSelect.appendChild(option);
                }
                
                modelSelect.onchange = function() {
                    currentModel = this.value;
                };
            } catch (error) {
                console.error('Error loading models:', error);
                addMessage('system', 'Error loading available models');
            }
        }

        async function loadContext() {
            try {
                const response = await fetch(`/api/context?session_id=${sessionId}`);
                const data = await response.json();
                data.messages.forEach(msg => {
                    addMessage(msg.role, msg.message);
                });
            } catch (error) {
                console.error('Error loading context:', error);
            }
        }

        function addMessage(type, content) {
            const messagesContainer = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.textContent = content;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageDiv;
        }

        let currentAssistantMessage = null;

        function updateMessage(content) {
            if (!currentAssistantMessage) {
                currentAssistantMessage = addMessage('assistant', '');
            }
            currentAssistantMessage.textContent += content;
            document.getElementById('messages').scrollTop = 
                document.getElementById('messages').scrollHeight;
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').classList.add('show');
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').classList.remove('show');
        }

        function saveContext(role, message) {
            fetch('/api/save_context', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    role: role,
                    message: message
                })
            });
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || !currentModel || isGenerating) return;
            
            addMessage('user', message);
            saveContext('user', message);
            input.value = '';
            
            showTypingIndicator();
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;
            currentAssistantMessage = null;
            
            socket.emit('send_message', {
                model: currentModel,
                message: message,
                session_id: sessionId
            });
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function createBackgroundAnimation() {
            const container = document.querySelector('.background-animation');
            
            for (let i = 0; i < 5; i++) {
                const line = document.createElement('div');
                line.className = 'circuit-line';
                line.style.top = Math.random() * 100 + '%';
                line.style.height = '1px';
                line.style.animationDelay = Math.random() * 4 + 's';
                line.style.animationDuration = (4 + Math.random() * 2) + 's';
                container.appendChild(line);
            }
        }
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    """Serve the main STONE interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/models')
def get_models():
    """Get available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to fetch models", "models": []}, 500
    except Exception as e:
        return {"error": str(e), "models": []}, 500

@app.route('/api/context')
def get_context():
    """Retrieve conversation context for a session"""
    session_id = request.args.get('session_id')
    conn = sqlite3.connect(CONTEXT_DB)
    c = conn.cursor()
    
    c.execute("SELECT message, role FROM context WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10", 
             (session_id,))
    messages = []
    for row in reversed(c.fetchall()):  # Reverse to get chronological order
        messages.append({"message": row[0], "role": row[1]})
    
    conn.close()
    return {"messages": messages}

@app.route('/api/save_context', methods=['POST'])
def save_context():
    """Save conversation context"""
    data = request.get_json()
    session_id = data.get('session_id')
    role = data.get('role')
    message = data.get('message')
    
    conn = sqlite3.connect(CONTEXT_DB)
    c = conn.cursor()
    
    c.execute("INSERT INTO context (session_id, timestamp, message, role) VALUES (?, ?, ?, ?)",
             (session_id, datetime.now().isoformat(), message, role))
    
    conn.commit()
    conn.close()
    
    # Store in RAG memory if it's important
    if role == 'user' and any(keyword in message.lower() for keyword in ['remember', 'important', 'note']):
        rag_memory.store_memory(session_id, message, importance=3)
    
    return {"status": "saved"}

@app.route('/api/memory/search')
def search_memory():
    """Search RAG memory"""
    query = request.args.get('query', '')
    session_id = request.args.get('session_id')
    
    memories = rag_memory.search_memory(query, session_id)
    return {"memories": memories}

@app.route('/api/knowledge', methods=['GET', 'POST'])
def knowledge_endpoint():
    """Get or store knowledge"""
    if request.method == 'GET':
        topic = request.args.get('topic', '')
        knowledge = rag_memory.get_knowledge(topic)
        return {"knowledge": knowledge}
    
    elif request.method == 'POST':
        data = request.get_json()
        topic = data.get('topic')
        content = data.get('content')
        source = data.get('source', 'api')
        
        rag_memory.store_knowledge(topic, content, source)
        return {"status": "stored"}

# WebSocket handlers
@socketio.on('send_message')
def handle_message(data):
    """Handle incoming messages with streaming response"""
    model = data.get('model')
    message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    if not model or not message:
        emit('error', {'message': 'Model and message are required'})
        return
    
    try:
        # Set session context for function calls
        remember_info.current_session = session_id
        recall_info.current_session = session_id
        
        # Check for function calls first
        function_name, parameter = detect_function_call(message)
        if function_name and parameter:
            try:
                result = TOOLS[function_name]['function'](parameter)
                emit('function_result', {
                    'function': function_name,
                    'parameter': parameter,
                    'result': result
                })
                
                # Continue with AI response about the function result
                message = f"I executed {function_name} with parameter '{parameter}' and got: {result}. Please provide a natural response about this result."
            except Exception as e:
                emit('function_result', {
                    'function': function_name,
                    'parameter': parameter,
                    'result': f"Error: {str(e)}"
                })
                return
        
        # Get relevant memories for context
        memories = rag_memory.search_memory(message, session_id, limit=3)
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant context from memory:\n"
            for mem in memories:
                memory_context += f"- {mem['content']}\n"
        
        # Get conversation context
        conn = sqlite3.connect(CONTEXT_DB)
        c = conn.cursor()
        c.execute("SELECT message, role FROM context WHERE session_id = ? ORDER BY timestamp DESC LIMIT 6", 
                 (session_id,))
        context_messages = []
        for row in reversed(c.fetchall()):
            context_messages.append({"role": row[1], "content": row[0]})
        conn.close()
        
        # Add memory context to the current message
        enhanced_message = message + memory_context
        
        # Prepare the request payload for Ollama
        messages_payload = context_messages + [{"role": "user", "content": enhanced_message}]
        
        payload = {
            "model": model,
            "messages": messages_payload,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        
        # Stream response from Ollama
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=30
        )
        
        if response.status_code != 200:
            emit('error', {'message': f'Ollama error: {response.status_code}'})
            return
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'message' in chunk and 'content' in chunk['message']:
                        token = chunk['message']['content']
                        full_response += token
                        emit('response_token', {'token': token})
                    
                    if chunk.get('done', False):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        # Store the response in memory if it contains useful information
        if len(full_response) > 50:  # Only store substantial responses
            rag_memory.store_memory(session_id, f"AI Response: {full_response}", importance=1)
        
        emit('response_complete', {'full_response': full_response})
        
    except requests.exceptions.Timeout:
        emit('error', {'message': 'Request timeout - Ollama may be busy'})
    except requests.exceptions.ConnectionError:
        emit('error', {'message': 'Cannot connect to Ollama - is it running?'})
    except Exception as e:
        emit('error', {'message': f'Unexpected error: {str(e)}'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to STONE server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

# Utility functions
def cleanup_old_context():
    conn = sqlite3.connect(CONTEXT_DB)
    c = conn.cursor()

    # Get all session_ids
    c.execute("SELECT DISTINCT session_id FROM context")
    session_ids = [row[0] for row in c.fetchall()]
    
    # For each session, keep only the latest 100 messages
    for sid in session_ids:
        c.execute("""
            DELETE FROM context 
            WHERE rowid NOT IN (
                SELECT rowid FROM context 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 100
            ) AND session_id = ?
        """, (sid, sid))
    
    # Clean RAG memory globally
    c.execute("""
        DELETE FROM rag_memory 
        WHERE rowid NOT IN (
            SELECT rowid FROM rag_memory 
            ORDER BY timestamp DESC 
            LIMIT 1000
        )
    """)
    
    conn.commit()
    conn.close()

def start_background_tasks():
    """Start background maintenance tasks"""
    def maintenance_loop():
        while True:
            time.sleep(3600)  # Run every hour
            try:
                cleanup_old_context()
                print("Performed maintenance cleanup")
            except Exception as e:
                print(f"Maintenance error: {e}")
    
    maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
    maintenance_thread.start()

if __name__ == '__main__':
    print("🚀 Starting STONE Enhanced Server...")
    print(f"   Server will run on: http://localhost:{PORT}")
    print(f"   Ollama URL: {OLLAMA_BASE_URL}")
    print(f"   Database: {CONTEXT_DB}")
    print("   Features: Token Streaming, Function Calling, RAG Memory, Context Storage")
    
    # Test Ollama connection
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama connected - {len(models)} models available")
        else:
            print(f"   ⚠️  Ollama connection issue: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ollama connection failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
    
    # Start background tasks
    start_background_tasks()
    
    # Run the server
    try:
        socketio.run(app, host='0.0.0.0', port=PORT, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n👋 STONE server shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")