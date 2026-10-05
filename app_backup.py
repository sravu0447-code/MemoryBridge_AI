from flask import Flask, request, redirect, url_for, render_template_string
from datetime import datetime
import json
import os

from rag.memory_engine import store_memory, search_memories

from google import genai
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# DATA STORAGE
# =========================================================

DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "memories.json")

os.makedirs(DATA_FOLDER, exist_ok=True)


# =========================================================
# GEMINI CLIENT
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None


# =========================================================
# LOAD MEMORIES
# =========================================================

def load_memories():

    if not os.path.exists(DATA_FILE):
        return []

    try:

        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:

        return []


# =========================================================
# SAVE MEMORIES
# =========================================================

def save_memories(memories):

    with open(DATA_FILE, "w", encoding="utf-8") as file:

        json.dump(
            memories,
            file,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# INFORMATION EXTRACTION
# =========================================================

def extract_information(text):

    lower_text = text.lower()

    tags = []


    # -------------------------
    # TASK
    # -------------------------

    task_words = [
        "need to",
        "have to",
        "must",
        "should",
        "submit",
        "complete",
        "finish",
        "remind me",
        "remember to"
    ]

    if any(word in lower_text for word in task_words):

        tags.append("Task")


    # -------------------------
    # STUDY
    # -------------------------

    study_words = [
        "college",
        "class",
        "exam",
        "assignment",
        "project",
        "lab",
        "semester",
        "study",
        "aiml",
        "btech"
    ]

    if any(word in lower_text for word in study_words):

        tags.append("Study")


    # -------------------------
    # PEOPLE
    # -------------------------

    people_words = [
        "friend",
        "sir",
        "madam",
        "teacher",
        "brother",
        "sister",
        "mom",
        "dad",
        "rahul"
    ]

    if any(word in lower_text for word in people_words):

        tags.append("People")


    # -------------------------
    # DATE
    # -------------------------

    date_words = [
        "today",
        "tomorrow",
        "yesterday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december"
    ]

    if any(word in lower_text for word in date_words):

        tags.append("Date")


    # -------------------------
    # PROJECT
    # -------------------------

    if "project" in lower_text:

        tags.append("Project")


    # -------------------------
    # DEFAULT
    # -------------------------

    if not tags:

        tags.append("Note")


    return list(dict.fromkeys(tags))


# =========================================================
# AI QUESTION ANSWERING
# =========================================================

def ask_memorybridge(question):

    try:

        # -----------------------------------------
        # SEARCH CHROMADB
        # -----------------------------------------

        results = search_memories(
            question,
            number_of_results=5
        )

        documents = results.get("documents", [[]])[0]


        if not documents:

            return "I could not find any relevant memories."


        # -----------------------------------------
        # CREATE MEMORY CONTEXT
        # -----------------------------------------

        memory_context = "\n".join(
            f"- {memory}"
            for memory in documents
        )


        # -----------------------------------------
        # IF GEMINI IS AVAILABLE
        # -----------------------------------------

        if gemini_client:

            prompt = f"""
You are MemoryBridge AI, a personal memory assistant.

Answer the user's question using ONLY the memories provided below.

Memories:
{memory_context}

User question:
{question}

Rules:
1. Use the memories as the main source of information.
2. Do not invent information.
3. If the memories do not contain enough information, clearly say that.
4. Give a short, natural and useful answer.
5. Mention relevant people, tasks, projects or dates when available.
"""

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text


        # -----------------------------------------
        # FALLBACK WITHOUT GEMINI
        # -----------------------------------------

        return (
            "I found these related memories:\n\n"
            + "\n".join(
                f"• {memory}"
                for memory in documents
            )
        )


    except Exception as error:

        print("AI question error:", error)

        return (
            "I found related memories, but the AI answer "
            "could not be generated right now.\n\n"
            + "\n".join(
                f"• {memory}"
                for memory in documents
            )
        )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    memories = load_memories()

    search = request.args.get(
        "search",
        ""
    ).strip()


    if search:

        filtered = [
            memory
            for memory in memories
            if search.lower()
            in memory["text"].lower()
        ]

    else:

        filtered = memories


    filtered = list(reversed(filtered))


    total = len(memories)

    tasks = sum(
        "Task" in memory.get("tags", [])
        for memory in memories
    )

    projects = sum(
        "Project" in memory.get("tags", [])
        for memory in memories
    )


    return render_template_string(

        PAGE,

        memories=filtered,

        total=total,

        tasks=tasks,

        projects=projects,

        search=search,

        answer=None,

        question=""
    )


# =========================================================
# SAVE MEMORY
# =========================================================

@app.route("/save", methods=["POST"])
def save_memory():

    text = request.form.get(
        "memory",
        ""
    ).strip()


    if text:

        memories = load_memories()


        if memories:

            memory_id = max(
                memory["id"]
                for memory in memories
            ) + 1

        else:

            memory_id = 1


        tags = extract_information(text)


        new_memory = {

            "id": memory_id,

            "text": text,

            "tags": tags,

            "date": datetime.now().strftime(
                "%d %b %Y"
            ),

            "time": datetime.now().strftime(
                "%I:%M %p"
            )
        }


        memories.append(new_memory)

        save_memories(memories)


        # Store in ChromaDB

        try:

            store_memory(
                memory_id,
                text
            )

            print(
                f"AI memory stored: {memory_id}"
            )

        except Exception as error:

            print(
                "AI memory storage error:",
                error
            )


    return redirect(
        url_for("home")
    )


# =========================================================
# ASK MEMORYBRIDGE
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get(
        "question",
        ""
    ).strip()


    answer = None


    if question:

        answer = ask_memorybridge(
            question
        )


    memories = load_memories()


    total = len(memories)

    tasks = sum(
        "Task" in memory.get("tags", [])
        for memory in memories
    )

    projects = sum(
        "Project" in memory.get("tags", [])
        for memory in memories
    )


    return render_template_string(

        PAGE,

        memories=list(reversed(memories)),

        total=total,

        tasks=tasks,

        projects=projects,

        search="",

        answer=answer,

        question=question
    )


# =========================================================
# DELETE MEMORY
# =========================================================

@app.route("/delete/<int:memory_id>")
def delete_memory(memory_id):

    memories = load_memories()


    memories = [

        memory

        for memory in memories

        if memory["id"] != memory_id
    ]


    save_memories(memories)


    return redirect(
        url_for("home")
    )


# =========================================================
# WEBSITE
# =========================================================

PAGE = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>MemoryBridge AI</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f5f7fb;

    color: #1f2937;
}


.sidebar {

    position: fixed;

    left: 0;

    top: 0;

    width: 240px;

    height: 100vh;

    background: #111827;

    color: white;

    padding: 28px 20px;
}


.logo {

    font-size: 25px;

    font-weight: bold;

    margin-bottom: 8px;
}


.subtitle {

    color: #9ca3af;

    font-size: 13px;

    margin-bottom: 35px;
}


.nav a {

    display: block;

    color: #d1d5db;

    text-decoration: none;

    padding: 13px;

    border-radius: 9px;

    margin-bottom: 8px;

    font-size: 14px;
}


.nav a:hover {

    background: #1f2937;

    color: white;
}


.main {

    margin-left: 240px;

    padding: 40px;

    max-width: 1300px;
}


.header {

    margin-bottom: 30px;
}


.header h1 {

    margin: 0;

    font-size: 34px;
}


.header p {

    color: #6b7280;

    margin-top: 8px;
}


.stats {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 18px;

    margin-bottom: 28px;
}


.stat {

    background: white;

    padding: 22px;

    border-radius: 15px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.05);
}


.stat-title {

    color: #6b7280;

    font-size: 13px;
}


.stat-number {

    font-size: 30px;

    font-weight: bold;

    margin-top: 8px;
}


.card {

    background: white;

    border-radius: 16px;

    padding: 25px;

    margin-bottom: 25px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.05);
}


.card h2 {

    margin-top: 0;

    font-size: 20px;
}


textarea {

    width: 100%;

    min-height: 120px;

    border: 1px solid #d1d5db;

    border-radius: 10px;

    padding: 15px;

    font-size: 15px;

    resize: vertical;

    outline: none;
}


textarea:focus {

    border-color: #4f46e5;
}


button {

    margin-top: 12px;

    background: #4f46e5;

    color: white;

    border: none;

    padding: 12px 22px;

    border-radius: 9px;

    cursor: pointer;

    font-size: 14px;
}


button:hover {

    background: #4338ca;
}


.ask-box {

    display: flex;

    gap: 10px;
}


.ask-box input {

    flex: 1;

    padding: 14px;

    border:
        1px solid #d1d5db;

    border-radius: 9px;

    font-size: 15px;

    outline: none;
}


.answer {

    margin-top: 20px;

    background: #eef2ff;

    border-left:
        4px solid #4f46e5;

    padding: 18px;

    border-radius: 8px;

    white-space: pre-line;

    line-height: 1.6;
}


.memory {

    border-bottom:
        1px solid #e5e7eb;

    padding: 18px 0;
}


.memory:last-child {

    border-bottom: none;
}


.memory-text {

    font-size: 15px;

    line-height: 1.5;

    margin-bottom: 10px;
}


.memory-date {

    font-size: 12px;

    color: #9ca3af;

    margin-bottom: 10px;
}


.tag {

    display: inline-block;

    background: #eef2ff;

    color: #4338ca;

    padding: 5px 9px;

    border-radius: 20px;

    font-size: 11px;

    margin-right: 5px;
}


.delete {

    float: right;

    color: #ef4444;

    text-decoration: none;

    font-size: 12px;
}


.empty {

    text-align: center;

    padding: 30px;

    color: #9ca3af;
}


@media(max-width: 800px) {

    .sidebar {

        position: relative;

        width: 100%;

        height: auto;
    }


    .main {

        margin-left: 0;

        padding: 20px;
    }


    .stats {

        grid-template-columns: 1fr;
    }

}


</style>

</head>


<body>


<!-- SIDEBAR -->

<div class="sidebar">

    <div class="logo">
        🧠 MemoryBridge
    </div>

    <div class="subtitle">
        Intelligent Memory Workspace
    </div>


    <div class="nav">

        <a href="/">
            ◉ Dashboard
        </a>

        <a href="#ask">
            🔎 Ask MemoryBridge
        </a>

        <a href="#add">
            ＋ Add Memory
        </a>

        <a href="#memories">
            ▣ My Memories
        </a>

    </div>

</div>


<!-- MAIN -->

<div class="main">


    <div class="header">

        <h1>
            Your Memory Space
        </h1>

        <p>
            Capture information today.
            Retrieve it when you need it.
        </p>

    </div>


    <!-- STATS -->

    <div class="stats">

        <div class="stat">

            <div class="stat-title">
                Total Memories
            </div>

            <div class="stat-number">
                {{ total }}
            </div>

        </div>


        <div class="stat">

            <div class="stat-title">
                Important Tasks
            </div>

            <div class="stat-number">
                {{ tasks }}
            </div>

        </div>


        <div class="stat">

            <div class="stat-title">
                Project Memories
            </div>

            <div class="stat-number">
                {{ projects }}
            </div>

        </div>

    </div>


    <!-- ASK MEMORYBRIDGE -->

    <div class="card" id="ask">

        <h2>
            🔎 Ask MemoryBridge
        </h2>

        <p>
            Ask a question about information
            you previously saved.
        </p>


        <form method="POST"
              action="/ask">

            <div class="ask-box">

                <input
                    type="text"
                    name="question"
                    placeholder="Example: What is pending for my AIML project?"
                    value="{{ question }}"
                    required
                >

                <button type="submit">
                    Ask AI
                </button>

            </div>

        </form>


        {% if answer %}

        <div class="answer">

            <strong>
                MemoryBridge AI
            </strong>

            <br><br>

            {{ answer }}

        </div>

        {% endif %}

    </div>


    <!-- ADD MEMORY -->

    <div class="card" id="add">

        <h2>
            🧠 Capture a Memory
        </h2>

        <p>
            Write anything you want
            MemoryBridge to remember.
        </p>


        <form method="POST"
              action="/save">

            <textarea
                name="memory"
                placeholder="Example: Rahul sent me the AIML project dataset..."
                required></textarea>


            <button type="submit">
                Save Memory
            </button>

        </form>

    </div>


    <!-- MEMORIES -->

    <div class="card" id="memories">

        <h2>
            Recent Memories
        </h2>


        {% if memories %}


            {% for memory in memories %}

            <div class="memory">

                <a
                    class="delete"
                    href="/delete/{{ memory.id }}"
                >
                    Delete
                </a>


                <div class="memory-date">

                    {{ memory.date }}
                    ·
                    {{ memory.time }}

                </div>


                <div class="memory-text">

                    {{ memory.text }}

                </div>


                <div>

                    {% for tag in memory.tags %}

                    <span class="tag">

                        {{ tag }}

                    </span>

                    {% endfor %}

                </div>

            </div>

            {% endfor %}


        {% else %}

            <div class="empty">

                Your memory space is empty.

                <br><br>

                Add your first memory above.

            </div>

        {% endif %}


    </div>


</div>


</body>

</html>

"""


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )