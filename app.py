from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document
import spacy
import json
import os
import re

from rag.memory_engine import store_memory, search_memories

app = Flask(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "memories.json")

DOCUMENT_FOLDER = "uploads/documents"
IMAGE_FOLDER = "uploads/images"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

ALLOWED_DOCUMENTS = {"pdf", "docx", "txt"}
ALLOWED_IMAGES = {"png", "jpg", "jpeg", "webp"}

# Load spaCy
nlp = spacy.load("en_core_web_sm")


# =========================================================
# DATABASE
# =========================================================

def load_memories():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def save_memories(memories):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(memories, file, indent=4, ensure_ascii=False)


# =========================================================
# NLP - PEOPLE
# =========================================================

def extract_people(text):
    people = []

    # First use spaCy PERSON detection
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)

    # Fallback detection
    words = text.split()

    action_words = {
        "sent",
        "gave",
        "shared",
        "told",
        "asked",
        "called",
        "said",
        "helped",
        "gave",
        "messaged",
        "contacted",
        "provided"
    }

    for i, word in enumerate(words):

        clean_word = word.strip(".,!?;:'\"()[]")

        if not clean_word:
            continue

        if i < len(words) - 1:

            next_word = words[i + 1].strip(
                ".,!?;:'\"()[]"
            ).lower()

            if next_word in action_words:

                if clean_word[0].isupper():

                    people.append(clean_word)

    # Also detect names after common phrases
    patterns = [
        r"\bfrom\s+([A-Z][a-z]+)\b",
        r"\bwith\s+([A-Z][a-z]+)\b",
        r"\bby\s+([A-Z][a-z]+)\b",
        r"\b([A-Z][a-z]+)\s+sent\b",
        r"\b([A-Z][a-z]+)\s+gave\b",
        r"\b([A-Z][a-z]+)\s+shared\b"
    ]

    for pattern in patterns:

        matches = re.findall(pattern, text)

        for name in matches:
            people.append(name)

    # Remove obvious non-person words
    ignored_words = {
        "The",
        "This",
        "That",
        "Yesterday",
        "Today",
        "Tomorrow",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "AIML",
        "Python",
        "Django",
        "Flask"
    }

    people = [
        person
        for person in people
        if person not in ignored_words
    ]

    return list(dict.fromkeys(people))


# =========================================================
# NLP - TASK
# =========================================================

def detect_task(text):

    lower_text = text.lower()

    task_words = [
        "need to",
        "have to",
        "must",
        "should",
        "finish",
        "complete",
        "submit",
        "prepare",
        "do",
        "remind me",
        "remember to",
        "pending",
        "deadline"
    ]

    return any(
        word in lower_text
        for word in task_words
    )


# =========================================================
# NLP - PROJECT
# =========================================================

def detect_project(text):

    lower_text = text.lower()

    project_words = [
        "project",
        "aiml project",
        "software project",
        "college project",
        "final year project",
        "mini project"
    ]

    return any(
        word in lower_text
        for word in project_words
    )


# =========================================================
# NLP - DATE
# =========================================================

def detect_date(text):

    lower_text = text.lower()

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
        "next week",
        "next month"
    ]

    for word in date_words:

        if word in lower_text:
            return word

    return ""


# =========================================================
# SMART TAGS
# =========================================================

def detect_tags(text):

    lower_text = text.lower()

    tags = []

    if detect_task(text):
        tags.append("Task")

    if any(
        word in lower_text
        for word in [
            "college",
            "class",
            "exam",
            "assignment",
            "study",
            "lab",
            "semester",
            "student"
        ]
    ):
        tags.append("Study")

    if extract_people(text):
        tags.append("People")

    if detect_date(text):
        tags.append("Date")

    if detect_project(text):
        tags.append("Project")

    if any(
        word in lower_text
        for word in [
            "dataset",
            "model",
            "machine learning",
            "artificial intelligence",
            "aiml",
            "python",
            "code"
        ]
    ):
        tags.append("AI/ML")

    if not tags:
        tags.append("Note")

    return list(dict.fromkeys(tags))


# =========================================================
# CREATE MEMORY
# =========================================================

def create_memory(text, memory_type="memory"):

    memories = load_memories()

    people = extract_people(text)
    task = detect_task(text)
    project = detect_project(text)
    date_reference = detect_date(text)
    tags = detect_tags(text)

    new_memory = {
        "id": len(memories) + 1,
        "text": text,
        "type": memory_type,
        "tags": tags,
        "people": people,
        "task": task,
        "project": project,
        "date_reference": date_reference,
        "date": datetime.now().strftime("%d %b %Y"),
        "time": datetime.now().strftime("%I:%M %p")
    }

    memories.append(new_memory)
    save_memories(memories)

    # Store in ChromaDB
    try:
        store_memory(
            f"memory-{new_memory['id']}",
            text
        )
    except Exception as error:
        print("ChromaDB warning:", error)

    return new_memory


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_pdf_text(filepath):

    text = ""

    try:

        reader = PdfReader(filepath)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as error:
        print("PDF error:", error)

    return text[:20000]


def extract_docx_text(filepath):

    text = ""

    try:

        document = Document(filepath)

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as error:
        print("DOCX error:", error)

    return text[:20000]


def extract_txt_text(filepath):

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            return file.read()[:20000]

    except Exception as error:

        print("TXT error:", error)

        return ""


# =========================================================
# DOCUMENT SAVE
# =========================================================

def save_document(file):

    filename = secure_filename(file.filename)

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_DOCUMENTS:
        return None

    filepath = os.path.join(
        DOCUMENT_FOLDER,
        filename
    )

    file.save(filepath)

    if extension == "pdf":
        text = extract_pdf_text(filepath)

    elif extension == "docx":
        text = extract_docx_text(filepath)

    else:
        text = extract_txt_text(filepath)

    memories = load_memories()

    document_id = len(memories) + 1

    document_memory = {
        "id": document_id,
        "text": text if text else filename,
        "type": "document",
        "filename": filename,
        "tags": ["Document"],
        "people": [],
        "task": False,
        "project": False,
        "date_reference": "",
        "date": datetime.now().strftime("%d %b %Y"),
        "time": datetime.now().strftime("%I:%M %p")
    }

    memories.append(document_memory)

    save_memories(memories)

    try:

        store_memory(
            f"document-{document_id}",
            text if text else filename
        )

    except Exception as error:

        print("Document ChromaDB warning:", error)

    return document_memory


# =========================================================
# IMAGE SAVE
# =========================================================

def save_image(file):

    filename = secure_filename(file.filename)

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_IMAGES:
        return None

    filepath = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    file.save(filepath)

    memories = load_memories()

    image_id = len(memories) + 1

    image_memory = {
        "id": image_id,
        "text": f"Important image: {filename}",
        "type": "image",
        "filename": filename,
        "tags": ["Image"],
        "people": [],
        "task": False,
        "project": False,
        "date_reference": "",
        "date": datetime.now().strftime("%d %b %Y"),
        "time": datetime.now().strftime("%I:%M %p")
    }

    memories.append(image_memory)

    save_memories(memories)

    try:

        store_memory(
            f"image-{image_id}",
            f"Important image {filename}"
        )

    except Exception as error:

        print("Image ChromaDB warning:", error)

    return image_memory


# =========================================================
# PEOPLE
# =========================================================

def get_all_people():

    memories = load_memories()

    people = {}

    for memory in memories:

        if memory.get("type", "memory") != "memory":
            continue

        for person in memory.get("people", []):

            if person not in people:
                people[person] = []

            people[person].append(memory)

    return people


# =========================================================
# TASKS
# =========================================================

def get_tasks():

    memories = load_memories()

    tasks = []

    for memory in memories:

        if memory.get("type", "memory") != "memory":
            continue

        if memory.get("task") or "Task" in memory.get("tags", []):

            tasks.append(memory)

    return tasks


# =========================================================
# PROJECTS
# =========================================================

def get_projects():

    memories = load_memories()

    projects = []

    for memory in memories:

        if memory.get("type", "memory") != "memory":
            continue

        if memory.get("project") or "Project" in memory.get("tags", []):

            projects.append(memory)

    return projects


# =========================================================
# DOCUMENTS
# =========================================================

def get_documents():

    memories = load_memories()

    return [
        memory
        for memory in memories
        if memory.get("type") == "document"
    ]


# =========================================================
# IMAGES
# =========================================================

def get_images():

    memories = load_memories()

    return [
        memory
        for memory in memories
        if memory.get("type") == "image"
    ]


# =========================================================
# SMART REMINDERS
# =========================================================

def get_reminders():

    memories = load_memories()

    reminders = []

    today = datetime.now().date()

    for memory in memories:

        if memory.get("type", "memory") != "memory":
            continue

        reference = memory.get(
            "date_reference",
            ""
        ).lower()

        if not reference:
            continue

        show = False

        if reference == "today":
            show = True

        elif reference == "tomorrow":
            show = True

        elif reference in [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ]:
            show = True

        elif "next week" in reference:
            show = True

        elif "next month" in reference:
            show = True

        if show:
            reminders.append(memory)

    return reminders


# =========================================================
# MEMORY CONNECTIONS
# =========================================================

def get_connections():

    memories = load_memories()

    normal_memories = [
        memory
        for memory in memories
        if memory.get("type", "memory") == "memory"
    ]

    connections = []

    useful_words = {
        "project",
        "dataset",
        "model",
        "assignment",
        "college",
        "python",
        "aiml",
        "exam",
        "rahul",
        "code",
        "machine",
        "learning"
    }

    for i in range(len(normal_memories)):

        for j in range(i + 1, len(normal_memories)):

            first = normal_memories[i]
            second = normal_memories[j]

            first_words = set(
                re.findall(
                    r"\b[a-zA-Z]+\b",
                    first["text"].lower()
                )
            )

            second_words = set(
                re.findall(
                    r"\b[a-zA-Z]+\b",
                    second["text"].lower()
                )
            )

            shared = (
                first_words &
                second_words &
                useful_words
            )

            if shared:

                connections.append({
                    "first": first,
                    "second": second,
                    "shared": list(shared)
                })

    return connections


# =========================================================
# SEARCH EVERYTHING
# =========================================================

def search_everything(query):

    memories = load_memories()

    query_lower = query.lower()

    direct_results = []

    for memory in memories:

        searchable = " ".join([
            str(memory.get("text", "")),
            str(memory.get("filename", "")),
            " ".join(memory.get("tags", [])),
            " ".join(memory.get("people", []))
        ]).lower()

        if query_lower in searchable:

            direct_results.append(memory)

    semantic_results = []

    try:

        result = search_memories(
            query,
            number_of_results=5
        )

        documents = result.get(
            "documents",
            [[]]
        )

        if documents and documents[0]:

            for document in documents[0]:

                for memory in memories:

                    if (
                        memory.get("text") == document
                        and memory not in semantic_results
                    ):

                        semantic_results.append(memory)

    except Exception as error:

        print("Search warning:", error)

    combined = direct_results + semantic_results

    unique = []

    seen = set()

    for memory in combined:

        memory_id = memory.get("id")

        if memory_id not in seen:

            seen.add(memory_id)
            unique.append(memory)

    return unique


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_answer(question):

    relevant = search_everything(question)

    if not relevant:
        return "I could not find any relevant memory."

    context = "\n\n".join(
        memory.get("text", "")
        for memory in relevant[:5]
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        return (
            "Here is what I found in your memories:\n\n"
            + context
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are MemoryBridge AI.

Answer the user's question using only
the information provided in the memory context.

Memory context:
{context}

User question:
{question}

Give a clear and useful answer.
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except Exception as error:

        print("Gemini error:", error)

        return (
            "I found these relevant memories:\n\n"
            + context
        )


# =========================================================
# DASHBOARD
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
            in memory.get("text", "").lower()
        ]

    else:

        filtered = memories

    filtered = list(
        reversed(filtered)
    )

    people = get_all_people()

    tasks = get_tasks()

    projects = get_projects()

    documents = get_documents()

    images = get_images()

    reminders = get_reminders()

    connections = get_connections()

    return render_template_string(
        PAGE,
        memories=filtered,
        total=len(memories),
        tasks=tasks,
        projects=projects,
        documents=documents,
        images=images,
        people=people,
        reminders=reminders,
        connections=connections,
        search=search
    )


# =========================================================
# ADD MEMORY
# =========================================================

@app.route(
    "/add-memory",
    methods=["POST"]
)
def add_memory():

    text = request.form.get(
        "memory",
        ""
    ).strip()

    if text:
        create_memory(text)

    return redirect(url_for("home"))


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@app.route(
    "/upload-document",
    methods=["POST"]
)
def upload_document():

    file = request.files.get("document")

    if file and file.filename:

        save_document(file)

    return redirect(url_for("home"))


# =========================================================
# UPLOAD IMAGE
# =========================================================

@app.route(
    "/upload-image",
    methods=["POST"]
)
def upload_image():

    file = request.files.get("image")

    if file and file.filename:

        save_image(file)

    return redirect(url_for("home"))


# =========================================================
# ASK MEMORYBRIDGE
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    question = request.form.get(
        "question",
        ""
    ).strip()

    answer = generate_answer(question)

    memories = load_memories()

    return render_template_string(
        PAGE,
        memories=list(reversed(memories)),
        total=len(memories),
        tasks=get_tasks(),
        projects=get_projects(),
        documents=get_documents(),
        images=get_images(),
        people=get_all_people(),
        reminders=get_reminders(),
        connections=get_connections(),
        search="",
        answer=answer,
        question=question
    )


# =========================================================
# SEARCH
# =========================================================

@app.route("/search")
def search():

    query = request.args.get(
        "q",
        ""
    ).strip()

    results = search_everything(query)

    memories = load_memories()

    return render_template_string(
        PAGE,
        memories=list(reversed(memories)),
        total=len(memories),
        tasks=get_tasks(),
        projects=get_projects(),
        documents=get_documents(),
        images=get_images(),
        people=get_all_people(),
        reminders=get_reminders(),
        connections=get_connections(),
        search=query,
        search_results=results
    )


# =========================================================
# IMAGE ROUTE
# =========================================================

@app.route(
    "/images/<filename>"
)
def static_image(filename):

    return send_from_directory(
        IMAGE_FOLDER,
        filename
    )


# =========================================================
# DELETE
# =========================================================

@app.route(
    "/delete/<int:memory_id>"
)
def delete_memory(memory_id):

    memories = load_memories()

    memories = [
        memory
        for memory in memories
        if memory.get("id") != memory_id
    ]

    save_memories(memories)

    return redirect(url_for("home"))


# =========================================================
# HTML
# =========================================================

PAGE = """
<!DOCTYPE html>

<html>

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
    font-family: Arial, sans-serif;
    background: #f5f7fb;
    color: #172033;
}

.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    width: 245px;
    height: 100vh;
    background: #111827;
    color: white;
    padding: 28px 20px;
}

.logo {
    font-size: 25px;
    font-weight: bold;
}

.logo span {
    color: #60a5fa;
}

.subtitle {
    color: #9ca3af;
    font-size: 13px;
    margin-top: 8px;
    margin-bottom: 35px;
}

.nav a {
    display: block;
    color: #d1d5db;
    text-decoration: none;
    padding: 13px 10px;
    border-radius: 8px;
    margin-bottom: 5px;
}

.nav a:hover {
    background: #1f2937;
    color: white;
}

.bottom {
    position: absolute;
    bottom: 25px;
    font-size: 11px;
    color: #6b7280;
}

.main {
    margin-left: 245px;
    padding: 35px;
}

.top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.heading h1 {
    margin: 0;
    font-size: 30px;
}

.heading p {
    color: #6b7280;
}

.search-box {
    display: flex;
    gap: 8px;
}

.search-box input {
    padding: 12px;
    width: 250px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
}

button {
    border: none;
    border-radius: 8px;
    padding: 11px 18px;
    background: #2563eb;
    color: white;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.stats {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 15px;
    margin-bottom: 25px;
}

.stat,
.card,
.composer,
.memory {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
}

.stat-label {
    color: #6b7280;
    font-size: 13px;
}

.stat-number {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}

.composer {
    margin-bottom: 25px;
}

textarea {
    width: 100%;
    min-height: 110px;
    padding: 14px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    resize: vertical;
    margin-bottom: 10px;
}

.section-title {
    font-size: 21px;
    font-weight: bold;
    margin: 28px 0 12px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

.memory {
    margin-bottom: 12px;
}

.memory-top {
    display: flex;
    justify-content: space-between;
}

.memory-date {
    font-size: 12px;
    color: #6b7280;
}

.delete {
    color: #ef4444;
    text-decoration: none;
    font-size: 12px;
}

.memory-text {
    margin-top: 12px;
    line-height: 1.6;
}

.tags {
    margin-top: 12px;
}

.tag {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    padding: 5px 9px;
    border-radius: 20px;
    font-size: 11px;
    margin: 3px;
}

.person,
.task,
.project,
.reminder,
.connection {
    padding: 12px;
    border-bottom: 1px solid #eee;
}

.person:last-child,
.task:last-child,
.project:last-child {
    border-bottom: none;
}

.upload-box {
    margin-top: 10px;
}

.answer {
    white-space: pre-wrap;
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 18px;
    border-radius: 8px;
    line-height: 1.6;
}

.image-preview {
    max-width: 180px;
    max-height: 140px;
    margin-top: 10px;
    border-radius: 8px;
}

.empty {
    color: #6b7280;
    padding: 15px;
}

@media(max-width: 900px) {

    .sidebar {
        position: static;
        width: 100%;
        height: auto;
    }

    .bottom {
        display: none;
    }

    .main {
        margin-left: 0;
        padding: 20px;
    }

    .stats {
        grid-template-columns: 1fr 1fr;
    }

    .grid {
        grid-template-columns: 1fr;
    }

    .top {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }

}

</style>

</head>


<body>


<div class="sidebar">

    <div class="logo">
        🧠 Memory<span>Bridge</span>
    </div>

    <div class="subtitle">
        Intelligent Memory Workspace
    </div>

    <div class="nav">

        <a href="/">
            🏠 Dashboard
        </a>

        <a href="#ask">
            🔎 Ask MemoryBridge
        </a>

        <a href="#add">
            🧠 Add Memory
        </a>

        <a href="#timeline">
            📅 Timeline
        </a>

        <a href="#people">
            👤 People
        </a>

        <a href="#tasks">
            ✅ Tasks
        </a>

        <a href="#projects">
            📁 Projects
        </a>

        <a href="#documents">
            📄 Documents
        </a>

        <a href="#images">
            🖼️ Important Images
        </a>

    </div>

    <div class="bottom">
        MemoryBridge AI · Intelligent Personal Memory System
    </div>

</div>


<div class="main">


<div class="top">

    <div class="heading">

        <h1>
            MemoryBridge AI
        </h1>

        <p>
            Your intelligent personal memory and knowledge assistant.
        </p>

    </div>

    <form
        class="search-box"
        method="GET"
        action="/"
    >

        <input
            type="text"
            name="search"
            placeholder="Search memories..."
            value="{{ search }}"
        >

        <button>
            Search
        </button>

    </form>

</div>


<!-- STATS -->

<div class="stats">

    <div class="stat">
        <div class="stat-label">
            Total Memories
        </div>

        <div class="stat-number">
            {{ total }}
        </div>
    </div>


    <div class="stat">
        <div class="stat-label">
            Tasks
        </div>

        <div class="stat-number">
            {{ tasks|length }}
        </div>
    </div>


    <div class="stat">
        <div class="stat-label">
            Projects
        </div>

        <div class="stat-number">
            {{ projects|length }}
        </div>
    </div>


    <div class="stat">
        <div class="stat-label">
            Documents
        </div>

        <div class="stat-number">
            {{ documents|length }}
        </div>
    </div>


    <div class="stat">
        <div class="stat-label">
            Images
        </div>

        <div class="stat-number">
            {{ images|length }}
        </div>
    </div>

</div>


<!-- ASK -->

<div class="section-title" id="ask">
    🔎 Ask MemoryBridge
</div>

<div class="card">

    <p>
        Ask questions about memories and uploaded documents.
    </p>

    <form method="POST" action="/ask">

        <input
            type="text"
            name="question"
            placeholder="Example: What is pending for my AIML project?"
            style="width:100%;padding:12px;border:1px solid #ddd;border-radius:8px;"
            required
        >

        <br><br>

        <button>
            Ask AI
        </button>

    </form>

    {% if answer %}

    <br>

    <div class="answer">
        {{ answer }}
    </div>

    {% endif %}

</div>


<!-- ADD MEMORY -->

<div class="section-title" id="add">
    🧠 Capture a Memory
</div>

<div class="composer">

    <p>
        Save people, tasks, dates and project information.
    </p>

    <form
        method="POST"
        action="/add-memory"
    >

        <textarea
            name="memory"
            placeholder="Example: Rahul sent me the AIML project dataset yesterday. I need to finish the model before Friday."
            required
        ></textarea>

        <button>
            Save Memory
        </button>

    </form>

</div>


<!-- DOCUMENT -->

<div class="section-title" id="documents">
    📄 Upload Documents
</div>

<div class="card">

    <p>
        Upload PDF, DOCX or TXT files. Their text becomes searchable.
    </p>

    <form
        method="POST"
        action="/upload-document"
        enctype="multipart/form-data"
    >

        <input
            type="file"
            name="document"
            accept=".pdf,.docx,.txt"
            required
        >

        <br><br>

        <button>
            Upload Document
        </button>

    </form>

</div>


<!-- IMAGE -->

<div class="section-title" id="images">
    🖼️ Save Important Image
</div>

<div class="card">

    <p>
        Store important images inside your MemoryBridge workspace.
    </p>

    <form
        method="POST"
        action="/upload-image"
        enctype="multipart/form-data"
    >

        <input
            type="file"
            name="image"
            accept=".png,.jpg,.jpeg,.webp"
            required
        >

        <br><br>

        <button>
            Save Image
        </button>

    </form>

</div>


<!-- PEOPLE -->

<div class="section-title" id="people">
    👤 People
</div>

<div class="card">

    <p>
        People discovered using NLP.
    </p>

    {% if people %}

        {% for person, person_memories in people.items() %}

            <div class="person">

                <strong>
                    {{ person }}
                </strong>

                <br>

                <small>
                    {{ person_memories|length }}
                    related memor{{ "y" if person_memories|length == 1 else "ies" }}
                </small>

            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            No people detected yet.
        </div>

    {% endif %}

</div>


<!-- TASKS -->

<div class="section-title" id="tasks">
    ✅ Important Tasks
</div>

<div class="card">

    <p>
        Tasks discovered from your memories.
    </p>

    {% if tasks %}

        {% for task in tasks %}

            <div class="task">

                {{ task.text }}

                {% if task.date_reference %}

                <br>

                <small>
                    📅 {{ task.date_reference }}
                </small>

                {% endif %}

            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            No tasks detected yet.
        </div>

    {% endif %}

</div>


<!-- PROJECTS -->

<div class="section-title" id="projects">
    📁 Projects
</div>

<div class="card">

    <p>
        Project-related information discovered by AI.
    </p>

    {% if projects %}

        {% for project in projects %}

            <div class="project">
                {{ project.text }}
            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            No project memories yet.
        </div>

    {% endif %}

</div>


<!-- REMINDERS -->

<div class="section-title">
    🔔 Smart Reminders
</div>

<div class="card">

    {% if reminders %}

        {% for reminder in reminders %}

            <div class="reminder">

                🔔 {{ reminder.text }}

                <br>

                <small>
                    Reminder: {{ reminder.date_reference }}
                </small>

            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            No smart reminders yet.
        </div>

    {% endif %}

</div>


<!-- CONNECTIONS -->

<div class="section-title">
    🔗 Memory Connections
</div>

<div class="card">

    {% if connections %}

        {% for connection in connections %}

            <div class="connection">

                <strong>
                    Related memories
                </strong>

                <br><br>

                {{ connection.first.text }}

                <br><br>

                ↕ Connected through:
                {{ connection.shared|join(", ") }}

                <br><br>

                {{ connection.second.text }}

            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            Save more related memories to discover connections.
        </div>

    {% endif %}

</div>


<!-- SAVED DOCUMENTS -->

<div class="section-title">
    📄 Saved Documents
</div>

<div class="card">

    {% if documents %}

        {% for document in documents %}

            <div class="project">

                📄 {{ document.filename }}

                <br>

                <small>
                    {{ document.date }}
                </small>

            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            No documents uploaded yet.
        </div>

    {% endif %}

</div>


<!-- SAVED IMAGES -->

<div class="section-title">
    🖼️ Important Images
</div>

<div class="card">

    {% if images %}

        {% for image in images %}

            <div class="project">

                🖼️ {{ image.filename }}

                <br>

                <img
                    class="image-preview"
                    src="{{ url_for('static_image', filename=image.filename) }}"
                >

            </div>

        {% endfor %}

    {% else %}

        <div class="empty">
            No important images saved yet.
        </div>

    {% endif %}

</div>


<!-- MEMORIES -->

<div class="section-title" id="timeline">
    🗂️ Your Memory Space
</div>

{% if memories %}

    {% for memory in memories %}

        <div class="memory">

            <div class="memory-top">

                <div class="memory-date">

                    📅 {{ memory.date }}
                    &nbsp; • &nbsp;
                    🕒 {{ memory.time }}

                </div>

                <a
                    class="delete"
                    href="/delete/{{ memory.id }}"
                >
                    Delete
                </a>

            </div>


            <div class="memory-text">

                {% if memory.type == "document" %}

                    📄

                {% elif memory.type == "image" %}

                    🖼️

                {% else %}

                    🧠

                {% endif %}

                {{ memory.text }}

            </div>


            {% if memory.date_reference %}

                <div style="margin-top:8px;">
                    📅 Task Date:
                    {{ memory.date_reference }}
                </div>

            {% endif %}


            <div class="tags">

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
    </div>

{% endif %}


</div>

</body>

</html>
"""


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )