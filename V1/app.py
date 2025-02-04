from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import sqlite3
import pandas as pd
from datetime import datetime
from groq import Groq
from fastapi.responses import JSONResponse

# Replace with your actual Groq API key
GROQ_API_KEY = "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"

templates = Jinja2Templates(directory="templates")

# Define the 4 tracks with their eligibility criteria
tracks = {
    "NIDHI Prayas": """Eligibility: Indian entrepreneurs, with a technology-based startup at proof of concept stage. No age limit.It is a must for the startup to have a hardware product, cannot be software. The applicant must have a working prototype and must be at least 18 years of age. The startup should be from an incubator that is recognized by DST.""",
    "NIDHI EIR": """Eligibility: Indian citizens, at least 18 years old, with a technology-based idea or startup. They must not be involved in another full-time job or education program and must commit to full-time work on their startup idea. The idea should have a significant impact in its field and must be incubated at a recognized incubator.""",
    "NIDHI SSS": """Eligibility: Startups should have been incorporated within the last 2 years, be technology-based, and have a scalable product idea. The startup must be Indian-owned, and at least 51% of the shares should be held by Indian promoters. Startups must be registered at a recognized incubator. A business plan and technical competence are essential.""",
    "Startup India Seed Fund": """Eligibility: Startups must be registered with DPIIT, have been incorporated for no more than 2 years, and should be technology-driven with a scalable business model. The startup should not have received more than ₹10 lakh from any other government scheme, and it should be incubated at a recognized incubator or startup hub."""
}

# Initialize the FastAPI app
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect("startup_classifier.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS classification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ip_address TEXT,
                    input_prompt TEXT,
                    output TEXT,
                    email TEXT
                )''')
    conn.commit()
    conn.close()

# Function to classify startup using Groq API
# Function to classify startup using Groq API
def classify_startup(brief: str):
    prompt = f"""
    You are an AI startup advisor. Your task is to analyze the startup's brief and rank the tracks from highest to lowest likelihood of eligibility.
    deeply understand what the startup is doing and all of its nuasces and make sure you take into account all the eligibilty criterias such as hardware software etc 
    Here are the 4 tracks and their eligibility criteria:
    {tracks}

    Startup Brief: "{brief}"

    Based on the brief, return only a ranked list of track names from highest to lowest likelihood,number them. If no track is a fit, return "None".
    Reply in a clean format, with only the track names, one per line,no extra commentary.
    """

    # Initialize the Groq client
    client = Groq(api_key=GROQ_API_KEY)

    try:
        # Create the completion using Groq's chat API
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Use the appropriate model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=0.5,
            stream=True,
            stop=None,
        )

        # Read and format the completion output
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""

        return result.strip()  # Only return the clean track names

    except Exception as e:
        return f"An error occurred: {str(e)}"


# Log data into the SQLite database
def log_to_db(ip_address: str, prompt: str, output: str, email: str = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("startup_classifier.db")
    c = conn.cursor()

    if email:
        # Delete the previous log without email
        c.execute('''DELETE FROM classification_logs
                     WHERE ip_address = ? AND email IS NULL''', (ip_address,))
    
    # Insert new log with or without email
    c.execute('''INSERT INTO classification_logs (timestamp, ip_address, input_prompt, output, email)
                 VALUES (?, ?, ?, ?, ?)''', (timestamp, ip_address, prompt, output, email))
    
    conn.commit()
    conn.close()

# Home page with form to input startup brief
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open('templates/index.html', 'r') as f:
        return f.read()
@app.get("/api/dashboard-data")
async def dashboard_data():
    total_users_today, unique_entries_email, unique_entries_ip, logs = get_dashboard_data()
    
    # Convert logs to a list of dictionaries for better JSON serialization
    formatted_logs = []
    for log in logs:
        formatted_logs.append({
            "id": log[0],
            "timestamp": log[1],
            "ip_address": log[2],
            "input": log[3],
            "output": log[4],
            "email": log[5] if log[5] else None
        })
    
    return {
        "total_users_today": total_users_today,
        "unique_entries_email": unique_entries_email,
        "unique_entries_ip": unique_entries_ip,
        "logs": formatted_logs
    }

# Endpoint to handle classification
@app.post("/classify")
async def classify(request: Request):
    data = await request.json()
    startup_brief = data.get("brief")
    ip_address = request.client.host
    result = classify_startup(startup_brief)

    # Log to DB without email
    log_to_db(ip_address, startup_brief, result)

    return {"result": result}

# Endpoint to save email
@app.post("/save-email")
async def save_email(request: Request):
    data = await request.json()
    email = data.get("email")
    startup_brief = data.get("brief")
    ip_address = request.client.host
    result = classify_startup(startup_brief)

    # Log to DB with email
    log_to_db(ip_address, startup_brief, result, email)

    return {"message": "Email saved successfully and data logged."}

# Function to get data for the dashboard
def get_dashboard_data():
    conn = sqlite3.connect("startup_classifier.db")
    c = conn.cursor()

    # Get total users today
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute('''SELECT COUNT(DISTINCT ip_address) FROM classification_logs
                 WHERE DATE(timestamp) = ?''', (today,))
    total_users_today = c.fetchone()[0]

    # Get unique entries (distinct email and ip address combinations)
    c.execute('''SELECT COUNT(DISTINCT ip_address, email) FROM classification_logs''')
    unique_entries = c.fetchone()[0]

    # Get logs for display in the table
    c.execute('''SELECT * FROM classification_logs ORDER BY timestamp DESC LIMIT 20''')
    logs = c.fetchall()

    conn.close()
    return total_users_today, unique_entries, logs

# Function to save logs to CSV
def save_logs_to_csv():
    conn = sqlite3.connect("startup_classifier.db")
    c = conn.cursor()
    c.execute('''SELECT * FROM classification_logs''')
    rows = c.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["ID", "Timestamp", "IP Address", "Input Prompt", "Output", "Email"])

    # Save DataFrame to CSV
    csv_path = "logs.csv"
    df.to_csv(csv_path, index=False)
    conn.close()
    return csv_path

def get_dashboard_data():
    conn = sqlite3.connect("startup_classifier.db")
    c = conn.cursor()

    # Get total users today
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute('''SELECT COUNT(DISTINCT ip_address) FROM classification_logs
                 WHERE DATE(timestamp) = ?''', (today,))
    total_users_today = c.fetchone()[0]

    # Get unique email entries
    c.execute('''SELECT COUNT(DISTINCT email) FROM classification_logs WHERE email IS NOT NULL''')
    unique_entries_email = c.fetchone()[0]

    # Get unique IP address entries
    c.execute('''SELECT COUNT(DISTINCT ip_address) FROM classification_logs''')
    unique_entries_ip = c.fetchone()[0]

    # Get logs for display in the table
    c.execute('''SELECT * FROM classification_logs ORDER BY timestamp DESC LIMIT 20''')
    logs = c.fetchall()

    conn.close()
    return total_users_today, unique_entries_email, unique_entries_ip, logs
# Dashboard page
@app.get("/dashboard")
async def dashboard(request: Request):
    total_users_today, unique_entries_email, unique_entries_ip, logs = get_dashboard_data()

    # Render the dashboard page with data
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_users_today": total_users_today,
        "unique_entries_email": unique_entries_email,
        "unique_entries_ip": unique_entries_ip,
        "logs": logs
    })
# CSV download endpoint
@app.get("/download-csv")
async def download_csv():
    csv_path = save_logs_to_csv()
    return FileResponse(csv_path, media_type='text/csv', filename='logs.csv')

# Initialize the DB when the app starts
init_db()
