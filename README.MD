# 🤖 Gemini API Request Example (Python)

This is a minimal Python project demonstrating how to use the [Google Gemini API](https://ai.google.dev/) via the official `google-generativeai` SDK.

---

## 📦 Features

- Sends text prompts to Gemini 1.5 Flash model
- Reads API key from a `.env` file
- Uses a virtual environment for dependency isolation
- Clean and minimal project structure

---

## 🛠️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/gemini-api-request.git
cd gemini-api-request
```

---

### 2. Set up a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Get your Gemini API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Generate your API key
3. Copy it

---

### 5. Create a `.env` file

```bash
cp .env.example .env
```

Edit `.env` and paste your API key:

```env
GEMINI_API_KEY=your-api-key-here
```

---

### 6. Run the script

```bash
python main.py
```

You should see a short response like:

```
AI is the simulation of human intelligence by machines.
```

---

## 📁 Project Structure

```
gemini-api-request/
├── .env              # Your API key (ignored by Git)
├── .env.example      # Template for .env
├── .gitignore        # Ignores venv, .env, etc.
├── main.py           # Main script
├── requirements.txt  # Python dependencies
└── README.md         # You're here
```
