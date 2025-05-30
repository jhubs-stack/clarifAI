👟 HappyFeet — Powered by ClarifAI
HappyFeet is a playful, locally-run AI shopping assistant demo. It pairs a modern React + Vite frontend with a Python + FAISS backend, enabling smart customer FAQ support through a conversational AI agent named Frankie.

Designed for in-person demos and local use, it requires no cloud deployment. Frankie is friendly, fast, and deeply integrated into the HappyFeet brand.

🧠 What Frankie Can Do
Answer product and support questions with style
Recommend shoes based on your personality and preferences
Help with order tracking, sizing, and return policies
Stay friendly, playful, and helpful—just like the HappyFeet brand

⚙️ Tech Stack
🔙 Backend
Python 3.10
FastAPI
FAISS (semantic search)
OpenAI API (text-embedding and chat-completion models)
dotenv for environment config

🖥 Frontend
React (with Vite)
Tailwind CSS for UI
Animated chat UI with avatar
Local asset integration

🚀 Local Setup
1. Clone the repository
git clone https://github.com/jhubs-stack/clarifAI.git
cd clarifAI

2. Backend Setup (Python)
cd backend
pyenv local 3.10.13  # Ensure correct Python version
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_index.py
uvicorn main:app --reload

3. Frontend Setup (React)
cd frontend
npm install
npm run dev

Frontend: http://localhost:5173
Backend: http://localhost:8000

🧪 Testing
Try asking Frankie things like:

“What’s your return policy?”
“Do you ship internationally?”
“Can I talk to a human?”
“I like jazz and Bauhaus design—got shoe recs?”

📄 License
MIT — have fun with it and customize for your own demo needs.
