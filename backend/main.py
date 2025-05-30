from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI
import random
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI()

app = FastAPI()

# Allow frontend dev server to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"  # local dev only
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 

# === Utility and Data Loading Functions ===

def load_index():
    """
    Loads the FAISS index and FAQ data from a pickle file.
    Returns:
        dict: A dictionary with 'index' (FAISS index) and 'faqs' (list of FAQs).
    """
    with open("vectorstore/faiss_index.pkl", "rb") as f:
        index, faqs = pickle.load(f)
    return {"index": index, "faqs": faqs}

def load_products():
    """
    Loads the product catalog from a local JSON file.
    Returns:
        list: List of product dictionaries.
    """
    with open("data/products.json", "r") as f:
        return json.load(f)

def embed_text(text):
    """
    Generates an embedding vector for a given query string using OpenAI's embedding API.
    Args:
        text (str): The input text to embed.
    Returns:
        list: The embedding vector as a list of floats.
    """
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_faq(data, query, k=3):
    """
    Searches the FAQ index for the top-k most similar questions to the user query.
    Args:
        data (dict): Dictionary containing 'index' and 'faqs'.
        query (str): The user query string.
        k (int): Number of top matches to return.
    Returns:
        list: List of the top-k most relevant FAQ entries.
    """
    index = data["index"]
    faqs = data["faqs"]
    query_vector = embed_text(query)
    query_vector_np = np.array(query_vector, dtype="float32").reshape(1, -1)
    D, I = index.search(query_vector_np, k)
    return [faqs[i] for i in I[0] if i < len(faqs)]

def match_products(query: str, products: list, max_results: int = 3) -> list:
    """
    Matches user queries with relevant products based on tag overlap.
    Args:
        query (str): The user query string.
        products (list): List of product dictionaries.
        max_results (int): Maximum number of products to return.
    Returns:
        list: List of matched product dictionaries.
    """
    query_words = set(query.lower().split())
    matches = []
    for product in products:
        score = sum(any(tag in word or word in tag for word in query_words) for tag in product["tags"])
        if score > 0:
            matches.append((score, product))
    matches.sort(reverse=True, key=lambda x: x[0])
    return [match[1] for match in matches[:max_results]]

def extract_style_keywords(history):
    """
    Extracts style-related keywords from the user's conversation history.
    Args:
        history (list): List of message dictionaries from the conversation.
    Returns:
        list: List of unique style keywords found in the conversation.
    """
    if not history:
        return []
    style_keywords = [
        "retro", "vintage", "90s", "hip hop", "hip-hop", "rock", "indie", "hawaiian", "preppy",
        "punk", "grunge", "boho", "sporty", "minimalist", "casual", "formal", "chic", "coastal",
        "beach", "urban", "weekend", "laid-back", "old school", "classic", "funky", "abstract", "modern"
    ]
    found = set()
    for msg in history:
        content = msg.get("content", "").lower()
        for keyword in style_keywords:
            if keyword in content:
                found.add(keyword.replace("-", " "))  # Normalize "hip-hop" → "hip hop"
    return list(found)

def summarize_style_keywords(keywords: list) -> str:
    """
    Summarizes a list of style keywords into a human-readable string.
    Args:
        keywords (list): List of keywords.
    Returns:
        str: A readable summary string.
    """
    if not keywords:
        return ""
    if len(keywords) == 1:
        return keywords[0]
    return ", ".join(keywords[:-1]) + f" and {keywords[-1]}"

def get_style_summary_from_history(history: list) -> str:
    """
    Builds a readable style summary from extracted keywords in conversation history.
    Args:
        history (list): Conversation history.
    Returns:
        str: Human-readable style summary.
    """
    keywords = extract_style_keywords(history)
    return summarize_style_keywords(keywords)

def needs_handoff(query: str) -> bool:
    """
    Detects whether the user request should be escalated to a human agent.
    Args:
        query (str): The user query string.
    Returns:
        bool: True if handoff is needed, else False.
    """
    triggers = [
        "human", "real person", "talk to someone", "speak to agent", "speak with someone",
        "customer service", "representative", "get help", "escalate", 
        "return", "about a return", "help with a return", "need a return", "talk to someone about a return"
    ]
    return any(trigger in query.lower() for trigger in triggers)

# === Handoff Messaging ===

handoff_responses = [
    "I’m handing this over to one of our real-life HappyFeet stars right now. 💫 They’ll step in with the perfect solution, tailored just for you. Sit tight — help is already lacing up!",
    "Looks like this one’s out of my sole jurisdiction 👟 — I’m looping in a HappyFeet human as we speak! They’ve got just the right fit for what you need. Hang tight, superstar!",
    "I totally get where you’re coming from — and I want to make sure everything’s crystal clear. I’m handing things off to one of our fabulous team members now so they can walk you through it step-by-step. 💬✨",
    "Oof — I’ve reached the edge of my insoles on this one. No worries though, I’m passing you over to one of our HappyFeet heroes right now. They’ve got the tools (and the shoes) to make things right. 👟🛠️",
    "I wish I could take this all the way, but this is a perfect moment for one of our real HappyFeet folks to step in. I’ve already started the pass-off — they’ll help you tie everything up in no time. 🎀👣"
]

def get_handoff_response():
    """
    Returns a randomized, brand-aligned handoff message for escalation to a human agent.
    Returns:
        str: Handoff message.
    """
    return random.choice(handoff_responses)


def generate_response(query, context, history=None):
    """
    Generates a brand-voiced, contextual response using retrieved FAQs, user history, and style/product logic.
    Args:
        query (str): The user's query.
        context (list): List of FAQ entries relevant to the query.
        history (list, optional): Conversation history for context and continuity.
    Returns:
        str: The generated response text.
    """
    context_text = "\n".join([
        f"- {item['question']}: {item['answer']}"
        for item in context
        if "support@happyfeet.com" not in item["answer"].lower()
    ])
    if not context_text:
        context_text = "- No relevant FAQs were found."
    if history is None:
        history = []
    # Combine style tags from history and the current query
    combined_history = history + [{"role": "user", "content": query}]
    style_summary = get_style_summary_from_history(combined_history)

    # Update history with most recent style summary for continuity
    if style_summary:
        history.append({"role": "system", "content": f"Running style so far: {style_summary}"})

    prior_styles = summarize_style_keywords(extract_style_keywords(history))
    logger.debug(f"🧠 Prior style summary passed to prompt: {prior_styles}")

    is_followup = len(history) > 1
    if prior_styles:
        if is_followup:
            style_intro = (
                f"Building on your style so far—I'm picking up {prior_styles}. "
                "Let’s layer in your latest inspiration.\n\n"
            )
        else:
            style_intro = (
                f"You’ve got some standout style going on—so far I’m picking up {prior_styles}. "
                "Now let’s add your latest inspiration to the mix.\n\n"
            )
    else:
        style_intro = ""

    product_keywords = [
        "recommend", "suggest", "what shoes", "match my style", "match my vibe",
        "pair well", "go with", "what should I wear", "fit my personality", "shoe ideas",
        "i’m inspired by", "i am inspired by", "i love", "my vibe is", "my style is"
    ]
    should_suggest = any(kw in query.lower() for kw in product_keywords)

    # Extract previous recommended product names from history
    previously_suggested = set()
    for msg in history:
        if msg.get("role") == "system" and msg.get("content", "").startswith("Recommended products:"):
            names = msg["content"].split("Recommended products:")[-1].split(",")
            previously_suggested.update(name.strip() for name in names if name.strip())

    new_matches = []
    if should_suggest:
        all_matches = match_products(query, products)
        for match in all_matches:
            if match["name"] not in previously_suggested:
                new_matches.append(match)

        if new_matches:
            product_text = "\n".join(
                [f"- [{p['name']}]({p['link']}): {p['description']}" for p in new_matches]
            )
            product_prompt = (
                "If any of these feel like a fit, you can suggest one or more of the following HappyFeet products.\n"
                "When doing so, be sure to include the clickable Markdown links provided:\n"
                f"{product_text}\n\n"
            )
            # Remove excessive leading/trailing whitespace and add a single newline
            product_prompt = product_prompt.strip() + "\n"
            # Append system message for new recommendations
            history.append({
                "role": "system",
                "content": "Recommended products: " + ", ".join([p["name"] for p in new_matches])
            })
            if previously_suggested:
                product_prompt = "You might also like some of the shoes we talked about earlier...\n\n" + product_prompt
        else:
            product_prompt = ""
    else:
        product_prompt = ""

    info_links = {
        "sizing": "https://happyfeet.com/sizing-guide",
        "return": "https://happyfeet.com/returns-center",
        "look book": "https://happyfeet.com/lookbook",
        "shipping": "https://happyfeet.com/shipping-info",
        "order": "https://happyfeet.com/shipping-info",
        "track": "https://happyfeet.com/shipping-info",
        "track order": "https://happyfeet.com/shipping-info",
        "track my order": "https://happyfeet.com/shipping-info",
        "order status": "https://happyfeet.com/shipping-info"
    }
    matched_links = []
    lower_query = query.lower()
    seen_links = set()
    # Match keywords in the user query with predefined informational links (e.g., sizing, returns, shipping).
    # Ensures relevant links are included once in the response, avoiding duplicates.
    for keyword, link in info_links.items():
        if keyword in lower_query:
            if link not in seen_links:
                # Assign a friendly title based on the type of link
                if "sizing" in keyword:
                    title = "Sizing Guide"
                elif "return" in keyword:
                    title = "Returns Center"
                elif "look" in keyword:
                    title = "Look Book"
                else:
                    title = "Shipping Info"
                matched_links.append(f"- [{title}]({link})")
                seen_links.add(link)

    embedded_link = None
    
    if matched_links:
        embedded_link = matched_links[0].replace("- ", "")
        info_prompt = f"You must include the following link naturally in your response: {embedded_link}\n\n"
    else:
        info_prompt = ""

    history_text = ""
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"Customer: {content}\n"
            else:
                history_text += f"Frankie: {content}\n"

    prompt_parts = [
        "You are Frankie, the conversational AI agent for HappyFeet—a fun-loving, fashion-forward shoe brand known for making people smile with every step.\n",
        "Your voice is upbeat, empathetic, and infused with playful personality. Think of yourself as a knowledgeable and friendly team member who walks customers through anything they need—one confident stride at a time.\n\n",
        f"A customer just asked:\n\"{query}\"\n\n",
        "Use the following FAQs as your only source of truth. Do not make up answers. Rephrase or combine relevant FAQ content to craft a response that feels warm, helpful, and human—never robotic.\n",
        "Weave in subtle foot, step, or shoe-related wordplay when it feels natural—phrases like 'getting you back on track,' 'taking the next step,' or 'let’s lace up a solution together' add personality without overdoing it.\n\n",
        "If the question is about style or recommendations, get creative and expressive—like a stylish best friend who knows how to help someone put their best foot forward.\n",
        "If the customer's request includes a personal style or cultural reference (like 90s hip hop, retro fashion, or beach vibes), follow this 4-part structure:\n"
        "1. Open with a themed, energetic greeting tied to the all the styles the user provided."
        "2. Provide 2–3 specific product recommendations as a bulleted list, using the exact Markdown link format provided. Each item should have a fun, brand-aligned one-sentence description.\n"
        "3. End with a warm, playful call to action inviting more personal style details or a fun closing line.\n\n",
        "If the FAQs don’t fully answer the question—or if it feels urgent, emotional, or complex—always offer to hand off to one of our fabulous HappyFeet humans. ",
        "Do not initiate the handoff unless the customer explicitly asks for it or seems confused, frustrated, or unable to proceed. ",
        "Make the offer feel caring, optional, and brand-aligned. Never reference the FAQs. ",
        "Speak as if the transfer is already underway (e.g., 'I'm looping in one of our fabulous humans now'). ",
        "Make the handoff feel real, warm, and immediate—even if it’s only a simulated step. Keep it upbeat, caring, and always in-brand: helpful, kind, and full of HappyFeet spirit.\n\n",
        f"FAQs:\n{context_text}\n\n",
        style_intro,
        product_prompt,
        # info_prompt is now embedded in the next instruction, not as a separate part
        f"{info_prompt}Now, write a response that is playful, brand-aligned, clear, and concise. Avoid unnecessary line breaks or overly long phrasing—keep things crisp, warm, and easy to read. Use just enough spacing to separate sections, and avoid excessive white space.",
        "If there are any matched links, always include them at the end of your reply as clickable Markdown links. Do not include dashes. If you reference any products from the list above, you must use the exact Markdown format provided to ensure they appear as clickable links."
    ]

    # Instead of using a single user prompt, construct a detailed system message (prompt engineering)
    # and pass history messages plus the current user query.
    system_message = {
        "role": "system",
        "content": "".join(prompt_parts)
    }

    # Compose messages array: system, historical messages, then current user query
    messages = [system_message]
    # Add historical messages, preserving their roles and content
    for msg in history:
        # Accept only valid roles for OpenAI API ('user', 'assistant', 'system')
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Only include messages with content
        if content:
            messages.append({"role": role, "content": content})
    # Add the current user query as the last message
    messages.append({"role": "user", "content": query})

    logger.debug(f"🧾 OpenAI context (FAQ and product info) for query '{query}':\n{context_text}")
    logger.debug(f"📝 OpenAI messages array for query '{query}':\n{messages}")

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
        )
        answer = completion.choices[0].message.content.strip()
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        formatted_lines = []
        seen_bullet = False
        for line in lines:
            if line.startswith("- "):
                if not seen_bullet:
                    formatted_lines.append("")  # Ensure one line before bullet list
                    seen_bullet = True
                formatted_lines.append(line)
            else:
                if seen_bullet:
                    formatted_lines.append("")  # Add one line after bullet list
                    seen_bullet = False
                formatted_lines.append(line)
        answer = "\n".join(formatted_lines)
        if not isinstance(answer, str):
            answer = str(answer)
        # Do not append info_prompt again; it's already included in the prompt and model's reply.
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.post("/chat")
async def chat(request: Request):
    """
    FastAPI endpoint that receives chat requests, processes logic, and returns Frankie’s response.
    Receives a JSON payload with 'message' and optionally 'history'.
    Returns:
        dict: The response generated by Frankie and updated conversation history.
    """
    try:
        body = await request.json()
        logger.info(f"📦 Received chat request body: {body}")
        query = body.get("message")
        history = body.get("history", [])
        if not query:
            return {"response": "No message provided."}
        if needs_handoff(query):
            response = get_handoff_response()
            # Append relevant link to handoff message if not already included
            if "shipping" in query.lower() or "track" in query.lower() or "order" in query.lower():
                if "shipping-info" not in response.lower():
                    response += "\n\nHere's a helpful link to keep things moving: [Shipping Info](https://happyfeet.com/shipping-info)"
            return {"response": response}
        try:
            context = search_faq(data, query)
            if not context:
                logger.warning(f"⚠️ No FAQ context found for query: {query}")
        except Exception as e:
            logger.error(f"❌ Error in search_faq for query '{query}': {str(e)}")
            context = []
        logger.debug(f"🔍 FAQ context for query '{query}': {context}")
        answer = generate_response(query, context, history)
        return {"response": str(answer), "history": history}
    except Exception as e:
        logger.error(f"❌ Exception in chat endpoint: {str(e)}")
        return {"response": f"An error occurred: {str(e)}"}

@app.get("/")
def read_root():
    """
    Health check route that confirms backend server is running.
    Returns:
        dict: Status message.
    """
    return {"message": "HappyFeet backend is up!"}
    
if __name__ == "__main__":
    try:
        data = load_index()
        products = load_products()
    except FileNotFoundError as e:
        logger.error(f"Startup data loading failed: {e}")
        data = {"index": None, "faqs": []}
        products = []

