import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from backend.main import match_products, extract_style_keywords, needs_handoff, summarize_style_keywords, generate_response

def test_match_products_basic():
    query = "I'm into beach vibes and 90s style"
    products = [
        {"name": "Wave Runner", "tags": ["beach", "sand", "summer"], "link": "#", "description": "Great for the beach."},
        {"name": "Retro Kicks", "tags": ["90s", "hip hop", "throwback"], "link": "#", "description": "Perfect for 90s lovers."},
        {"name": "Modern Loafer", "tags": ["minimalist", "urban"], "link": "#", "description": "A clean, modern design."}
    ]
    matches = match_products(query, products)
    match_names = [p["name"] for p in matches]
    assert "Wave Runner" in match_names
    assert "Retro Kicks" in match_names
    assert "Modern Loafer" not in match_names

def test_extract_style_keywords_multiple_matches():
    history = [
        {"role": "user", "content": "I'm into 90s hip hop and beach fashion."},
        {"role": "user", "content": "Something coastal and vintage, please."}
    ]
    keywords = extract_style_keywords(history)
    assert "90s" in keywords
    assert "hip hop" in keywords
    assert "beach" in keywords
    assert "coastal" in keywords
    assert "vintage" in keywords


def test_summarize_style_keywords_basic():
    history = [
        {"role": "user", "content": "I'm into abstract art and modern design."},
        {"role": "user", "content": "I also like beach vibes and hip hop fashion."}
    ]
    summary = extract_style_keywords(history)
    assert isinstance(summary, list)
    assert "abstract" in summary
    assert "modern" in summary
    assert "beach" in summary
    assert "hip hop" in summary

def test_needs_handoff_true_case():
    user_input = "Can I speak with someone about a return?"
    assert needs_handoff(user_input) == True

def test_needs_handoff_false_case():
    user_input = "What shoes match a vintage style?"
    assert needs_handoff(user_input) == False


# Additional tests for coverage
def test_match_products_no_matches():
    query = "I'm into futuristic cyberpunk style"
    products = [
        {"name": "Classic Sneaker", "tags": ["retro", "vintage"], "link": "#", "description": "Old-school vibe."}
    ]
    matches = match_products(query, products)
    assert matches == []

def test_match_products_case_insensitivity():
    query = "I'm into BEACH vibes and 90S style"
    products = [
        {"name": "Wave Runner", "tags": ["beach", "sand"], "link": "#", "description": "Beachy vibes."},
        {"name": "Retro Kicks", "tags": ["90s", "hip hop"], "link": "#", "description": "Throwback style."}
    ]
    matches = match_products(query, products)
    match_names = [p["name"] for p in matches]
    assert "Wave Runner" in match_names
    assert "Retro Kicks" in match_names

def test_extract_style_keywords_empty_history():
    history = []
    keywords = extract_style_keywords(history)
    assert keywords == []

def test_extract_style_keywords_ignores_assistant():
    history = [
        {"role": "user", "content": "I like vintage and retro looks."},
        {"role": "assistant", "content": "Great! Let me suggest something."},
        {"role": "user", "content": "Also into coastal vibes."}
    ]
    keywords = extract_style_keywords(history)
    assert "vintage" in keywords
    assert "retro" in keywords
    assert "coastal" in keywords
    assert "suggest" not in keywords

def test_summarize_style_keywords_empty():
    history = []
    summary = extract_style_keywords(history)
    assert summary == []

def test_needs_handoff_variation():
    user_input = "Can I talk to your customer service rep?"
    assert needs_handoff(user_input) == True

def test_generate_response_basic():
    query = "What is your return policy?"
    context = [
        {
            "question": "What is your return policy?",
            "answer": "You can return any item within 30 days of purchase."
        }
    ]
    history = []
    response = generate_response(query, context, history)
    assert isinstance(response, str)
    assert "return" in response.lower()
