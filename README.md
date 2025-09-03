Asset Information Extraction System:
This project is an AI-powered asset metadata extractor. It takes a product’s model number and classification name, searches the web for relevant information, and uses Google Gemini LLM to generate structured product data.

Features:

Accepts input as JSON with fields like model_number, asset_classification_name, and manufacturer.
Uses SERP API for web search.

Uses Google Gemini for intelligent extraction and summarization.

Returns clean, structured JSON output with fields such as classification, manufacturer, model number, product line, and summary.

Includes retry and fallback mechanism if data is incomplete.

Local Setup:
Create and activate a virtual environment.
Install dependencies using pip install -r requirements.txt.
Set environment variables for GEMINI_API_KEY and SERP_API_KEY.
Run the Flask server locally.
