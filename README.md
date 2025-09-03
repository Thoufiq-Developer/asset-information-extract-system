Asset Information Extraction System:

This project is an AI-powered asset metadata extractor. It takes a product’s model number and classification name, searches the web for relevant information, and uses Google Gemini LLM to generate structured product data.

Features:

1.Accepts input as JSON with fields like model_number, asset_classification_name, and manufacturer.
 Uses SERP API for web search.

2.Uses Google Gemini for intelligent extraction and summarization.

3.Returns clean, structured JSON output with fields such as classification, manufacturer, model number, product line, and summary.

4.Includes retry and fallback mechanism if data is incomplete.

Local Setup:

1.Create and activate a virtual environment.

2.Install dependencies using pip install -r requirements.txt.

3.Set environment variables for GEMINI_API_KEY and SERP_API_KEY.

4.Run the Flask server locally.
