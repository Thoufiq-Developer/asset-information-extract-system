import os
import json
import time
import requests
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asset_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AssetInput:
    model_number: str
    asset_classification_name: str
    manufacturer: Optional[str] = ""
    asset_classification_guid2: Optional[str] = ""

@dataclass
class AssetOutput:
    asset_classification: str
    manufacturer: str
    model_number: str
    product_line: str
    summary: str

class SerpAPIService:
    """Service for web search using SERP API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search.json"
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform web search and return results"""
        try:
            logger.info(f"Searching for: {query}")
            
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google',
                'num': num_results,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            if 'organic_results' in data:
                for result in data['organic_results']:
                    results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'displayed_link': result.get('displayed_link', '')
                    })
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SERP API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            return []

class GeminiService:
    """Service for processing data with Gemini AI"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def extract_asset_info(self, search_results: List[Dict], asset_input: AssetInput) -> Dict:
        """Extract structured asset information using Gemini"""
        try:
            # Prepare context from search results
            search_context = self._format_search_results(search_results)
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(asset_input, search_context)
            
            logger.info("Processing with Gemini AI...")
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            json_text = self._extract_json_from_response(response.text)
            result = json.loads(json_text)
            
            logger.info("Gemini processing completed successfully")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            raise Exception("Invalid JSON response from Gemini")
        except Exception as e:
            logger.error(f"Gemini processing failed: {e}")
            raise
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results for prompt context"""
        if not search_results:
            return "No search results found."
        
        formatted = "SEARCH RESULTS:\n\n"
        for i, result in enumerate(search_results, 1):
            formatted += f"Result {i}:\n"
            formatted += f"Title: {result.get('title', 'N/A')}\n"
            formatted += f"Content: {result.get('snippet', 'N/A')}\n"
            formatted += f"Source: {result.get('displayed_link', 'N/A')}\n\n"
        
        return formatted
    
    def _create_extraction_prompt(self, asset_input: AssetInput, search_context: str) -> str:
        """Create the extraction prompt for Gemini"""
        return f"""
You are an AI assistant specialized in extracting structured product information from web search results.

TASK: Extract asset information and return it as a valid JSON object.

INPUT ASSET INFORMATION:
- Model Number: {asset_input.model_number}
- Asset Classification: {asset_input.asset_classification_name}
- Manufacturer: {asset_input.manufacturer or 'Not specified'}
- GUID: {asset_input.asset_classification_guid2 or 'Not specified'}

{search_context}

INSTRUCTIONS:
1. Analyze the search results to extract information about the specified asset
2. Return ONLY a valid JSON object with the following structure:
{{
    "asset_classification": "string - refined/corrected classification based on search results",
    "manufacturer": "string - manufacturer name from search results",
    "model_number": "string - original model number: {asset_input.model_number}",
    "product_line": "string - product line/series name if found",
    "summary": "string - 2-3 sentence summary of the asset based on search results"
}}

REQUIREMENTS:
- ALL fields must be filled with meaningful information
- If manufacturer is not found in search results, return empty string ""
- If product line is not found, return empty string ""
- Summary must be informative and based on search results
- Return ONLY the JSON object, no additional text or explanations
- Ensure the JSON is properly formatted and valid

JSON OUTPUT:
"""
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from Gemini response text"""
        # Find JSON in the response
        text = response_text.strip()
        
        # Look for JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        
        # If no clear JSON boundaries found, try the whole response
        return text

class AssetExtractionSystem:
    """Main system for asset information extraction"""
    
    MAX_RETRIES = 5
    RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, gemini_api_key: str, serp_api_key: str):
        self.serp_service = SerpAPIService(serp_api_key)
        self.gemini_service = GeminiService(gemini_api_key)
    
    def process_asset(self, input_data: Dict) -> Dict:
        """Main processing function"""
        try:
            # Validate and parse input
            asset_input = AssetInput(**input_data)
            logger.info(f"Processing asset: {asset_input.model_number}")
            
            # Build search query
            search_query = self._build_search_query(asset_input)
            
            # Perform web search
            search_results = self.serp_service.search(search_query)
            
            if not search_results:
                logger.warning("No search results found, using fallback")
                return self._get_fallback_response(asset_input)
            
            # Extract information with retry logic
            result = self._extract_with_retry(search_results, asset_input)
            
            # Save results to JSON file
            self._save_to_file(result, asset_input.model_number)
            
            return result
            
        except Exception as e:
            logger.error(f"System error: {e}")
            return self._get_fallback_response(AssetInput(**input_data))
    
    def _build_search_query(self, asset_input: AssetInput) -> str:
        """Build optimized search query"""
        query_parts = []
        
        # Add manufacturer if provided
        if asset_input.manufacturer:
            query_parts.append(asset_input.manufacturer)
        
        # Add model number
        query_parts.append(asset_input.model_number)
        
        # Add classification terms (remove parentheses and common words)
        classification_clean = asset_input.asset_classification_name.replace('(', '').replace(')', '')
        classification_terms = [term for term in classification_clean.split() 
                              if len(term) > 2 and term.lower() not in ['the', 'and', 'or']]
        query_parts.extend(classification_terms)
        
        # Add relevant search terms
        query_parts.extend(['specifications', 'manual'])
        
        query = ' '.join(query_parts)
        logger.info(f"Built search query: {query}")
        return query
    
    def _extract_with_retry(self, search_results: List[Dict], asset_input: AssetInput) -> Dict:
        """Extract information with retry logic"""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"Extraction attempt {attempt}/{self.MAX_RETRIES}")
                
                result = self.gemini_service.extract_asset_info(search_results, asset_input)
                
                if self._is_complete_response(result):
                    logger.info(f"Extraction successful on attempt {attempt}")
                    return result
                
                logger.warning(f"Incomplete response on attempt {attempt}")
                
                if attempt < self.MAX_RETRIES:
                    logger.info(f"Retrying in {self.RETRY_DELAY} seconds...")
                    time.sleep(self.RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                
                if attempt == self.MAX_RETRIES:
                    logger.warning("Max retries exceeded, using fallback")
                    break
                
                time.sleep(self.RETRY_DELAY)
        
        return self._get_fallback_response(asset_input)
    
    def _is_complete_response(self, response: Dict) -> bool:
        """Check if response has all required fields with meaningful data"""
        required_fields = ['asset_classification', 'manufacturer', 'model_number', 'product_line', 'summary']
        
        for field in required_fields:
            if field not in response:
                return False
            
            value = str(response[field]).strip()
            if field in ['asset_classification', 'model_number', 'summary'] and len(value) == 0:
                return False
        
        return True
    
    def _get_fallback_response(self, asset_input: AssetInput) -> Dict:
        """Generate fallback response as specified"""
        logger.warning("Generating fallback response")
        return {
            "asset_classification": "Generator Emissions/UREA/DPF Systems",
            "manufacturer": "",
            "model_number": asset_input.model_number,
            "product_line": "",
            "summary": ""
        }
    
    def _save_to_file(self, data: Dict, model_number: str) -> None:
        """Save results to JSON file"""
        filename = f"asset_extraction_{model_number}_{int(time.time())}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

def main():
    """Main function for testing the system"""
    
    # API Keys (set these as environment variables)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    SERP_API_KEY = os.getenv('SERP_API_KEY')
    
    if not GEMINI_API_KEY or not SERP_API_KEY:
        logger.error("Please set GEMINI_API_KEY and SERP_API_KEY environment variables")
        return
    
    # Initialize system
    system = AssetExtractionSystem(GEMINI_API_KEY, SERP_API_KEY)
    
    # Example input
    test_input = {
        "model_number": "MRN85HD",
        "asset_classification_name": "Generator (Marine)",
        "manufacturer": "",
        "asset_classification_guid2": ""
    }
    
    logger.info("=" * 50)
    logger.info("STARTING ASSET EXTRACTION SYSTEM")
    logger.info("=" * 50)
    
    # Process the asset
    result = system.process_asset(test_input)
    
    # Display results
    logger.info("=" * 50)
    logger.info("EXTRACTION RESULTS:")
    logger.info("=" * 50)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

# Flask Web API Version
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize extraction system (you'll need to set these environment variables)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')

if GEMINI_API_KEY and SERP_API_KEY:
    extraction_system = AssetExtractionSystem(GEMINI_API_KEY, SERP_API_KEY)
else:
    extraction_system = None
    logger.warning("API keys not found. Set GEMINI_API_KEY and SERP_API_KEY environment variables.")

@app.route('/extract', methods=['POST'])
def extract_asset():
    """API endpoint for asset extraction"""
    try:
        if not extraction_system:
            return jsonify({
                "error": "System not initialized. Please check API keys.",
                "status": "error"
            }), 500
        
        # Get JSON data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400
        
        # Validate required fields
        if 'model_number' not in input_data or 'asset_classification_name' not in input_data:
            return jsonify({
                "error": "model_number and asset_classification_name are required",
                "status": "error"
            }), 400
        
        # Process the asset
        result = extraction_system.process_asset(input_data)
        
        return jsonify({
            "data": result,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "system_ready": extraction_system is not None
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

# Requirements.txt content (save this as requirements.txt):
"""
google-generativeai>=0.3.0
requests>=2.31.0
flask>=2.3.0
flask-cors>=4.0.0
python-dotenv>=1.0.0
"""

# Example usage script
def run_example():
    """Example of how to use the system"""
    
    # Set your API keys here or as environment variables
    os.environ['GEMINI_API_KEY'] = 'your_gemini_api_key_here'
    os.environ['SERP_API_KEY'] = 'your_serp_api_key_here'
    
    # Test cases
    test_cases = [
        {
            "model_number": "MRN85HD",
            "asset_classification_name": "Generator (Marine)",
            "manufacturer": "",
            "asset_classification_guid2": ""
        },
        {
            "model_number": "CAT3516",
            "asset_classification_name": "Generator (Industrial)",
            "manufacturer": "Caterpillar",
            "asset_classification_guid2": ""
        },
        {
            "model_number": "GP7500E",
            "asset_classification_name": "Generator (Portable)",
            "manufacturer": "Generac",
            "asset_classification_guid2": ""
        }
    ]
    
    system = AssetExtractionSystem(
        os.getenv('GEMINI_API_KEY'),
        os.getenv('SERP_API_KEY')
    )
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['model_number']}")
        print(f"{'='*60}")
        
        result = system.process_asset(test_case)
        print(json.dumps(result, indent=2))
        
        # Add delay between requests to respect rate limits
        if i < len(test_cases):
            time.sleep(2)

