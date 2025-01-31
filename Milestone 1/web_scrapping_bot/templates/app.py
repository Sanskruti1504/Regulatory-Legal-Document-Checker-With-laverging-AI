from flask import Flask, render_template, request, jsonify, send_file
from scraper import ScrapingChatbot
import asyncio
import os
import logging
from functools import partial
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable nested event loops
nest_asyncio.apply()

app = Flask(__name__)

# Initialize chatbot
chatbot = ScrapingChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Handle URL scraping
        if message.startswith(('http://', 'https://')):
            return await handle_url_scraping(message)
        
        # Handle normal chat queries
        return await handle_chat_query(message)
        
    except Exception as e:
        logger.error(f'Unexpected error in chat endpoint: {str(e)}')
        return jsonify({'error': str(e)}), 500

async def handle_url_scraping(url):
    try:
        content = await chatbot.scrape_url(url)
        if content:
            response = {
                'title': content.get('title', ''),
                'meta_description': content.get('meta_description', ''),
                'headings': content.get('headings', {}),
                'paragraphs': content.get('paragraphs', []),
                'images': content.get('images', []),
                'links': content.get('links', []),
                'lists': content.get('lists', {'ordered': [], 'unordered': []}),
                'tables': content.get('tables', []),
                'contact_info': content.get('contact_info', {'emails': [], 'phones': [], 'addresses': []}),
                'social_links': content.get('social_links', [])
            }
            return jsonify({'response': response, 'message': 'Content scraped successfully!'}), 200
        else:
            logger.error('No content found in scraping response')
            return jsonify({'error': 'No content found on the webpage'}), 404
    except Exception as e:
        logger.error(f'Error during scraping: {str(e)}')
        return jsonify({'error': f'Failed to scrape webpage: {str(e)}'}), 500

async def handle_chat_query(message):
    try:
        answer = chatbot.chat_with_groq(message)
        return jsonify({'response': answer}), 200
    except Exception as e:
        logger.error(f'Error during chat: {str(e)}')
        return jsonify({'error': f'Failed to get answer: {str(e)}'}), 500

@app.route('/export', methods=['POST'])
async def export_data():
    try:
        if not chatbot.current_content:
            return jsonify({'error': 'No content to export'}), 400

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        format_type = data.get('format', 'csv')
        if format_type not in ['csv', 'excel']:
            return jsonify({'error': 'Invalid format type'}), 400

        filename = chatbot.export_data([chatbot.current_content], format_type)
        response = send_file(filename, as_attachment=True)
        os.remove(filename)  # Clean up file after sending
        return response
    except Exception as e:
        logger.error(f'Unexpected error in export endpoint: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.environ['GROQ_API_KEY'] = 'your_api_key_here'
    app.run(debug=True, port=5006)
