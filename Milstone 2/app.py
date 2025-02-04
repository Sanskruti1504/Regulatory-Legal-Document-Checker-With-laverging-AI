from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from groq import Groq
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import PyPDF2
import io
import re

app = Flask(__name__)
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)

def extract_youtube_id(url):
    """Extract YouTube video ID from URL."""
    pattern = r'(?:v=|\/videos\/|youtu\.be\/|\/v\/|\/e\/|watch\?v%3D|watch\?feature=player_embedded&v=)([^#\&\?]*).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_transcript(video_id):
    """Get transcript of a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript_list])
    except Exception as e:
        return f"Error getting transcript: {str(e)}"

def extract_text_from_pdf(pdf_url):
    """Extract text from a PDF URL."""
    try:
        response = requests.get(pdf_url)
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_web_content(url):
    """Extract text content from a web page."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        return f"Error extracting web content: {str(e)}"

def summarize_content(content):
    """Generate summary using Groq API."""
    try:
        prompt = f"""Please analyze the following content and provide:
        1. A concise summary of key points
        2. Main insights and takeaways
        3. Important concepts discussed

        Content:
        {content[:4000]}  # Limiting content length for API
        """
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an expert content analyzer and summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Determine content type and extract accordingly
    if 'youtube.com' in url or 'youtu.be' in url:
        video_id = extract_youtube_id(url)
        content = get_youtube_transcript(video_id)
    elif url.lower().endswith('.pdf'):
        content = extract_text_from_pdf(url)
    else:
        content = extract_web_content(url)
    
    summary = summarize_content(content)
    return jsonify({'summary': summary})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')
    
    if not question or not context:
        return jsonify({'error': 'Question or context missing'}), 400
    
    try:
        prompt = f"""Based on the following content, please answer the question:

        Content: {context}

        Question: {question}
        """
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an expert at answering questions about content accurately and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return jsonify({'answer': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001)