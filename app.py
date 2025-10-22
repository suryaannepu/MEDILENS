import os
import io
import base64
from flask import Flask, render_template, request, jsonify, Response
import google.generativeai as genai
from groq import Groq
from PIL import Image
import logging
import json
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max file size

# Configure APIs (Replace with your actual API keys)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize models
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

def optimize_image(image_data):
    """Optimize image for faster processing"""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize for faster processing
        image.thumbnail((512, 512))
        
        # Convert back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
        raise e

def analyze_medical_image(image_data):
    """Analyze medical image using Gemini with optimized prompt"""
    try:
        image = Image.open(io.BytesIO(image_data))
        prompt = """
        QUICK MEDICAL SCAN - Provide immediate findings in this exact format:
        
        ABNORMALITIES: [List key visible abnormalities, one per line with •]
        SUSPECTED_CONDITIONS: [List possible conditions, one per line with •]
        URGENCY_LEVEL: [Low/Medium/High]
        RECOMMENDATIONS: [List critical recommendations, one per line with •]
        
        Be concise and focus only on critical findings. Maximum 200 words.
        """

        response = gemini_model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {str(e)}")
        raise e

def generate_detailed_report_stream(findings):
    """Generate detailed report using Groq with streaming"""
    try:
        system_prompt = """You are a medical AI assistant. Generate a structured medical report with these exact sections:
        
        ## CLINICAL FINDINGS
        [Expand on abnormalities found]
        
        ## DIFFERENTIAL DIAGNOSIS  
        [List possible conditions with likelihood]
        
        ## RECOMMENDED ACTIONS
        [Specific medical recommendations]
        
        ## FOLLOW-UP SUGGESTIONS
        [Next steps and monitoring]
        
        Keep each section concise. Start with CLINICAL FINDINGS immediately."""
        
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create detailed medical report from these initial findings:\n{findings}"}
            ],
            stream=True,
            max_tokens=1000,
            temperature=0.3
        )
        
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Error in Groq streaming: {str(e)}")
        yield f"Error generating detailed report: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Read and optimize file data
            file_data = file.read()
            optimized_data = optimize_image(file_data)
            filename = file.filename
            
            # Check if it's an image file
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
            if '.' in filename and filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
            
            # Analyze the image with Gemini
            logger.info(f"Analyzing image: {filename}")
            initial_findings = analyze_medical_image(optimized_data)
            
            # Convert image to base64 for display
            image_base64 = base64.b64encode(optimized_data).decode('utf-8')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'initial_findings': initial_findings,
                'image_data': f"data:image/jpeg;base64,{image_base64}"
            })
            
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/stream_report')
def stream_report():
    findings = request.args.get('findings', '')
    
    def generate():
        try:
            yield "data: Starting detailed analysis...\n\n"
            for chunk in generate_detailed_report_stream(findings):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 8MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)