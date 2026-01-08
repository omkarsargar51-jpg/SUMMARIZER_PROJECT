from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the summarization pipeline (uses the DistilBART model by default)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route('/')
def index():            
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    input_text = data.get('text', '')
    
    if len(input_text) < 50:
        return jsonify({'summary': "Text is too short to summarize!"})

    # ADJUSTED PARAMETERS FOR LONGER SUMMARY:
    # max_length: The maximum tokens in the summary
    # min_length: Forces the AI to keep writing until it hits this limit
    # length_penalty: Higher values (e.g., 2.0) encourage longer sequences
    summary = summarizer(
        input_text, 
        max_length=250,    # Increased from 130
        min_length=80,     # Increased from 30
        length_penalty=2.0, 
        repetition_penalty=1.2,
        do_sample=False
    )
    
    return jsonify({'summary': summary[0]['summary_text']})

if __name__ == '__main__':
    # use_reloader=False stops the Windows Socket error
    app.run(debug=True, use_reloader=False)