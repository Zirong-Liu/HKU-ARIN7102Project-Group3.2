from flask import Flask, render_template, request, jsonify
import chatbot

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/send', methods=['POST'])
def predict():
    input_data = request.get_json()
    query_message = input_data.get('message')
    ans = chatbot.start_chat(query_message)
    return jsonify({'answer': ans})  # For now, just returning the user message as the bot's answer

if __name__ == '__main__':
    app.run(debug=True)
