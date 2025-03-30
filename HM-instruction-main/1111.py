from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from PIL import Image
import io
import base64
import os 
torch.manual_seed(1234)
from ipdb import set_trace
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Load tokenizer and model10.24.5.63'
tokenizer = AutoTokenizer.from_pretrained("/home/smbu/BBY/Qwen-VL-master/Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/smbu/BBY/Qwen-VL-master/Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()



@app.route('/process', methods=['POST'])

def process():
    
    try:
        text = request.form['text']
        history = request.form.get('history')
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            
                
            query = tokenizer.from_list_format([
                {'image': image_path},
                {'text': text},
            ])
        else:
            query = text
        
        if isinstance(history, str) :
            history = eval(history) 
        else:
            history = None

        response, history = model.chat(tokenizer, query=query, history=history)
        image = tokenizer.draw_bbox_on_latest_picture(response, history)
        print(response)
        if image:
            image.save('output_chat1.jpg')
        else:
            print("no box")
        return jsonify({"response": response, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    #app.run(host='127.0.0.2', port=8001)
