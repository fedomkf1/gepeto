import numpy as np
import random
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
no_deprecation_warning = True

device = torch.device("cpu")
control_code = "_gepeto_"
batch_size = 1

def dd(algo):
    print("  ")
    print("  ")
    print("---------------")
    print(algo)
    print("---------------")
    print("  ")
    print("  ")

# Load a trained model and vocabulary that you have fine-tuned
output_dir = './model_gpt_ibai'
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

# coloca el modelo en modo de evaluación
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return 'Bienvenido a GePeTo!'

@app.route('/chat', methods=['POST'])
def chat():

    question = request.form['question']
    dd(question)

    # prompt = "<|startoftext|>" + "<|"+control_code+"|>" + "¿ "+question+" ?"
    prompt = "<|startoftext|>" +"¿"+question+"?"

    generated = torch.tensor(tokenizer.encode(question)).unsqueeze(0)
    generated = generated.to(device)
    # print(generated)

    sample_outputs = model.generate(
      generated, 
      do_sample=True,   
      top_k = 50, 
      max_length = 300,
      top_p = 0.95, 
      num_return_sequences = 1
    )

    response = tokenizer.decode(sample_outputs[0], skip_special_tokens = True)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
