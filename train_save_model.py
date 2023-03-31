import os
import time
import datetime

import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup

no_deprecation_warning=True

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

if torch.cuda.is_available():
    print("Usar GPU")
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")
    batch_size = 3

else:
    print("usar CPU")
    device = torch.device("cpu")
    batch_size = 1

# Load the GPT tokenizer.
tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-2-spanish", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = AutoModelForCausalLM.from_pretrained("flax-community/gpt-2-spanish")

control_code = "ibai"

special_tokens_dict = {
        "additional_special_tokens": ['f"<|{control_code}|>"'],
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
unk_tok_emb = model.transformer.wte.weight.data[tokenizer.unk_token_id, :]
for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i+1), :] = unk_tok_emb

class GPT2Dataset(Dataset):

  def __init__(self, control_code, tokenizer, archivo_texto = 'all.txt', max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    print('loading text...')
    sentences = open(archivo_texto, 'r', encoding="utf-8").read().lower().split('\n')
    print('qty:',len(sentences))

    for row in tqdm(sentences):
      encodings_dict = tokenizer('<|startoftext|>'+ f"<|{control_code}|>" + row + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 

dataset = GPT2Dataset(control_code, tokenizer, archivo_texto="ibai_textos.txt", max_length=768)

# Split into training and validation sets
train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler = RandomSampler(train_dataset), # Select batches randomly
    batch_size = batch_size # Trains with this batch size.
)

# some parameters to train
epochs = 1
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
# this produces sample output every x steps
sample_every = 500
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(
    model.parameters(),
    lr = learning_rate,
    eps = epsilon
)
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = warmup_steps, 
    num_training_steps = total_steps
)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))  







total_t0 = time.time()

model = model.to(device)

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  
            b_input_ids,
            labels=b_labels, 
            attention_mask = b_masks,
            token_type_ids=None
        )

        loss = outputs[0]
        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()

            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    
    t0 = time.time()

    total_eval_loss = 0
    nb_eval_steps = 0

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
#Average training loss: 0.28
#Training epoch took: 1:23:32 mode_save2 va bastante bien, solo 1 epoch 

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
output_dir = './model_gpt_ibai/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

