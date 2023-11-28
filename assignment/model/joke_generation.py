import tiktoken
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from .gpt import Config, GPT

from assignment import ROOT_DIR

model_1, model_2 = None, None
enc = tiktoken.get_encoding("gpt2")
tokenizer = None

models = {
    'gpt': model_1,
    'gpt2': model_2,
}

device = 'cpu'


def load_gpt_model():
    global model_1, enc
    load_model_path = f'{ROOT_DIR}/assignment/model/model_1'
    max_seq_len = 50

    config = Config(enc.n_vocab, max_seq_len)
    model_1 = GPT(config).to(device)
    model_1.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))
    model_1.eval()
    models['gpt'] = model_1


def load_gpt2_model():
    global tokenizer, model_2
    load_model_path = f'{ROOT_DIR}/assignment/model/try_this.h'
    config_gpt2_pre = GPT2Config.from_pretrained('gpt2')
    config_gpt2_pre.do_sample = config_gpt2_pre.task_specific_params['text-generation']['do_sample']
    config_gpt2_pre.max_length = config_gpt2_pre.task_specific_params['text-generation']['max_length']

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_2 = GPT2LMHeadModel.from_pretrained(load_model_path, config=config_gpt2_pre)
    models['gpt2'] = model_2


def generate_joke(starting_string, model_choice):
    '''
    Something smart happens here.
    :param sentences:
    :return:
    '''

    if model_choice == 'gpt':
        start = torch.tensor([enc.encode(starting_string)], dtype=torch.long, device=device)
        output = models[model_choice].generate(xs=start, max_new_tokens=30, do_sample=True)[0].tolist()
        try:
            end_index = output.index(enc.eot_token)
            output = enc.decode(output[:end_index])
        except:
            output = "Please try again!"
        return output
    else:
        encoded_input = tokenizer.encode(starting_string, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(device)
        models[model_choice].to(device)
        output = models[model_choice].generate(encoded_input)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
