import tiktoken
import torch

from .gpt import Config, GPT

from assignment import ROOT_DIR

model_1, model_2 = None, None
enc = None

models = {
    'gpt': model_1,
    'gpt2': model_2,
}

device = 'cpu'


def load_gpt_model():
    global model_1, enc
    load_model_path = f'{ROOT_DIR}/assignment/model/model_1'
    enc = tiktoken.get_encoding("gpt2")
    tokens = 50257
    max_seq_len = 50

    config = Config(enc.n_vocab, max_seq_len)
    model_1 = GPT(config).to(device)
    model_1.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))
    model_1.eval()


def load_gpt2_model():
    global model_2
    load_model_path = f'{ROOT_DIR}/assignment/model_1'
    # TODO: write script that loads second model


def generate_joke(starting_string, model_choice):
    '''
    Something smart happens here.
    :param sentences:
    :return:
    '''

    start = torch.tensor([enc.encode(starting_string)], dtype=torch.long, device=device)

    output = model_choice[model_choice].generate(xs=start, max_new_tokens=30, do_sample=True)[0].tolist()
    output = output[:output.index(enc.eot_token)]
    return enc.decode(output)
