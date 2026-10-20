"""Greedy native256 generation with real frozen-encoder64-token rollover."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import torch
from models.token_memory import Model
from models.token_memory.data import load_encoder,encode_history
from models.token_memory.runtime import load,worker_lock,pin_cpu
from models.token_memory.training import FORMAT

@torch.inference_mode()
def generate(model,codec,history,max_new_tokens=128):
    if max_new_tokens < 1:
        raise ValueError('Positive generation length required')
    history=torch.as_tensor(history,dtype=torch.long,device=codec.device)
    if history.ndim != 1:
        raise ValueError('One history required')
    old=len(history)//64*64
    memory,valid=encode_history(codec,[history[:old]])
    cache=model.cache(1,old//64+(len(history)-old+max_new_tokens+63)//64)
    model.append_codes(cache,memory,valid)
    raw=torch.cat((torch.ones(1,dtype=torch.long,device=codec.device),history[old:]))[None]
    scores,_=model.feed_tokens(cache,raw)
    def encode(tokens):
        codes,_=codec.model(tokens,torch.ones_like(tokens,dtype=torch.bool))
        return codes[:,0].float()
    output=[]
    for index in range(max_new_tokens):
        token=scores[:,-1].argmax(-1)
        output.append(int(token))
        if int(token)==2:break
        if index+1<max_new_tokens:scores,_=model.advance(cache,token,encode)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--prompt',required=True,help='Serialized text, for example User: ... followed by Assistant:')
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();pin_cpu();torch.set_num_threads(1)
    with worker_lock():
        state=load(args.run)
        if state.get('format')!=FORMAT:raise ValueError('Use archived source for previous architectures')
        model=Model().to(args.device);model.load_state_dict(state['model']);model.eval()
        codec=load_encoder(args.device)
        ids=codec.tokenizer.encode(args.prompt).ids
        output=generate(model,codec,ids,args.max_new_tokens)
        print(codec.tokenizer.decode(output))

if __name__=='__main__':main()
