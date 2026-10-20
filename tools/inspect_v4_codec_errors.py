"""Locate frozen-codec reconstruction errors in the grounding diagnostic."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from models.token_memory.runtime import pin_cpu,write_json,digest
from prepare_chat_corpus import memory_episode


def main():
    pin_cpu()
    import torch
    from models.token_memory.data import load_encoder
    from models.position_compressor.codec import model_from_config
    from chat_data.unified_format import encode_prompt,encode_dialogue,with_mode
    torch.set_num_threads(1)
    out=ROOT/'dump/unified-v4-grounding-diagnosis'
    contrasts=json.loads((out/'contrasts.json').read_text())
    codec=load_encoder('cuda');path=ROOT/'dump/pretrained-latent64-fced6bd/model.pt'
    assert digest(path)=='abc271f5d1de3f8168ed07a5c0351e71def86c849745791d230f0b02efad91f6'
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=model_from_config(state['protocol']['model']).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model']);errors=[];fact_tokens=0;fact_errors=0
    with torch.inference_mode():
        for r in contrasts:
            original,_=memory_episode(r['index'],r['names'])
            filler=dict(role='user',content=original[2]['content'].split('On day 21,')[0].strip())
            messages=[original[0],original[1],filler,original[3],dict(role='user',content=r['question'])]
            ids=encode_prompt(messages,codec.tokenizer,thinking=False)
            _,_,spans=encode_dialogue(with_mode(messages),codec.tokenizer)
            lo,hi=spans[1]['body_start'],spans[1]['body_end'];begin=lo//64*64;end=(hi+63)//64*64
            tokens=torch.tensor(ids[begin:end],device='cuda').reshape(-1,64)
            z,lengths=codec.model(tokens,torch.ones_like(tokens,dtype=torch.bool))
            decoded=model.generate(z,lengths)[:,:64]
            fact_tokens+=hi-lo
            for chunk,position in (decoded!=tokens).nonzero().tolist():
                absolute=begin+chunk*64+position;is_fact=lo<=absolute<hi
                fact_errors+=is_fact
                expected=int(tokens[chunk,position]);actual=int(decoded[chunk,position])
                errors.append(dict(names=r['names'],wording=r['wording'],index=r['index'],position=absolute,
                    in_fact=is_fact,expected_id=expected,actual_id=actual,
                    expected=codec.tokenizer.decode([expected],skip_special_tokens=False),
                    actual=codec.tokenizer.decode([actual],skip_special_tokens=False)))
    report=dict(fact_tokens=fact_tokens,fact_token_errors=fact_errors,total_chunk_errors=len(errors),errors=errors)
    write_json(out/'codec_errors.json',report)
    print(json.dumps(dict(fact_tokens=fact_tokens,fact_token_errors=fact_errors,total_chunk_errors=len(errors),example=errors[:2]),indent=2))


if __name__=='__main__':main()
