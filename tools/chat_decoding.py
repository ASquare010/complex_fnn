"""Inference-only sampling; does not modify any trained weights."""
import torch
from models.token_memory.data import encode_history

@torch.no_grad()
def generate_reply(model,codec,text,max_new_tokens=96,temperature=.7,top_p=.9,repetition_penalty=1.15,seed=47):
 if temperature<0 or not 0<top_p<=1 or repetition_penalty<1:raise ValueError('Invalid sampling parameters')
 was=model.training;model.eval();rng=torch.Generator(device=codec.device).manual_seed(seed)
 ids=codec.tokenizer.encode(text).ids;boundary=len(ids)//512*512
 mem,mask=encode_history(codec,[torch.tensor(ids[:boundary],dtype=torch.long)])
 active=ids[boundary:];answer=[]
 try:
  for _ in range(max_new_tokens):
   x=torch.tensor([[1]+active],device=codec.device)
   scores=model(mem,mask,x)[0,-1].float().clone()
   # Suppress training-only pad and block-start markers from user-visible text.
   scores[0:2]=float('-inf')
   if temperature:
    if answer:
     used=torch.tensor(sorted(set(answer)),device=scores.device)
     scores[used]=torch.where(scores[used]<0,scores[used]*repetition_penalty,scores[used]/repetition_penalty)
    # Avoid a repeated generated4-gram; prefixes from the user's text are not banned.
    if len(answer)>=3:
     prefix=answer[-3:]
     for i in range(len(answer)-3):
      if answer[i:i+3]==prefix:scores[answer[i+3]]=float('-inf')
    scores/=temperature
    sorted_scores,order=scores.sort(descending=True)
    cumulative=sorted_scores.softmax(-1).cumsum(-1)
    remove=cumulative>top_p;remove[1:]=remove[:-1].clone();remove[0]=False
    sorted_scores[remove]=float('-inf')
    token=int(order[torch.multinomial(sorted_scores.softmax(-1),1,generator=rng)])
   else:token=int(scores.argmax())
   if token==2:break
   answer.append(token);active.append(token)
   if len(active)==512:
    z,v=encode_history(codec,[torch.tensor(active,dtype=torch.long)])
    mem=torch.cat((mem,z),1);mask=torch.cat((mask,v),1);active=[]
 finally:model.train(was)
 return codec.tokenizer.decode(answer)
