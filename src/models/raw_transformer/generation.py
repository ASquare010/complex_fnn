import torch


@torch.no_grad()
def generate(model,codec,prompt,max_new_tokens):
    was_training=model.training;model.eval()
    boundary=len(prompt)//512*512
    history=list(prompt[:boundary]);active=list(prompt[boundary:]);result=[]
    try:
        for _ in range(max_new_tokens):
            h=torch.tensor([history[-2048:]],device=codec.device,dtype=torch.long)
            x=torch.tensor([[1]+active],device=codec.device,dtype=torch.long)
            logits=model(h,torch.ones_like(h,dtype=torch.bool),x,torch.ones_like(x,dtype=torch.bool))
            token=int(logits[0,-1].argmax());active.append(token);result.append(token)
            if len(active)==512:history+=active;active=[]
    finally:model.train(was_training)
    return result
