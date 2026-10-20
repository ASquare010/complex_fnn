"""Actual mixed loss, generic raw1/64 coordinates and serialized Adam."""
import copy
import io
import unittest
import torch
import test_token_memory_canonical as fixture
from models.token_memory.wordbits import unpack_values
from models.token_memory import objective


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.Checks.setUpClass()
        cls.fixture = fixture.Checks()

    def packet(self, per=64):
        values = self.fixture.packet(per, n=32)
        packet = tuple(v.repeat(2, *([1]*(v.ndim-1))) for v in values)
        memory, mv, x, y, tv, vis = packet
        query = self.fixture.tokenizer.encode('\nUser: where is Amina notebook?\nAssistant:').ids
        at = len(query)
        name = self.fixture.tokenizer.encode(' Amina').ids[0]
        topic = self.fixture.tokenizer.encode(' notebook').ids[0]
        red = self.fixture.tokenizer.encode(' red').ids[0]
        source = unpack_values(memory, per).reshape(2,-1)
        ss = source[0].tolist().index(red)
        y[:, at] = name; y[:, at+1] = red; y[:, -1] = 2; y[:, 30] = -100
        terms = dict(query=torch.tensor([[r,at,128+query.index(name),128+query.index(topic)] for r in (0,1)]),
                     source=torch.tensor([[0,at,0,0],[0,at+1,ss,1],[1,at,0,0]]))
        return packet, terms

    def test_zero_aux_exact_ordinary_adam_and_generic_validation(self):
        for per in (64,1):
            a=self.fixture.model(per);b=copy.deepcopy(a);packet,terms=self.packet(per)
            oa=torch.optim.AdamW(a.parameters(),lr=3e-4);ob=torch.optim.AdamW(b.parameters(),lr=3e-4)
            count=int((packet[3]!=-100).sum())
            objective.update(a,oa,[(packet,terms)],1,8,count,2,2,query_coefficient=0,source_coefficient=0)
            from models.token_memory.schedule import rate
            for group in ob.param_groups:group['lr']=rate(1,8,3e-4)
            (b.loss_sum(*packet)/count).backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.);ob.step()
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0,msg=key)
            bad={k:v.clone() for k,v in terms.items()};bad['source'][0,2]=source_length=packet[0].shape[1]*per
            with self.assertRaises(ValueError):objective.validate(a,packet,bad)

    def test_auxiliary_only_new_selector_and_entity_gates(self):
        for per in (64,1):
            model=self.fixture.model(per);packet,terms=self.packet(per)
            ce,q,s,stats=objective.losses(model,packet,terms)
            (.05*q/2+.02*s/2).backward()
            active=[]
            for key,p in model.named_parameters():
                if p.grad is not None:
                    self.assertTrue(key=='reader.selector.weight' or key.startswith('reader.entity_gate.'),key)
                    if torch.count_nonzero(p.grad):active.append(key)
            self.assertIn('reader.selector.weight',active)
            self.assertTrue(any(k.startswith('reader.entity_gate.') for k in active))
            self.assertEqual(stats['critical_name'],2);self.assertEqual(stats['critical_value'],1)

    def test_actual_global_microbatch_and_exact_serialized_adam_resume(self):
        for per in (64,1):
            a=self.fixture.model(per);b=copy.deepcopy(a);packet,terms=self.packet(per)
            count=int((packet[3]!=-100).sum());pieces=[]
            for row in (0,1):
                ts={k:v[v[:,0]==row].clone() for k,v in terms.items()}
                for v in ts.values():v[:,0]=0
                pieces.append((tuple(v[row:row+1] for v in packet),ts))
            oa=torch.optim.AdamW(a.parameters(),lr=3e-4);ob=torch.optim.AdamW(b.parameters(),lr=3e-4)
            objective.update(a,oa,[(packet,terms)],1,8,count,2,2,do_step=False)
            objective.update(b,ob,pieces,1,8,count,2,2,do_step=False)
            for key,p in a.named_parameters():torch.testing.assert_close(p.grad,dict(b.named_parameters())[key].grad,atol=4e-6,rtol=5e-4,msg=key)
            for step in range(1,9):
                objective.update(a,oa,pieces,step,8,count,2,2)
                if step==4:
                    saved=io.BytesIO();torch.save(dict(model=a.state_dict(),optimizer=oa.state_dict(),step=step,rng=torch.get_rng_state(),cursor=dict(epoch=1,batch_index=0)),saved)
            saved.seek(0);state=torch.load(saved,weights_only=True);b=self.fixture.model(per);b.load_state_dict(state['model']);ob=torch.optim.AdamW(b.parameters(),lr=3e-4);ob.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
            self.assertEqual(state['cursor'],dict(epoch=1,batch_index=0))
            for step in range(5,9):objective.update(b,ob,pieces,step,8,count,2,2)
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0,msg=key)
            for pa,pb in zip(a.parameters(),b.parameters()):
                for key,value in oa.state[pa].items():torch.testing.assert_close(value,ob.state[pb][key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
