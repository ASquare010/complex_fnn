"""Independent causal unfolding, word coordinates and live cache contracts."""
import copy
from dataclasses import replace
import unittest
import torch
from tokenizers import Tokenizer
from models.token_memory import Model, Config
from models.token_memory.surface import geometry, scope, recent
from models.token_memory.core import Config as CoreConfig
from models.token_memory.wordbits import pack_values, unpack_values
from models.token_memory import surface as reference


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = Tokenizer.from_file(str(reference.TOKENIZER))

    def model(self, per=64):
        return Model(Config(words_per_memory=per, read_layers=(0, 1)),
                     core_config=replace(CoreConfig(), width=256, heads=4, layers=2, ffn_hidden_total=32))

    def packet(self, per=64, n=24, blocks=2):
        fact = self.tokenizer.encode(' Amina keeps a notebook in the red box. Quinn keeps a notebook in the blue box.').ids
        fact = (fact+self.tokenizer.encode(' filler.').ids*128)[:blocks*64]
        ids = torch.tensor(fact).reshape(1, blocks, 64)
        g = torch.Generator().manual_seed(4718)
        z = pack_values(torch.randn(1, blocks, 256, generator=g), ids)
        if per == 1:
            z = pack_values(torch.zeros(1, blocks*64, 256), ids.reshape(1, blocks*64, 1))
        query = self.tokenizer.encode('\nUser: where is Amina notebook?\nAssistant:').ids
        x = torch.tensor([[1]+query+[16]*max(0, n-len(query)-1)])[:, :n]
        x[:, 64::64] = 1
        y = torch.randint(3, 4096, x.shape, generator=g)
        mv = torch.ones(z.shape[:2], dtype=torch.bool)
        tv = torch.ones_like(x, dtype=torch.bool)
        visible = torch.full((1, (n+63)//64), z.shape[1], dtype=torch.long)
        return z, mv, x, y, tv, visible

    def test_full_budget_initial_identity_rng_and_scale(self):
        before = torch.get_rng_state().clone()
        a = Model()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = Model(Config(words_per_memory=1))
        self.assertEqual(sum(v.numel() for v in a.parameters()), 23209216)
        self.assertEqual(sum(v.numel() for v in b.parameters()), 23209216)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        z = torch.randn(1, 64, 256)
        projected, _ = a._source_input(z)
        self.assertLess(float(projected.detach().square().mean().sqrt()), .03)
        self.assertGreater(float(projected.detach().square().mean().sqrt()), .01)
        self.assertFalse(any('project' in k for k in a.state_dict()))
        self.assertEqual(a.core.embedding.embedding_dim,256)
        self.assertEqual(len(a.core.blocks),8)
        cache = a.cache(1, 39)
        self.assertEqual(cache.allocated_bytes(), 8*2*103*256*4+39+64+64*8)
        self.assertEqual(set(vars(cache)), {'keys','values','memory_valid','raw_valid','raw_tokens','capacity','memory_count','raw_count'})

    def test_geometry_scope_and_generic_word_coordinates(self):
        model = self.model()
        reader = model.reader
        memory, mv, x, y, tv, visible = self.packet()
        ids = unpack_values(memory).reshape(1, -1)
        valid = torch.ones_like(ids, dtype=torch.bool)
        coordinates = torch.arange(x.shape[1])
        a = recent(ids, valid, x, tv, visible.expand_as(x), coordinates, 64)
        b = recent(ids, valid, x, tv, (visible*64).expand_as(x), coordinates, 1)
        for left, right in zip(a,b):
            torch.testing.assert_close(left,right,atol=0,rtol=0)
        source64 = reader.literal_features(ids.reshape(1,2,64),mv,model.core.embedding.weight,64)
        source1 = reader.literal_features(ids.reshape(1,128,1),mv.repeat_interleave(64,1),model.core.embedding.weight,1)
        with reader.inputs(x,tv,coordinates,model.core.embedding.weight):
            q64 = reader.select_query(source64,visible.expand_as(x))
            q1 = reader.select_query(source1,(visible*64).expand_as(x))
        for key in ('logits','selected','confidence','match','presence','positions','eligible','scope_valid'):
            torch.testing.assert_close(getattr(q64,key),getattr(q1,key),atol=0,rtol=0)

    def test_packed_isolated_logits_gradients_and_future_padding(self):
        a = self.model()
        b = copy.deepcopy(a)
        memory,mv,x,y,tv,visible = self.packet(n=72,blocks=3)
        visible[:] = torch.tensor([[1,2]])
        x[:,64]=1
        selected = (0,3,20,63,64,71)
        labels = torch.full_like(y,-100)
        labels[:,selected] = y[:,selected]
        lp = a.packed(memory,mv,x,tv,visible)
        a.loss_sum(memory,mv,x,labels,tv,visible).backward()
        losses=[]
        for col in selected:
            start=col//64*64;count=int(visible[0,col//64])
            piece=b(memory[:,:count],mv[:,:count],x[:,start:col+1])[:,-1]
            torch.testing.assert_close(piece,lp[:,col],atol=2e-5,rtol=2e-5)
            losses.append(-piece[0,y[0,col]])
        sum(losses).backward()
        for key,p in a.named_parameters():
            self.assertIsNotNone(p.grad,key)
            torch.testing.assert_close(p.grad,dict(b.named_parameters())[key].grad,atol=4e-6,rtol=5e-4,msg=key)
        changed=memory.clone();changed[:,1:]=pack_values(torch.randn_like(changed[:,1:]),torch.randint(3,4096,(1,2,64)))
        future=a.packed(changed,mv,x,tv,visible)
        torch.testing.assert_close(lp[:,:64],future[:,:64],atol=2e-5,rtol=2e-5)
        leaf=memory.detach().requires_grad_()
        (-a.packed(leaf,mv,x,tv,visible)[0,20,17]).backward()
        torch.testing.assert_close(leaf.grad[:,1:],torch.zeros_like(leaf.grad[:,1:]),atol=0,rtol=0)
        padded=torch.cat((memory,torch.randn(1,2,256)),1)
        padvalid=torch.cat((mv,torch.zeros(1,2,dtype=torch.bool)),1)
        torch.testing.assert_close(a.packed(padded,padvalid,x,tv,visible),lp,atol=2e-5,rtol=2e-5)

    def test_public_chunked_cache_payload_blocking_and_empty(self):
        for per in (64,1):
            model=self.model(per)
            memory,mv,x,y,tv,visible=self.packet(per)
            mv[:,0]=False
            expected=model(memory,mv,x,visible=visible)
            cache=model.cache(1,memory.shape[1]+64)
            attrs=set(vars(cache));size=cache.allocated_bytes()
            split=max(1,memory.shape[1]//2)
            model.append_codes(cache,memory[:,:split],mv[:,:split])
            model.append_codes(cache,memory[:,split:],mv[:,split:])
            parts=[]
            for lo,hi in ((0,7),(7,13),(13,x.shape[1])):
                parts.append(model.feed_tokens(cache,x[:,lo:hi])[0])
            torch.testing.assert_close(torch.cat(parts,1),expected,atol=3e-5,rtol=3e-5)
            for layer in model.config.read_layers:
                payload=unpack_values(cache.payload_values(layer)[...,:256],per)
                torch.testing.assert_close(payload,unpack_values(memory,per),atol=0,rtol=0)
            self.assertEqual(attrs,set(vars(cache)));self.assertEqual(size,cache.allocated_bytes())
            self.assertIsNone(model.reader._inputs);self.assertIsNone(model.reader._embedding);self.assertIsNone(model.reader._query)
            empty=torch.zeros(1,0,256);valid=torch.zeros(1,0,dtype=torch.bool)
            public=model(empty,valid,x)
            ec=model.cache(1,64);model.append_codes(ec,empty,valid);cached=model.feed_tokens(ec,x)[0]
            torch.testing.assert_close(public,cached,atol=2e-5,rtol=2e-5)

    def test_rollover_exact_64_words_one_or_64_positions(self):
        for per in (64,1):
            model=self.model(per);memory,mv,x,y,tv,visible=self.packet(per,n=64)
            cache=model.cache(1,memory.shape[1]+(1 if per==64 else 64))
            model.append_codes(cache,memory,mv);model.feed_tokens(cache,x)
            before=cache.allocated_bytes();next_token=torch.tensor([27]);actual=torch.cat((x[:,1:],next_token[:,None]),1);calls=[]
            def encode(words):
                calls.append(words.clone())
                return words.float().mean(-1,keepdim=True).expand(-1,256)*.001
            result=model.advance(cache,next_token,encode)[0]
            if per==64:
                self.assertEqual(len(calls),1);torch.testing.assert_close(calls[0],actual)
                source=pack_values(encode(actual)[:,None],actual[:,None])
            else:
                self.assertEqual(calls,[])
                source=pack_values(torch.zeros(1,64,256),actual[:,:,None])
            expected=model(torch.cat((memory,source),1),torch.ones(1,memory.shape[1]+source.shape[1],dtype=torch.bool),torch.ones(1,1,dtype=torch.long))
            torch.testing.assert_close(result,expected,atol=3e-5,rtol=3e-5)
            self.assertEqual(cache.memory_count,memory.shape[1]+(1 if per==64 else 64))
            self.assertEqual(cache.raw_count,1);self.assertEqual(before,cache.allocated_bytes())


if __name__=='__main__':
    unittest.main()
