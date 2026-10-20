"""Raw-token baseline training; memory-model training has its own entry point."""
import sys,argparse,json,math,random,time,signal,zipfile,os
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.token_memory.runtime import ROOT,ENCODER_SHA,pin_cpu,digest,write_json,save,load
from models.raw_transformer.runtime import worker_lock


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--steps',type=int,help='Total updates, including restored updates; fresh default 20000')
    p.add_argument('--resume',type=Path)
    p.add_argument('--microbatch',type=int,help='Fresh default 8; may change on resume while total batch stays fixed')
    p.add_argument('--batch-size',type=int,help='Fresh default 32')
    p.add_argument('--lr',type=float,help='Fresh peak LR default 0.0003')
    p.add_argument('--valid-documents',type=int,help='Fixed development sample, fresh default 100')
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cpu',type=int,help='Pin to this allowed logical CPU; default highest allowed')
    p.add_argument('--inspect',action='store_true',help='Print model/settings only; no training')
    return p


def source_hashes(arm):
    from models.token_memory.training import source_hashes as original_sources
    sources=original_sources()
    for directory in (arm,):
        for path in (ROOT/'src/models'/directory).glob('*.py'):
            sources[str(path.relative_to(ROOT))]=digest(path)
    return sources


def _check_legacy_sources(previous,current):
    previous={k.replace('\\','/'):v for k,v in previous.items()}
    current={k.replace('\\','/'):v for k,v in current.items()}
    changed={k for k in previous.keys()|current.keys() if previous.get(k)!=current.get(k)}
    # Reviewed execution-only migration: disable raw activation recomputation,
    # allow its single-worker memory budget, and record this migration.
    approved={
        'src/models/raw_transformer/transformer.py':'0c601eafb4b44e2d53a72165b12df2e0a7289c5131085b490cf5d230666862eb',
        'src/models/comparison/training.py':'0075262d156c3ffe5261828214c417d88378be0f0a2db49e6cda6e0325bb89f0',
    }
    approved={k:{v} for k,v in approved.items()}
    # Hierarchy retirement: raw forward/optimizer/data/schedule are unchanged.
    retired={'src/models/comparison/training.py': 'fa1ffc9a9aa9ffdc9b401a7a238db8b6fe10ecf150a1698fb778009f9495f16a', 'src/models/comparison/layers.py': '7a7a70e88f32f06e4f3d830746504f9094a3cddd138b566779f936900d91ac20', 'src/models/comparison/runtime.py': '7d5d5e1eb0da55ef9b25a583cd604bc10440374e0bfdd5c618b2d559e1d28e2d'}
    for k,v in retired.items(): approved.setdefault(k,set()).add(v)
    if any(k not in current or previous.get(k) not in approved.get(k,set()) for k in changed):
        raise ValueError('Sources changed; use the saved source archive')
    return sorted(changed)


def check_resume_sources(previous,current):
    previous={k.replace(chr(92),'/'):v for k,v in previous.items()}
    current={k.replace(chr(92),'/'):v for k,v in current.items()}
    if previous==current:return []
    entry_migration=Path(__file__).with_name('chat_entry_migration.json')
    if entry_migration.exists():
        entry=json.loads(entry_migration.read_text())
        if current==entry['after']:
            current=entry['before']
    if any(k.startswith('src/models/comparison/') for k in previous):
        migration=json.loads((Path(__file__).with_name('source_migration.json')).read_text())
        # Validate the old tree, including previously approved execution-only changes.
        old_current={k:v for k,v in current.items() if not k.startswith('src/models/raw_transformer/')}
        old_current.update(migration['before'])
        _check_legacy_sources(previous,old_current)
        for k,v in migration['after'].items():
            if current.get(k)!=v:raise ValueError('Relocated sources changed; use saved source archive')
        expected=set(old_current)-set(migration['before'])|set(migration['after'])
        if set(current)!=expected:raise ValueError('Unexpected source additions/removals')
        return sorted(k for k in previous.keys()|current.keys() if previous.get(k)!=current.get(k))
    return _check_legacy_sources(previous,current)


def resolve(args,parent):
    if args.arm!='raw_transformer': raise ValueError('Hierarchical training is retired; use raw_transformer or token_memory')
    recipe=dict(data=str((ROOT/'dump/data/token-memory').resolve()),steps=20000,microbatch=8,arm=args.arm,
                batch_size=32,lr=3e-4,seed=47,valid_documents=100)
    if parent:
        if parent.get('format')!='comparison-v1' or parent['recipe'].get('arm')!=args.arm: raise ValueError('Resume requires the same comparison architecture')
        recipe=dict(parent['recipe'])
    for key in ('data','steps','microbatch','batch_size','lr','valid_documents'):
        value=getattr(args,key)
        if value is None: continue
        if key=='data': value=str(value.resolve())
        if parent and key not in ('steps','microbatch') and value!=recipe[key]: raise ValueError(f'Resume preserves {key}')
        recipe[key]=value
    if any(recipe[k]<1 for k in ('steps','microbatch','batch_size','valid_documents')): raise ValueError('Counts must be positive')
    if recipe['batch_size']%recipe['microbatch']: raise ValueError('Batch size must be divisible by microbatch')
    if not math.isfinite(recipe['lr']) or recipe['lr']<=0: raise ValueError('LR must be finite and positive')
    start=parent['step'] if parent else 0
    if recipe['steps']<=start: raise ValueError('Set total --steps above saved step')
    if parent and recipe['steps']!=parent['recipe']['steps']:
        # Explicit extension continues at the old schedule endpoint without resetting warmup.
        old=parent['schedule'];rate=learning_rate(old,start)
        schedule=dict(start=start,end=recipe['steps'],peak=rate,minimum=rate,warmup=0)
    else:
        schedule=parent['schedule'] if parent else dict(start=0,end=recipe['steps'],peak=recipe['lr'],minimum=recipe['lr']*.1,warmup=200)
    return recipe,schedule


def learning_rate(s,step):
    offset=step-s['start'];progress=min(1,max(0,(offset-1)/max(1,s['end']-s['start']-1)))
    warm=min(1,offset/s['warmup']) if s['warmup'] else 1
    return warm*(s['minimum']+(s['peak']-s['minimum'])*(1+math.cos(math.pi*progress))/2)


def execute(args,parent=None):
    import torch
    from torch.nn import functional as F
    import importlib
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from models.raw_transformer.data import Dataset
    module=importlib.import_module('models.'+args.arm+'.transformer')
    Model,Config=module.Model,module.Config
    batch=importlib.import_module('models.'+args.arm+'.data').batch
    generate=importlib.import_module('models.'+args.arm+'.generation').generate
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this Python environment')
    recipe,schedule=resolve(args,parent);torch.manual_seed(recipe['seed'])
    model=Model(Config(),recipe['seed']).to(args.device)
    count=sum(p.numel() for p in model.parameters())
    print(f'{"RESUME" if parent else "FRESH"} {args.arm} | {count:,} trainable parameters | {args.device}\n'
          f'512 target positions with up to 2048 older raw tokens; CE only, FP32.\n'
          f'Batch {recipe["batch_size"]} = micro {recipe["microbatch"]} x {recipe["batch_size"]//recipe["microbatch"]}; '
          f'updates {parent["step"] if parent else 0} -> {recipe["steps"]}',flush=True)
    if args.inspect: return
    if Path(args.name).name!=args.name or args.name in ('.','..'): raise ValueError('Name must be one folder name')
    out=ROOT/'dump'/args.name
    if out.exists(): raise ValueError(f'Output exists: {out}; choose a new name')
    sources=source_hashes(args.arm)
    migrated=check_resume_sources(parent['sources'],sources) if parent else []
    if parent and parent['encoder_sha256']!=ENCODER_SHA: raise ValueError('Encoder identity changed')
    codec=SimpleNamespace(device=args.device,model=torch.nn.Identity(),
            tokenizer=Tokenizer.from_file(str(Path(recipe['data'])/'tokenizer.json')))
    data=Path(recipe['data']);manifest_hash=digest(data/'manifest.json')
    if parent and parent['data_manifest_sha256']!=manifest_hash: raise ValueError('Dataset identity changed')
    train=Dataset(data,'train');valid=Dataset(data,'valid')
    if digest(data/'tokenizer.json')!=train.meta['tokenizer_sha256']: raise ValueError('Tokenizer file changed')
    from tokenizers import Tokenizer
    if Tokenizer.from_file(str(data/'tokenizer.json')).to_str()!=codec.tokenizer.to_str(): raise ValueError('Encoder/data tokenizer mismatch')
    print(f'Memory-mapped {len(train.rows):,} documents / {len(train):,} train segments; no full-corpus encoding.\n'
          'Same shuffled segments and target tokens as token_memory; no encoder pass.',flush=True)
    n=min(recipe['valid_documents'],len(valid.rows))
    valid_indices=[(i*len(valid.rows)//n)*valid.blocks+j for i in range(n) for j in range(valid.blocks)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=.01)
    step=0;best=float('inf')
    if parent:
        model.load_state_dict(parent['model']);optimizer.load_state_dict(parent['optimizer']);step=parent['step']
        torch.set_rng_state(parent['torch_rng'])
        if args.device=='cuda': torch.cuda.set_rng_state_all(parent['cuda_rng'])
    out.mkdir(parents=True)
    if parent:
        write_json(out/'resume.json',dict(parent=str(args.resume),step=step,
            previous_microbatch=parent['recipe']['microbatch'],microbatch=recipe['microbatch'],
            batch_size=recipe['batch_size'],optimizer_restored=True,
            next_sample_draw=step*recipe['batch_size'],source_migration=migrated,
            note='Microbatch regrouping preserves targets and token-weighted loss, but floating-point rounding may differ.'))
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources: archive.write(ROOT/path,path)
        archive.write(Path(__file__).with_name('source_migration.json'),'src/models/raw_transformer/source_migration.json')
        entry_migration=Path(__file__).with_name('chat_entry_migration.json')
        if entry_migration.exists():archive.write(entry_migration,'src/models/raw_transformer/chat_entry_migration.json')
    write_json(out/'recipe.json',{**recipe,'config':asdict(model.config),'parameters':count,'encoder_sha256':ENCODER_SHA,
        'frozen_parameters':sum(p.numel() for p in codec.model.parameters()),
        'schedule':schedule,'data_manifest_sha256':manifest_hash,'data_checks':'Tokenizer identity, manifest identity and binary sizes; no repeated full corpus hashing'})
    stopping=[False]
    def stop(*_):
        stopping[0]=True;print('Stopping after current optimizer update; saving checkpoint.',flush=True)
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    def checkpoint(name):
        save(out/name,dict(format='comparison-v1',step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
             recipe=recipe,schedule=schedule,best=best,sources=sources,data_manifest_sha256=manifest_hash,
             encoder_sha256=ENCODER_SHA,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else []))
    @torch.no_grad()
    def evaluate():
        model.eval();total=correct=0;loss_sum=0.
        for start in range(0,len(valid_indices),recipe['microbatch']):
            items=[valid[i] for i in valid_indices[start:start+recipe['microbatch']]]
            inputs,y=batch(items,codec);logits=model(*inputs)
            loss_sum+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'))
            total+=int((y!=-100).sum());correct+=int(((logits.argmax(-1)==y)&(y!=-100)).sum())
        model.train()
        return dict(ce=loss_sum/total,token_accuracy=correct/total,tokens=total)
    history=[];begin=time.monotonic();start_step=step;epoch_cache=None;order=None
    def validate():
        nonlocal best
        score=evaluate();history.append(dict(step=step,**score));write_json(out/'validation.json',history)
        print(f'VALID {step}: CE {score["ce"]:.5f} | token accuracy {score["token_accuracy"]:.2%}',flush=True)
        if score['ce']<best: best=score['ce'];checkpoint('best.pt')
        return score
    try:
        checkpoint('initial.pt');checkpoint('last.pt');validate()
        with (out/'training.jsonl').open('x',encoding='utf-8') as log:
            while step<recipe['steps'] and not stopping[0]:
                items=[]
                for draw in range(step*recipe['batch_size'],(step+1)*recipe['batch_size']):
                    epoch,offset=divmod(draw,len(train))
                    if epoch!=epoch_cache:
                        order=list(range(len(train)));random.Random(recipe['seed']+epoch).shuffle(order);epoch_cache=epoch
                    items.append(train[order[offset]])
                total=sum(len(y) for _,y in items);optimizer.zero_grad(set_to_none=True);value=0.
                rate=learning_rate(schedule,step+1)
                for group in optimizer.param_groups: group['lr']=rate
                for offset in range(0,len(items),recipe['microbatch']):
                    inputs,y=batch(items[offset:offset+recipe['microbatch']],codec)
                    logits=model(*inputs)
                    loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum')/total
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    loss.backward();value+=float(loss.detach())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if any(p.grad is not None for p in codec.model.parameters()): raise ValueError('Encoder received gradients')
                memory_limit=10
                if args.device=='cuda' and torch.cuda.max_memory_reserved()>memory_limit*1024**3: raise RuntimeError(f'{memory_limit} GiB memory guard exceeded; resume the last saved checkpoint with a smaller microbatch and new name')
                optimizer.step();step+=1
                entry=dict(step=step,ce=value,lr=rate,targets=total,grad_norm=float(norm))
                log.write(json.dumps(entry)+'\n');log.flush()
                if step%25==0 or step==start_step+1:
                    elapsed=time.monotonic()-begin;speed=(step-start_step)/max(elapsed,.001)
                    reserved=torch.cuda.max_memory_reserved()/1024**3 if args.device=='cuda' else 0
                    print(f'TRAIN {step}/{recipe["steps"]} | CE {value:.5f} | LR {rate:.3g} | {speed:.2f} updates/s | ETA {(recipe["steps"]-step)/speed/60:.1f}m | reserved {reserved:.2f} GiB',flush=True)
                    write_json(out/'status.json',dict(state='training',pid=os.getpid(),total=recipe['steps'],**entry))
                if step%args.eval_every==0: validate()
                if step%args.save_every==0:
                    checkpoint('last.pt')
                    if step%1000==0: checkpoint(f'step-{step}.pt')
            checkpoint('last.pt')
        final=validate();checkpoint('last.pt')
        restored=load(out/'last.pt');model.load_state_dict(restored['model'])
        if evaluate()!=final: raise ValueError('Final checkpoint replay mismatch')
        samples=[]
        for i in range(min(2,len(valid.rows))):
            prompt=valid.rows[i,:480].astype('int64').tolist()
            ids=generate(model,codec,prompt,96)
            samples.append(dict(prompt=codec.tokenizer.decode(prompt),generated=codec.tokenizer.decode(ids)))
        write_json(out/'samples.json',samples)
        result=dict(state='complete' if step==recipe['steps'] else 'paused',step=step,total=recipe['steps'],final=final,
                    last_sha256=digest(out/'last.pt'),final_reload_replay=True)
        write_json(out/'result.json',result);write_json(out/'status.json',result)
        print(f'{result["state"].upper()}: {out}',flush=True)
    except BaseException as error:
        write_json(out/'status.json',dict(state='failed',step=step,error=repr(error)));raise
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


def main(arm):
    args=parser().parse_args();args.arm=arm
    if args.eval_every<1 or args.save_every<1: raise SystemExit('Intervals must be positive')
    if args.cpu is not None:
        import psutil
        process=psutil.Process()
        if args.cpu not in process.cpu_affinity():raise SystemExit('Requested CPU is not available')
        process.cpu_affinity([args.cpu])
    print(f'CPU affinity: {pin_cpu()} | Python: {sys.executable}',flush=True)
    parent=load(args.resume) if args.resume else None
    if args.inspect: execute(args,parent)
    else:
        with worker_lock(args.arm): execute(args,parent)
