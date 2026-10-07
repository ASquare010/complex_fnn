"""Two selected models: Branch Sigmoid and the span-32 context encoder."""
import argparse
import json
from storage import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect', help='Show retained model endpoints')
    generate = sub.add_parser('generate', help='Greedy generation from the Step 1 winner')
    generate.add_argument('text')
    generate.add_argument('--tokens', type=int, default=64)
    generate.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    encode = sub.add_parser('encode', help='Encode with a trained Branch Sigmoid compressor')
    encode.add_argument('text')
    encode.add_argument('--output', default='dump/selected-memory.pt')
    encode.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    reconstruct = sub.add_parser('reconstruct', help='Reconstruct using the selected encoder/decoder')
    reconstruct.add_argument('text')
    train = sub.add_parser('train', help='Explicitly start a new Branch Sigmoid run')
    train.add_argument('--config', default='src/config/branch_sigmoid.json')
    train.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps({'step1':read_json('src/config/branch_sigmoid.json'),
                          'step2':read_json('src/config/position_compressor.json')},indent=2))
        return
    if args.command == 'generate':
        import torch
        from models.branch_sigmoid.loader import load_winner
        model, tokenizer = load_winner(args.device)
        ids = tokenizer.encode(args.text).ids
        if not ids or args.tokens < 0:
            raise ValueError('Provide nonempty tokenized text and nonnegative tokens')
        with torch.no_grad():
            for _ in range(args.tokens):
                x = torch.tensor([ids[-model.config.context:]],device=args.device)
                ids.append(int(model(x)[0,-1].argmax()))
        print(tokenizer.decode(ids))
        return
    if args.command == 'train':
        from models.branch_sigmoid.training import run
        run(read_json(args.config),'branch_sigmoid',resume=args.resume)
        return
    from models.position_compressor.codec import Codec
    config = read_json('src/config/position_compressor.json')
    from pathlib import Path
    endpoint = config['encoder' if args.command == 'encode' else 'checkpoint']
    if not Path(endpoint).is_file():
        raise FileNotFoundError('Branch Sigmoid compressor weights are not trained yet. Legacy CurveFFN weights are incompatible.')
    if args.command == 'encode':
        codec = Codec.load_encoder(config['encoder'],args.device)
        print(codec.save_memory(args.text,args.output))
    else:
        codec = Codec.load(config['checkpoint'])
        print(json.dumps(codec.reconstruct(args.text),indent=2))

if __name__ == '__main__':
    main()
