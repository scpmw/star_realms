#!/usr/bin/python
"""
Command line utility for generating training data

Usage:
  make_training.py [options] (<name>) [<count>]

Options:
  --max-turns <N>      Maximum number of turns [default: 30]
  --samples <N>        Number of (nested) games to play to determine evaluation [default: 500]
  --model <file>       Reference model to determine "interesting" states
  --health <addr>      Expose HTTP health check at network address
  
Specific to random generator:
  --threshold <x>      Model difference to count as "interesting" [default: 0.1]
  --thr-samples <N>    Check threshold every time after collecting
                       given number of samples  [default: 50]

Specific to greedy tree search generator:
  --greedy <depths>    Do greedy limited tree search, sample at depths [example: 3,4,5,6]
  --trace              Trace if deriving from model (allows replays)
"""

import sys
import torch
import numpy
import docopt
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
    
from star_realms import training
from star_realms.nn import make_model

args = docopt.docopt(__doc__, argv=sys.argv[1:])
if args['<count>'] is None:
    args['<count>'] = 10

# Load model, if appropriate
model = None
if args['--model'] is not None:
    model = make_model(torch.load(args['--model'], map_location=torch.device('cpu')))
    model.train(False)

# Health check? For the moment we just run an HTTP server that responds with 200
if args['--health'] is not None:
    serverAddr, serverPort = args['--health'].split(':')
    serverPort = int(serverPort)
    class HealthServer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes("OK", "utf-8"))
        def log_message(*_a, **_kw):
            pass
    def health_serve_thread():
        HTTPServer((serverAddr, serverPort), HealthServer).serve_forever()
    threading.Thread(target=health_serve_thread, daemon=True).start()

# Generate training data
for i in range(int(args.get('<count>'))):
    print()
    print("Game %d" % i)
    train = []
    if args['--greedy'] is None:
        data = training.make_training(
            model=model,
            max_turns=int(args.get('--max-turns')),
            samples=int(args.get('--samples')),
            threshold=float(args.get('--threshold')),
            threshold_samples=int(args.get('--thr-samples'))
            )
    else:
        if model is None:
            print("Tree search requires model!")
            exit(1)
        data = training.make_greedy_training(
            model=model,
            max_turns=int(args.get('--max-turns')),
            samples=int(args.get('--samples')),
            depths=[ int(d) for d in args['--greedy'].split(',') ],
            collect_traces=args.get('--trace')
            )
            
    # Concatenate training data, write out
    if len(data[0]) > 0:
        training.append_training(args['<name>'], *data)
