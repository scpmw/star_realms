#!/usr/bin/python
"""
Command line utility for generating training data

Usage:
  make_training.py [options] (<name>) [<count>]

Options:
  --greedy <depths>    Do greedy limited tree search, sample at depths [example: 3,4,5,6]
  --max-turns <N>      Maximum number of turns [default: 30]
  --samples <N>        Number of (nested) games to play to determine evaluation [default: 500]
  --model <file>       Reference model to determine "interesting" states
  --threshold <x>      Model difference to count as "interesting" [default: 0.1]
  --thr-samples <N>    Check threshold every time after collecting
                       given number of samples  [default: 50]
"""

import sys
import torch
import numpy
import docopt
from star_realms import training
from star_realms.nn import make_model

args = docopt.docopt(__doc__, argv=sys.argv[1:])
if args['<count>'] is None:
    args['<count>'] = 10

# Load model, if appropriate
model = None
if args['--model'] is not None:
    model = make_model(torch.load(args['--model']))
    model.train(False)
    
# Generate training data
for i in range(int(args.get('<count>'))):
    print()
    print("Game %d" % i)
    train = []
    if args['--greedy'] is None:
        train.append(training.make_training(
            model=model,
            max_turns=int(args.get('--max-turns')),
            samples=int(args.get('--samples')),
            threshold=float(args.get('--threshold')),
            threshold_samples=int(args.get('--thr-samples'))
            ))
    else:
        if model is None:
            print("Tree search requires model!")
            exit(1)
        train.append(training.make_greedy_training(
            model=model,
            max_turns=int(args.get('--max-turns')),
            samples=int(args.get('--samples')),
            depths=[ int(d) for d in args['--greedy'].split(',') ]
            ))    
            
    # Concatenate training data, write out
    statess, valss = list(zip(*train))
    states = numpy.vstack([states for states in statess if len(states.shape) > 1])
    vals = numpy.hstack(valss)
    training.append_training(args['<name>'], states, vals)
