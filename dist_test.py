"""
Trace Training

Usage:
  dist_test.py [options] <training-data> [<model-file>]

Options:
  --check-fraction=<n>  Fraction of training data to use for check (default 0.05)
  --depths=<dpts>       Trace depths in training data (default 3,4,5,6)
  --learning-rate=<n>   Start learning rate (default 2e-4)
  --batch-size=<n>      Training batch size (default 4096)
  --num-workers=<n>     Worker count for loading training data (default 3)
"""


import docopt
import os

import torch
import numpy
import pylab
import os
import h5py
import time

from star_realms import nn, training

if __name__ == '__main__':

    arguments = docopt.docopt(__doc__, version='Trace Training')
    if arguments['--check-fraction'] is None:
        arguments['--check-fraction'] = '0.05'
    if arguments['--depths'] is None:
        arguments['--depths'] = '3,4,5,6'
    if arguments['--learning-rate'] is None:
        arguments['--learning-rate'] = '3e-4'
    if arguments['--batch-size'] is None:
        arguments['--batch-size'] = '4096'
    if arguments['--num-workers'] is None:
        arguments['--num-workers'] = '3'
    print(arguments)
    
    # Initialise PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    device = torch.device('cuda')
    device_cpu = torch.device('cpu')

    # Load training data    
    training_set = training.TrainingDataset(arguments['<training-data>'])
    training_idxs, check_idxs = torch.utils.data.random_split(range(len(training_set)),
        [1-float(arguments['--check-fraction']), float(arguments['--check-fraction'])])
    loader_params = dict(
        batch_size=int(arguments['--batch-size']),
        pin_memory=True,
        num_workers=int(arguments['--num-workers']),
        persistent_workers=True,
    )
    check_loader = training.make_loader(training_set, check_idxs, **loader_params)
    training_loader = training.make_loader(training_set, training_idxs, **loader_params)
    print(len(training_loader))
    
    # Determine sizes
    depths = [ int(d) for d in arguments['--depths'].split(',') ]
    samples = training_set[0]['trace'].shape[0]
    samples_per_depth = samples // len(depths)
    print(f"{samples} samples, {samples_per_depth} per depth")

    # Create signs array
    signs = training.make_signs(depths, samples_per_depth)    
    signs_cuda = signs.cuda()
    
    model_pars = dict(
        layout = (600, 200, 5, 100),
        dropout = 0.1,
        dropout_in = 0.05)
    def make_model(nn_state, **model_pars):
        return nn.make_model(nn_state, **model_pars)
        
    # Try to load last model. Start anew if it fails
    new_model = False
    try:
        model_cpu = torch.load(arguments['<model-file>']).cpu()
        print("Loaded existing model")
    except Exception as e:
        print(e)
        model_cpu = make_model(None, **model_pars) 
        print("Starting new model")
        new_model = True
    
    model = make_model(model_cpu.state_dict(), **model_pars).to(device)
    model_cpu = make_model(model.state_dict(), **model_pars)

    # Make optimiser
    learning_rate = start_learning_rate = float(arguments['--learning-rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_losses_rmse = 1e9
    last_improvement = 0
    optimize_step = 1
    
    for generation in range(100000):
         
        losses = []
        check_losses = []
        start_time = time.time()
        
        # Check current state
        model.train(False)
        for data in check_loader:
            check_losses.append(training.traces_training_step(
                model, data['state'].cuda().float(), data['trace'].cuda().float(), signs_cuda).item())
            print('c', end='', flush=True)
        check_losses_rmse = numpy.sqrt(numpy.average(check_losses))
        print(f" {check_losses_rmse:.04f} {check_losses_rmse < best_losses_rmse}")
        scheduler.step(check_losses_rmse)
        
        # New best? Change learning rate?
        if check_losses_rmse < best_losses_rmse:
            best_losses_rmse = check_losses_rmse
            torch.save(model, f'model-trace-best.pt')
            last_improvement = generation
            
        # Below start learning rate, but can still use coarser optimisation step? Do that first
        if optimizer.param_groups[0]['lr'] < start_learning_rate and \
            optimize_step < len(training_loader):
            
            optimize_step *= 2
            for pg in optimizer.param_groups:
                pg['lr'] = start_learning_rate
            
        print("LR =", [ float(param_group['lr']) for param_group in optimizer.param_groups ], " STEP=", optimize_step)
        
        # Now train model using other chunks
        optimizer.zero_grad()
        model.train(True)
        for i, data in enumerate(training_loader):
            if i > 0 and i % optimize_step == 0:
                print('o', end='', flush=True)
                optimizer.step()
            loss = training.traces_training_step(
                model, data['state'].cuda().float(), data['trace'].cuda().float(), signs_cuda)
            loss.backward()
            losses.append(loss.item())
            print('.', end='', flush=True)
            
        # Do optimiser step
        print('o', end='', flush=True)
        optimizer.step()
        model.train(False)
        torch.save(model, f'model-trace-{generation}.pt')

        # Log loss
        losses_rmse = numpy.sqrt(numpy.average(losses))
        print(f" {losses_rmse:.04f} ({time.time()-start_time}) s")            