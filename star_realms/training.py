
import os
import numpy
import torch

from star_realms.state import GameState
from star_realms.play import play_random_turn, random_win_prob
from star_realms.nn import prob_to_value, value_to_prob

TRAINING_DIR = 'training'

def append_training(name, states, vals):
    """ Append training data to a training data set
    
    :param name: name of data set
    :param states: State description as numpy array
    :param vals: State evaluations
    """
    
    state_path = os.path.join(TRAINING_DIR, 
        "states-{}.npy".format(name))
    vals_path = os.path.join(TRAINING_DIR, 
        "vals-{}.npy".format(name))

    # Load previously existing data
    if os.path.isfile(state_path):
        assert os.path.isfile(vals_path)
        statesd = numpy.load("states-{}.npy".format(name))
        valsd = numpy.load("vals-{}.npy".format(name))
        statesd = numpy.vstack([statesd, numpy.array(states)])
        valsd = numpy.hstack([valsd, vals])
        # Make sure shape is as expected
        assert len(statesd.shape) == 2
        assert len(valsd.shape) == 1
        assert statesd.shape[1] == states.shape[1]
    else:
        statesd = numpy.empty((0, statesd.shape[1]))
        valsd = numpy.empty((0,))

    # Write to separate file (so we don't overwrite everything if something goes wrong)
    numpy.save(state_path + "-new", statesd)
    numpy.save(vals_path + "-new", valsd)
    
    # Replace
    if os.path.isfile(state_path):
        os.rename(state_path, state_path + "-old")
        os.rename(vals_path, vals_path + "-old")
    os.rename(state_path + "-new", state_path)
    os.rename(vals_path + "-new", vals_path)
    
def make_training(max_turns=30, samples=500, 
                  model=None, threshold=0.1, threshold_samples=50,
                  show_progress=True, show_new=True):
    """ Play a random game, generate training data from states along the way.
    
    :param max_turns: Maximum number of turns to play
    :param samples: Number of (nested) games to play to determine evaluation
    :param model: Reference model to determine "interesting" states
    :param threshold: Model difference to count as "interesting"
    :param threshold_samples: Check threshold every time after collecting
       given number of samples
    :param show_progress: Show progress bar?
    :param show_new: Show states added to training set?
    """
    states = []; vals = []
    gs = GameState()
    for _ in range(30):
    
        # Play, check whether game is over
        play_random_turn(gs)
        if gs.is_over() or gs.player1.authority < 5 or gs.player2.authority < 5:
            break
        if show_progress:
            print(".", end='')
        
        # Get model evaluation
        if model is not None:
            m = model(torch.tensor(gs.to_array(), dtype=torch.float)).item()
            
        # Determine real evaluation (in steps - bail out early if the
        # model looks pretty much right)
        probs = []
        under_threshold = False
        for nsample in range(samples // threshold_samples):
            probs.append(random_win_prob(gs, threshold_samples))
            if model is not None:
                under_threshold = numpy.abs(value_to_prob(m) - numpy.average(probs)) < threshold
            if under_threshold:
                break
                
        # Add to training set?
        if not under_threshold:
            print()
            print(gs.describe())
            if model is not None:
                print("Model:", m)
            print("Actual:", prob_to_value(numpy.average(probs)))
            states.append(gs.to_array())
            vals.append(prob_to_value(numpy.average(probs)))
            
    return numpy.array(states), numpy.array(vals)