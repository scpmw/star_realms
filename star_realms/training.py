
import os
import random
import numpy
import torch

from star_realms.state import GameState
from star_realms.play import play_random_turn, play_greedy_turn, random_win_prob
from star_realms.nn import prob_to_value, value_to_prob, model_game_prob

TRAINING_DIR = 'training'


def load_training(name, path=TRAINING_DIR, load_traces=False):
    """ Load training data from disk
    
    :param name: name of data set
    :returns: (states, vals) pair
    """
    
    state_path = os.path.join(path, 
        "states-{}.npy".format(name))
    vals_path = os.path.join(path, 
        "vals-{}.npy".format(name))
    traces_path = os.path.join(TRAINING_DIR, 
        "traces-{}.npy".format(name))
    
    statesd = numpy.load(state_path)
    valsd = numpy.load(vals_path)
    if not load_traces:
        return statesd, valsd
    
    tracesd = numpy.load(traces_path)
    return statesd, valsd, tracesd


def append_training(name, states, vals, traces=None):
    """ Append training data to a training data set
    
    :param name: name of data set
    :param states: State description as numpy array
    :param vals: State evaluations
    """

    assert len(states.shape) == 2
    assert len(vals.shape) == 1
    assert states.shape[0] == vals.shape[0]
    if traces is not None:
        assert len(traces.shape) == 3
        assert traces.shape[0] == vals.shape[0]
        assert traces.shape[2] == states.shape[1]

    state_path = os.path.join(TRAINING_DIR, 
        "states-{}.npy".format(name))
    vals_path = os.path.join(TRAINING_DIR, 
        "vals-{}.npy".format(name))
    traces_path = os.path.join(TRAINING_DIR, 
        "traces-{}.npy".format(name))

    # Load previously existing data
    if os.path.isfile(state_path):
        assert os.path.isfile(vals_path)
        statesd = numpy.load(state_path)
        valsd = numpy.load(vals_path)
        # If there are traces, we must provide them on append as well
        if traces is not None:
            assert os.path.isfile(traces_path)
            tracesd = numpy.load(traces_path)
        else:
            assert not os.path.isfile(traces_path)
        # Make sure shape is as expected
        assert len(statesd.shape) == 2
        assert len(valsd.shape) == 1
        assert statesd.shape[1] == states.shape[1], 'Feature vector length mismatch!'
        if traces is not None:
            assert tracesd.shape[1] == traces.shape[1], 'Sample count mismatch!'
            assert tracesd.shape[2] == traces.shape[2], 'Feature vector length mismatch!'
        # Concatenate
        statesd = numpy.vstack([statesd, numpy.array(states)])
        valsd = numpy.hstack([valsd, vals])
        if traces is not None:
            tracesd = numpy.concatenate([tracesd, traces], axis=0)
    else:
        statesd = states
        valsd = vals
        if traces is not None:
            tracesd = traces
        

    # Write to separate file (so we don't overwrite everything if something goes wrong)
    numpy.save(state_path + "-new", statesd.astype('int16'))
    numpy.save(vals_path + "-new", valsd)
    if traces is not None:
        numpy.save(traces_path + "-new", tracesd.astype('int16'))
    
    # Replace
    if os.path.isfile(state_path):
        if os.path.isfile(state_path + "-old"):
            os.remove(state_path + "-old")
        if os.path.isfile(vals_path + "-old"):
            os.remove(vals_path + "-old")
        if os.path.isfile(traces_path + "-old"):
            os.remove(traces_path + "-old")
        os.rename(state_path, state_path + "-old")
        os.rename(vals_path, vals_path + "-old")
        if traces is not None:
            os.rename(traces_path, traces_path + "-old")
    os.rename(state_path + "-new.npy", state_path)
    os.rename(vals_path + "-new.npy", vals_path)
    if traces is not None:
        os.rename(traces_path + "-new.npy", traces_path)
    
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
    for _ in range(max_turns):
    
        # Play, check whether game is over
        play_random_turn(gs)
        if gs.is_over() or gs.player1.authority < 5 or gs.player2.authority < 5:
            break
        if show_progress:
            print(".", end='', flush=True)
        
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
            if show_new:
                print()
                print(gs.describe())
                if model is not None:
                    print("Model:", m)
                print("Actual:", prob_to_value(numpy.average(probs)))
            states.append(gs.to_array())
            vals.append(prob_to_value(numpy.average(probs)))
            
    return numpy.array(states), numpy.array(vals)
    

def make_greedy_training(model, max_turns=30, samples=20, depths=[3,4,5,6], skip_chance=0.9,
                         collect_traces=False, show_progress=True, show_new=True):
    
    states = []; vals = []; traces = []
    
    gs = GameState()
    for _ in range(max_turns):
    
        # Clear caches after every turn - some efficiency might get lost, but
        # this prevents excessive memory buildup
        finish_move_cache = {}
        game_prob_cache = {}
    
        # Play, check whether game is (close to) over
        play_random_turn(gs)
        if gs.is_over() or gs.player1.authority < 5 or gs.player2.authority < 5:
            break
        if show_progress:
            print(".", end='', flush=True)

        if random.random() < skip_chance:
            continue

        # Take samples
        probs = []; trcs = []
        probs_d = { d : [] for d in range(max(depths)+1) }
        for i in range(samples):
            gs2 = GameState(gs)
            for depth in range(max(depths)+1):

                # Continue adding probabilities even if game is over
                if gs2.is_over():
                    gs2.move += 1
                else:
                    play_greedy_turn(gs2, model, finish_move_cache, game_prob_cache, verbose=0)

                # Get model opinion on the state
                prob = model_game_prob(model, gs2)
                if (gs2.move - gs.move) % 2 != 0:
                    prob = 1 - prob
                if depth in depths:
                    probs.append(prob)
                    if collect_traces:
                        trcs.append(gs2.to_array())
                probs_d[depth].append(prob)
        
        # Take average
        if show_new:
            print()
            print("########################")
            print(gs.describe())
            if model is not None:
                print("Model: {:.3f}".format(model(torch.tensor(gs.to_array(), dtype=torch.float)).item()))
            print("Actual: {:.3f}+-{:.3f} ({} samples)".format(
                prob_to_value(numpy.average(probs)), numpy.std(prob_to_value(probs),ddof=1) / numpy.sqrt(len(probs)), len(probs)))
            print("At depths:", ", ".join([ "{}: {:.3f}+-{:.3f}".format(
                d, prob_to_value(numpy.average(probs_d[d])), numpy.std(prob_to_value(probs_d[d]),ddof=1) / numpy.sqrt(len(probs_d[d])))
                for d in probs_d ]))

        states.append(gs.to_array())
        vals.append(prob_to_value(numpy.average(probs)))
        if collect_traces:
            traces.append(trcs)

    if collect_traces:
        return numpy.array(states), numpy.array(vals), numpy.array(traces)
    else:
        return numpy.array(states), numpy.array(vals)
