
import os
import random
import numpy
import torch
import h5py

from star_realms.state import GameState, PlayerState
from star_realms.play import play_random_turn, play_greedy_turn, random_win_prob
from star_realms.nn import prob_to_value, value_to_prob, model_game_prob, model_game_prob_array

TRAINING_DIR = 'training'


def load_training(name, path=TRAINING_DIR, load_traces=False):
    """ Load training data from disk
    
    :param name: name of data set
    :returns: (states, vals) pair
    """
    
    state_path = os.path.join(path, 
        "{}-states.npy".format(name))
    vals_path = os.path.join(path, 
        "{}-vals.npy".format(name))
    traces_path = os.path.join(path, 
        "{}-traces.npy".format(name))
    
    statesd = numpy.load(state_path)
    valsd = numpy.load(vals_path)
    if not load_traces:
        return statesd, valsd
    
    tracesd = numpy.load(traces_path)
    return statesd, valsd, tracesd

def load_training_hdf5(name, path=TRAINING_DIR, load_traces=False, slc=slice(None)):
    """ Load training data from disk
    
    :param name: name of data set
    :returns: (states, vals) pair
    """
    
    h5_path = os.path.join(path, 
        "{}.h5".format(name))
    with h5py.File(h5_path, 'r') as f:
        return load_training_hdf5_file(f, load_traces, slc)


def load_training_hdf5_file(file, load_traces=False, slc=slice(None)):
    """ Load training data from disk
    
    :param name: name of data set
    :returns: (states, vals) pair
    """
    
    statesd = numpy.array(file['states'][slc])
    valsd = numpy.array(file['vals'][slc])
    if not load_traces:
        return statesd, valsd
    
    tracesd = numpy.array(file['traces'][slc])
    return statesd, valsd, tracesd

def append_training(name, states, vals, traces=None, path=TRAINING_DIR):
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

    state_path = os.path.join(path, 
        "{}-states.npy".format(name))
    vals_path = os.path.join(path, 
        "{}-vals.npy".format(name))
    traces_path = os.path.join(path, 
        "{}-traces.npy".format(name))

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


def append_training_hdf5(name, states, vals, traces=None, path=TRAINING_DIR):
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

    path = os.path.join(path, "{}.h5".format(name))

    # Open if already exists
    if os.path.isfile(path):
        file = h5py.File(path, 'r+')
        statesd = file['states']
        valsd = file['vals']
        # If there are traces, we must provide them on append as well
        if traces is not None:
            tracesd = file['traces']
        else:
            assert not os.path.isfile(traces_path)
        # Make sure shape is as expected
        assert len(statesd.shape) == 2
        assert len(valsd.shape) == 1
        assert statesd.shape[1] == states.shape[1], 'Feature vector length mismatch!'
        assert statesd.shape[0] == valsd.shape[0], 'Record number mismatch!'
        assert statesd.maxshape[0] is None, 'States cannot be appended to!'
        assert valsd.maxshape[0] is None, 'Values cannot be appended to!'
        if traces is not None:
            assert tracesd.shape[1] == traces.shape[1], 'Sample count mismatch!'
            assert tracesd.shape[2] == traces.shape[2], 'Feature vector length mismatch!'
            assert tracesd.shape[0] == valsd.shape[0], 'Record number mismatch!'
            assert tracesd.maxshape[0] is None, 'Traces cannot be appended to!'
        # Write after existing data
        start = valsd.shape[0]
        total = start + vals.shape[0]
        valsd.resize(total, axis=0)
        statesd.resize(total, axis=0)
        if traces is not None:
            tracesd.resize(total, axis=0)
        valsd[start:] = vals
        statesd[start:,:] = states
        tracesd[start:,:,:] = traces
        file.close()
    else:
        # Create otherwise
        file = h5py.File(path, 'w')
        statesd = file.create_dataset('states', data=states,
            dtype='i2', maxshape=(None, states.shape[1]), compression="lzf")
        valsd = file.create_dataset('vals', data=vals,
            dtype=float, maxshape=(None, ), compression="lzf", shuffle=True)
        if traces is not None:
            tracesd = file.create_dataset('traces', data=traces,
                dtype='i2', maxshape=(None, traces.shape[1], traces.shape[2]), compression="lzf", shuffle=True)
        file.close()
        
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
                print("Model: {:.3f}".format(model_game_prob(model, gs)))
            print("Actual: {:.3f}+-{:.3f} ({} samples)".format(
                numpy.average(probs), numpy.std(probs,ddof=1) / numpy.sqrt(len(probs)), len(probs)))
            print("At depths:", ", ".join([ "{}: {:.3f}+-{:.3f}".format(
                d, numpy.average(probs_d[d]), numpy.std(probs_d[d],ddof=1) / numpy.sqrt(len(probs_d[d])))
                for d in probs_d ]))

        states.append(gs.to_array())
        vals.append(numpy.average(probs))
        if collect_traces:
            traces.append(trcs)

    if collect_traces:
        return numpy.array(states), numpy.array(vals), numpy.array(traces)
    else:
        return numpy.array(states), numpy.array(vals)

def reevaluate_game_probs(model, traces, depths=[3,4,5,6], device=None):
    """ Re-evaluate the game from traces """
    
    assert traces.shape[1] % len(depths) == 0
    samples = traces.shape[1] // len(depths)

    # Determine what game states are going to go in with what sign
    # (i.e. same player to move as the game state we're interested in?)    
    signs = []; ix_d = { d : [] for d in depths }
    for sample in range(samples):
        for depth in depths:
            ix_d[depth].append(len(signs))
            signs.append(-1 if depth % 2 == 0 else 1)
    signs = numpy.array(signs)
    
    # Now run all traces through the model, and average up to get the
    # new evaluations
    new_vals = []
    model.train(False)
    for trace in traces:
        probs = signs * (model_game_prob_array(model, trace, device) - 0.5) + 0.5
        new_vals.append(numpy.average(probs))

    return numpy.array(new_vals)

class TrainingDataset(torch.utils.data.Dataset):
    """Interface for PyTorch to load (HDF5) training data
    
    Note that for efficiency we implement __getitems__ such that it
    returns a dictionary of tensors instead of a list of dictionaries.
    This means that for use with a batch_sampler, the collate_fn
    must be overridden to just do conversions.
    """
    def __init__(self, h5_file, traces=True, transforms=[]):
        self._file = h5py.File(h5_file, 'r')
        self._states = self._file['states']
        self._vals = self._file['vals']
        if traces:
            self._traces = self._file['traces']
        else:
            self._traces = None
        self._transforms = transforms
    def __len__(self):
        return self._vals.shape[0]
    def __getitem__(self, idx):
        result = dict(
            state=self._states[idx],
            val=self._vals[idx]
        )   
        if self._traces:
            result['trace'] = self._traces[idx]
        if self._transforms:
            result = self._transforms(result)
        return result
    def __getitems__(self, idxs):
        result = dict(
            state=self._states[idxs],
            val=self._vals[idxs]
        )   
        if self._traces:
            result['trace'] = self._traces[idxs]
        if self._transforms:
            result = self._transforms(result)
        return result
    def __getstate__(self):
        return (self._file.filename, self._traces is not None, self._transforms)
    def __setstate__(self,state):
        self._file = h5py.File(state[0], 'r')
        self._states = self._file['states']
        self._vals = self._file['vals']
        if state[1]:
            self._traces = self._file['traces']
        else:
            self._traces = None
        self._transforms = state[2]
        
        
def make_loader(ds :TrainingDataset, sampler_ds : torch.utils.data.Dataset = None, batch_size :int = 2048, **kwargs):

    if sampler_ds is None:
        sampler_ds = ds
    batch_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(sampler_ds), batch_size=batch_size, drop_last=False
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_sampler=batch_sampler,
        # Data loader already collates results, just need to convert
        collate_fn=torch.utils.data.default_convert,
        **kwargs
    )

def make_signs(depths, samples_per_depth):
    """ Create signs array
    
    Used for indicating whether we are 
    
    
    """
    

    # Create signs array
    signs = []; ix_d = { d : [] for d in depths }
    for sample in range(samples_per_depth):
        for depth in depths:
            ix_d[depth].append(len(signs))
            signs.append(-1 if depth % 2 == 0 else 1)
    return torch.tensor(signs)
    
PLAYERSTATE_LEN = len(PlayerState().to_array())

def traces_training_step(model, states, traces, signs):
    """ Perform traces training on a number of states.
    
    This calculates the (sum of squared) differences between the evaluation
    of the states and the average of the evaluation of their "traces".
    Minimising the difference should optimise for a model that correctly predicts
    the avraage long-term prospect of a game position.
    
    :param model: Game state evaluation model
    :param states: Game states
    :param traces: Sampling of game states down the line
    :param signs: Sign to apply to trace game states (positive if the same player
      is to move as in "states", negative if opposite player)
    :returns: Diff
    """
       
    state_count = states.shape[0]
    state_len = states.shape[1]
    samples = traces.shape[1]
    
    # Load traces + evaluate
    y_pred_2 = model(traces.reshape(state_count*samples, state_len))

    # Hard-code won and lost game states
    y_pred_2 = y_pred_2.reshape(state_count, samples)
    y_pred_2 = torch.where(traces[:,:,0] <= 0, 0, y_pred_2)
    y_pred_2 = torch.where(traces[:,:,PLAYERSTATE_LEN] <= 0, 1, y_pred_2)

    # Average
    y_pred_2 = y_pred_2 * signs
    y_pred_2 = torch.mean(y_pred_2, dim=1) - torch.mean(0.5 * signs) + 0.5

    # Evaluate states
    y_pred = model(states)

    # Determine difference
    loss = torch.nn.MSELoss()(y_pred_2, y_pred.reshape(state_count))
    return loss
