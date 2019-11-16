
import torch.nn
from . import state
import numpy

def make_model(nn_state = None, layout = (25,5,2), dropout=0.3):
    """
    Make neural network for evaluating game states
    """
    D_in = state.GameState().to_array().shape[0]
    H1, H2, H3 = layout
    H1 *= 50; H2 *= 50; H3 *= 50
    # Three levels with a large first level, and a "softplus" final layer.
    # Idea is that we need to keep track of a lot of individual card
    # interactions, so the first layer must be fairly large. From there
    # we need to identify and weight a lot of possible strategic options,
    # for which medium-sized ReLU + Softplus seems to make sense.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1), torch.nn.Dropout(dropout), torch.nn.ReLU(),
        torch.nn.Linear(H1,   H2), torch.nn.Dropout(dropout), torch.nn.ReLU(),
        torch.nn.Linear(H2,   H3), torch.nn.Dropout(dropout), torch.nn.Softplus(),
        torch.nn.Linear(H3,   1)
    )
    if nn_state is not None:
        model.load_state_dict(nn_state)
    return model

min_prob = 1 / 1000
max_prob = 1 - min_prob

def prob_to_value(prob):
    return -numpy.log(1 / numpy.maximum(min_prob, numpy.minimum(max_prob, prob)) - 1)
def value_to_prob(val):
    return 1 / (numpy.exp(-val) + 1)
    
def model_game_prob(model, game_state):
    """ Returns model propability that next player to move wins.
    
    :param model: Model to use
    :param game_state: Input game state
    """
    
    # Check whether game is over
    p1, p2 = game_state.move_players()
    if p1.authority <= 0:
        return 0
    if p2.authority <= 0:
        return 1
        
    # Otherwise check model
    return model_game_prob_array(model, game_state.to_array())

def model_game_prob_array(model, gs_array):
    """ Returns model propability that next player to move wins, using an
    array representation of the game state (see GameState.to_array)
    
    :param model: Model to use
    :param gs_array: Input game state(s) as arrays
    """
    result = model(torch.tensor(gs_array, dtype=torch.float))
    return value_to_prob(result.detach().numpy()[...,0])