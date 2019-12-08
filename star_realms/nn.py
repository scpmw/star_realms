
import torch.nn
from . import state
import numpy

def make_model(nn_state = None, layout = (20,5,2), dropout=0.5, dropout_in=0.2):
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
        torch.nn.Linear(D_in, H1), torch.nn.Dropout(dropout_in), torch.nn.ReLU(),
        torch.nn.Linear(H1,   H2), torch.nn.Dropout(dropout), torch.nn.ReLU(),
        torch.nn.Linear(H2,   H2), torch.nn.Dropout(dropout), torch.nn.ReLU(),
        torch.nn.Linear(H2,   H3), torch.nn.Dropout(dropout), torch.nn.ReLU(),
        torch.nn.Linear(H3,   1), torch.nn.Sigmoid(),
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
    return model(torch.tensor(game_state.to_array(), dtype=torch.float)).item()

def _get_player_authority_index(player1):
    gs = state.GameState()
    if player1:
        gs.player1.authority = 9999
    else:
        gs.player2.authority = 9999
    return int(numpy.where(gs.to_array() == 9999)[0])
_P1_AUTHORITY_INDEX = _get_player_authority_index(True)
_P2_AUTHORITY_INDEX = _get_player_authority_index(False)

def model_game_prob_array(model, gs_array, device=None):
    """ Returns model propability that next player to move wins, using an
    array representation of the game state (see GameState.to_array)
    
    :param model: Model to use
    :param gs_array: Input game state(s) as arrays
    """
    
    # Check whether any player has lost
    gs_array = numpy.array(gs_array)
    sel_p1 = (gs_array[...,_P1_AUTHORITY_INDEX] <= 0)
    sel_p2 = (gs_array[...,_P2_AUTHORITY_INDEX] <= 0)
    sel_np = numpy.logical_not(sel_p1 | sel_p2)

    # Get results
    result = numpy.empty(gs_array.shape[0])
    if numpy.any(sel_np):
        result[sel_np] = model(torch.tensor(gs_array[sel_np], dtype=torch.float, device=device)).cpu().detach().numpy()[...,0]
    result[sel_p1] = 0
    result[sel_p2] = 1
    return result
