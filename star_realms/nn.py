
import torch
import torch.nn
from . import state
import numpy

_PLAYER_STATE_VECTOR_LEN = state.PlayerState().to_array().shape[0]

class PlayerStateModule(torch.nn.Module):
    """ Apply separate modules to parts of a game state
        
    Args:
        player_module: Module to apply to player state
        trade_module: Module to apply to trade row

    Shape:
        - Input: GameState state vector
        - Output: Vector of size "2xplayer_output + trade_output"

    Attributes:
        convs: Learnable convolutions. Shape (grid_size, grid_size, kernel_size, kernel_size, 2)
    """
    def __init__(self, player_module, trade_module):
        super(PlayerStateModule, self).__init__()
        self.player_module = player_module
        self.trade_module = trade_module

    def forward(self, inp):
        return torch.cat([
            self.player_module(inp[:,:_PLAYER_STATE_VECTOR_LEN]),
            self.player_module(inp[:,_PLAYER_STATE_VECTOR_LEN:_PLAYER_STATE_VECTOR_LEN*2]),
            self.trade_module(inp[:,_PLAYER_STATE_VECTOR_LEN*2:])
        ], dim=1)
    
    def extra_repr(self):
        return 'player_module={}, trade_module={}'.format(
            self.player_module, self.trade_module
        )

def make_model(nn_state = None, layout = (200,100,10,200), dropout=0.5, dropout_in=0.2):
    """
    Make neural network for evaluating game states
    """
    
    PS_inter, PS_out, TR_out, GS_inter2 = layout
    
    PS_in = state.PlayerState().to_array().shape[0]
    GS_in = state.GameState().to_array().shape[0]
    TR_in = GS_in - 2 * PS_in
    
    GS_inter = 2 * PS_out + TR_out
    
    model = torch.nn.Sequential(
        PlayerStateModule(
            torch.nn.Sequential(
                torch.nn.Linear(PS_in, PS_inter), torch.nn.Dropout(dropout_in), torch.nn.ReLU(),
                torch.nn.Linear(PS_inter, PS_out), torch.nn.Dropout(dropout), torch.nn.ReLU()
            ),
            torch.nn.Sequential(
                torch.nn.Linear(TR_in, TR_out),  torch.nn.Dropout(dropout_in), torch.nn.ReLU()
            )
        ),
        torch.nn.Linear(GS_inter, GS_inter2), torch.nn.Softplus(),
        torch.nn.Dropout(dropout), torch.nn.Linear(GS_inter2, 1), torch.nn.Sigmoid(),
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
        tensor = torch.tensor(gs_array[sel_np], dtype=torch.float, device=device)
        mod = model(tensor) 
        result[sel_np] = mod.cpu().detach().numpy()[...,0]
    result[sel_p1] = 0
    result[sel_p2] = 1
    return result
