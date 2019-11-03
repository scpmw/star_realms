
import random
from . import state, mset, actions, cards

def describe_action(action, move):
    choices, is_rand = action.choices(move)
    if len(choices) == 1 and choices.get_ix(0) is None:
        return action.__str__()
    elif is_rand:
        return "%s{%s}" % (action, mset.describe_mset(choices))
    else:
        return "%s[%s]" % (action, mset.describe_mset(choices))
    
def heuristic_actions(acts, move, ab):
    """ Remove 'stupid' actions from the action pool"""
    # Don't scrap anything but Explorers
    no_scrap = list([a for a in acts if not isinstance(a, actions.Scrap) or a.card == cards.Explorer])
    # Greatly prefer playing cards or otherwise getting resources before doing anything else (especially purchasing)
    play_actions = 5*list([a for a in acts if isinstance(a, actions.PlayCard) or isinstance(a, actions.Trade) or isinstance(a, actions.Combat)])
    return play_actions + no_scrap

def heuristic_choices(action, choices, rand, move, ab):
    """ Filter and re-priotitise choices roughly """
    
    # Safety: Can't optimise random choices!
    if rand: return choices
    
    # Always scrap/discard one of the cheapest cards available
    if isinstance(action, actions.ScrapHand) or isinstance(action, actions.ScrapHandDiscard) or \
       isinstance(action, actions.DiscardCard):
        for cost in range(10):
            choices_ = choices.afilter(lambda card: card.cost <= cost)
            if not choices_.empty():
                return choices_
    # Preferably acquire expensive cards
    elif isinstance(action, actions.Acquire) or isinstance(action, actions.AcquireShipFree) or \
         isinstance(action, actions.DestroyBaseFree):
        factions = mset.MSet()
        for card, count in (move.player.discard + move.player.deck + move.hand + move.played + move.player.bases).items():
            factions.add(card.faction, count)
        return mset.MSet({card : (1 if card==cards.Explorer else card.cost**2 * (1+factions.count(card.faction)))
                     for card in choices.values() })
    elif isinstance(action, actions.Choice) and ab:
        return mset.MSet({choice : (1 if isinstance(choice, actions.Noop) else 10) for choice in choices.values() })
    return choices
    
def make_random_move(move, ab=False, take_action=True):
    # Get possible actions, apply heuristic
    possible = list([a for a in move.possible_actions() if a.possible(move)])
    acts = heuristic_actions(possible, move, ab)
    if len(acts) == 0:
        return None
    # Choose action
    action = acts[random.randrange(len(acts))]
    # Determine choices
    choices, is_rand = action.choices(move)
    assert isinstance(choices, mset.MSet), action
    assert choices, "{} - possible {}, but choices {}!".format(action, action.possible(move), choices)
    # Choose randomly
    choice = heuristic_choices(action, choices, is_rand, move, ab).random()
    if take_action:
        move.take_action(action, choice)
    return possible, action, choices, choice

def play_random_turn(gs, ab=False):
    move = state.MoveState(gs)
    while make_random_move(move, ab) is not None:
        pass
    move.finish()    

def random_win_prob(start_state, game_count, ab_switch=False):
    """ Returns statistical win percentage with random play for next player to move """
    win_count = 0
    player = start_state.next_player()
    for j in range(game_count):
        gs = state.GameState(start_state)
        for i in range(100):
            ab = ((i % 2 == 0) != ab_switch)
            play_random_turn(gs, ab)
            if gs.player1.authority <= 0:
                if player == 1:
                    win_count += 1
                break
            if gs.player2.authority <= 0:
                if player == 0:
                    win_count += 1
                break
    return(win_count / game_count)
