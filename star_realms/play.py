
import random
from functools import lru_cache

from . import state, mset, actions, cards, nn

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
            choices_ = choices.afilter(lambda card: card != () and card.cost <= cost)
            if not choices_.empty():
                return choices_
        if () in choices:
            return mset.MSet([()])
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

def finish_move_det(move, acquire_priority=[], cache=None, verbose=False, indent='', destructive=False):
    """
    Finishes move in a deterministic fashion.
    
    This means that we take all actions we typically would (play
    cards), but simply ignore any action that would require us
    to make a random choice.
    
    :param acquire_priority: If given, buy cards according to
      given priority order. Any cards with priority zero or worse
      (or not in dictionary) are going to get ignored.
    :param cache: If given, will be checked+updated for/with result
      at intermediate move states. Performance optimisation.
    :param verbose: Produce debug output
    :oaram destructive: Update move state passed in instead of making a copy
    :returns: Move state
    """

    # Check hash on initial position
    if cache is not None:
        # Salt with acquire priority
        if acquire_priority is None:
            acquire_hash = 0
        else:
            acquire_hash = hash(tuple(acquire_priority))
        # Calculate for initial position
        move_hash = hash(move) ^ acquire_hash
        if move_hash in cache:
            if verbose: print(indent+"(found in cache)")
            assert cache[move_hash].game.move == move.game.move+1
            if destructive: move.copy_from(cache[move_hash])
            return cache[move_hash]
        hashes = [move_hash]
        
    # Start going down the rabbit hole
    if not destructive:
        move = state.MoveState(move)
    while True:
        possible = move.possible_actions()
        # No choice left?
        if not possible:
            break
        # Find an action we want to take (deterministically)
        action = choice = None
        for act in possible:
            choices, is_rand = act.choices(move)
            # Automatic or card playing? Do it! We are assuming that for all such actions
            # the order we do them in does not matter.
            if act.automatic() or isinstance(act, actions.PlayCard):
                assert not is_rand or len(choices) == 1, act
                action = act
                choice = choices.get_ix(0)
                break
            # Acquire? This is deterministic as long as acquire priorities are distinct
            if isinstance(act, actions.Acquire):
                # Find choice that appears first in priority list
                for choice in acquire_priority:
                    if choice in choices:
                        action = act
                        break
            # Only choice, *and* forced? That's deterministic too
            if len(possible) == 1 and len(choices) == 1 and move.forced_actions:
                action = act
                choice = choices.get_ix(0)
        # Perform action - or not
        if action is not None:
            if verbose: print(indent, action.describe(choice))
            move.take_action(action, choice)
        # Simply ignore forced actions if they involve making a choice
        elif move.forced_actions:
            move.forced_actions = move.forced_actions[1:]
            continue
        else:
            # We're done!
            break
        # Check cache
        if cache is not None:
            move_hash = hash(move) ^ acquire_hash
            if move_hash in cache:
                if verbose: print(indent+"(found in cache)")
                assert cache[move_hash].game.move == move.game.move+1
                if destructive: move.copy_from(cache[move_hash])
                return cache[move_hash]
            hashes.append(move_hash)
    move.finish()
    # Add move state to cache
    if cache is not None:
        for h in hashes:
            cache[h] = move
    return move

def model_movestate_prob(model, move, game_prob_cache=None, verbose=False, indent=''):
    """
    Calculate model win propability for next player to move, with caching
    support
    
    :param model: Model to use
    :param move: Move state to assess
    :param finish_move_cache: Cache for finish_move_det
    :param verbose: Produce debug output
    :return: Propability (float)
    """
    if verbose >= 1:
        for line in move.describe().split('\n'):
            print(indent + line)
    # Check cache
    if game_prob_cache is not None:
        game_hash = hash(move.game)
        if game_hash in game_prob_cache:
            if verbose >= 1: print(indent + "->", game_prob_cache[game_hash], "(cached)")
            return game_prob_cache[game_hash]
    # Evaluate
    prob = nn.model_game_prob(model, move.game)
    if game_prob_cache is not None:
        game_prob_cache[game_hash] = prob
    if verbose >= 1: print(indent + "->", prob)
    return prob

def model_movestate_prob_det(model, move, acquire_priority=None, finish_move_cache=None, game_prob_cache=None,
                             verbose=False, indent='', destructive=False):
    """
    Calculate model win propability for next player to move, by
    finishing move deterministically first
    
    :param model: Model to use
    :param move: Move state to assess
    :param acquire_priority: Acquire priority for finish_move_det
    :param finish_move_cache: Cache for finish_move_det
    :param game_prob_cache: Cache for game propability evaluation
    :param verbose: Produce debug output
    :oaram destructive: Update move state passed in instead of making a copy
    :return: Propability (float)
    """
    # Finish move
    move = finish_move_det(move, acquire_priority, finish_move_cache, verbose>1, indent+"> ", destructive=destructive)
    # Calculate probability
    return model_movestate_prob(model, move, game_prob_cache, verbose>1, indent+"> ")

def trade_row_ratings(model, move0, finish_move_cache=None, game_prob_cache=None, verbose=False, indent=''):
    """
    Rates how much we would like to acquire what is currently available
    for purchase, according to an evaluation model.
    
    :param model: Model to use
    :param move0: Move state to assess 
    :param finish_move_cache: Cache for finish_move_det
    :param game_prob_cache: Cache for game propability evaluation
    :param verbose: Produce debug output
    :return: Propability (float)
    """
    ratings = {}
    
    # Get move end state if we do *not* buy anything (shouldn't do much,
    # but just in case)
    moves = move0.game.move
    move = state.MoveState(move0)
    base = model_movestate_prob_det(model, move, [],
        finish_move_cache, game_prob_cache, verbose, indent,
        destructive=True)
    if verbose:
        print(indent + "Base:", base)
    
    for card in list(move0.game.trade.values()) + [cards.Explorer]:
        # Now simulate buy by adding card to the discard pile.
        # Note that the move is finished, so the original player is
        # the opposing side now.
        assert move.opponent is move.game.move_players()[1]
        assert move.game.move == moves+1
        move.opponent.discard.add(card)
        # Also remove card from trade. For very good cards can
        # have a significant impact on the rating.
        if card != cards.Explorer:
            move.game.trade.remove(card)
        # Calculate rating
        ratings[card] = base - model_movestate_prob(model, move, finish_move_cache, verbose, indent)
        # Roll back changes
        move.opponent.discard.remove(card)
        if card != cards.Explorer:
            move.game.trade.add(card)
    return ratings
    
def ratings_to_priority(ratings, min_rating = 0):
    """ Converts a dictionary of ratings into a priotity list
    """
    return sorted([ card for card in ratings.keys() if ratings[card] >= min_rating ],
                    key = lambda card: -ratings[card] )

def make_greedy_move(move, model, take_action=True, finish_move_cache=None, game_prob_cache=None, verbose=0, indent='', skip_forced_rand = True):
    """Select a move guided by a model.
    
    This will select a move based on the assessment of a model function.
    Note that this does no tree search, instead just tries to greedily
    make some progress with respect to what finish_move_det sees as
    the end of the move.
    
    :param move: Move state
    :param model: Model function to use
    :param take_action: Actually take action?
    :param finish_move_cache: Cache for finish_move_det
    :param game_prob_cache: Cache for game propability evaluation
    :param verbose: Verbosity level (0,1,2)
    :returns: tuple (possible actions, action, choice, best_rating). Returns
        all "None" values if no move is possible. 
    """
    best_move = None
    
    # Make caches, if not passed in. We will likely be re-visiting states a lot here.
    if finish_move_cache is None:
        finish_move_cache = {}
    if game_prob_cache is None:
        game_prob_cache = {}

    # Will initialise lazily, in case we don't actually have a decision to make
    best_rating = None
    best_state = None
    nested_verbose = max(0, verbose - 1)
    nested_indent = indent + "> "

    possible = move.possible_actions()
    for act in possible:

        # Find and evaluate choices
        choices, is_rand = act.choices(move)
        rating_sum = 0
        rating_count = 0
        rand_continuations = {} # How to proceed for every random choice
        rand_states = {}

        # Only one random choice?
        if skip_forced_rand and move.forced_actions and len(possible) == 1 and (is_rand or len(choices.values()) == 1):
            best_move = [(act, choices.random())]
            best_rand = is_rand
            break

        for choice, count in choices.items():

            # Need to initialise trade ratings and best_rating?
            # We only do that here so obvious decisions don't need
            # us to go through all of this trouble
            if best_rating is None:
                
                # Rate trade row, convert into priority list
                trade_ratings = trade_row_ratings(model, move,
                    finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
                if verbose:
                    print("Trade ratings: ",
                        ", ".join([ "{}:{:.2f}".format(card.__name__, rating)
                                    for card, rating in trade_ratings.items() ]))
                trade_priority = ratings_to_priority(trade_ratings)

                # Determine the rating of doing nothing. Except if we
                # have a forced action, of course.
                if move.forced_actions:
                    if verbose: print(indent + '... forced, skipping evaluation')
                    best_rating = 1
                elif not move.hand.empty():
                    # Always force hand to be played entirely.
                    best_rating = 1
                else:
                    if verbose: print(indent + "Initial...")
                    best_rating = model_movestate_prob_det(model, move, [],
                        finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
                    if verbose: print(best_rating)
            
            # Determine rating of the choice
            if verbose: print(indent + act.describe(choice), "...")
            move2 = state.MoveState(move)
            move2.take_action(act, choice)
            
            # Forced action left? Execute recursively
            rating = None
            best_continuation = []
            if move2.forced_actions or any([act.automatic() for act in move2.possible_actions()]):
                _, best_continuation, rating = make_greedy_move(move2, model, True,
                    finish_move_cache, game_prob_cache, nested_verbose, nested_indent,
                    skip_forced_rand=False)
                if verbose: print(indent + "Continuation:", '; '.join([ act.describe(choice) for act, choice in best_continuation]))
            
            # Still need to re-evaluate rating?
            if rating is None:
                rating = model_movestate_prob_det(model, move2, trade_priority,
                    finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
            if verbose: print(indent, rating)

            # If not random, we take the best choice. Otherwise just sum up
            if not is_rand:
                if rating <= best_rating:
                    best_move = [(act, choice)] + best_continuation
                    best_state = move2
                    best_rating = rating
                    best_rand = False
            else:
                rand_continuations[choice] = best_continuation
                rand_states[choice] = move2
                rating_sum += count * rating
                rating_count += count
        # Random choice? Evaluate the average
        if is_rand and rating_sum / rating_count <= best_rating:
            choice = choices.random()
            best_move = [(act, choice)] + rand_continuations[choice]
            best_state = rand_states[choice]
            best_rating = rating_sum / rating_count
            best_rand = True

    # Best move is not to play?
    if best_move is None:
        return None, [], None
        
    # Otherwise take action
    if take_action:
        # Just copy state from known result (bit of a hack, admittedly)
        if best_state is not None:
            move.copy_from(best_state)
        else:
            for act, choice in best_move:
                move.take_action(act, choice)
    return possible, best_move, best_rating

def play_greedy_turn(gs, model, finish_move_cache=None, game_prob_cache=None, verbose=0):
    """Select a move guided by a model."""

    # Make caches, if not passed in. We will likely be re-visiting states even more here.
    if finish_move_cache is None:
        finish_move_cache = {}
    if game_prob_cache is None:
        game_prob_cache = {}
    
    # Play a move through
    move = state.MoveState(gs)
    while True:
        _, acts, rating = make_greedy_move(move, model, True,
            finish_move_cache, game_prob_cache, max(0,verbose-1))
        if not acts:
            if verbose: print("# Best: End turn")
            break
        else:
            if verbose: print("# Best: ", '; '.join([ act.describe(choice) for act, choice in acts]), rating)
    move.finish()
