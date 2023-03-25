
import random
from functools import lru_cache
import numpy
import time

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
            if verbose: print(indent+"(found in cache: {})".format(hash(cache[move_hash])))
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
            if move.forced_actions and len(possible) == 1 and len(choices) == 1:
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
                if verbose: print(indent+"(found in cache: {})".format(hash(cache[move_hash])))
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
    # Trade row not filled?
    if move.game.trade.count() < 5:
        return evaluate_trade_row_draw(move, model, [], None, game_prob_cache, verbose, indent)
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
        finish_move_cache, game_prob_cache, verbose, indent)
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

# Determine indices in game state array
def _get_trade_row_array_index(card):
    gs = state.GameState()
    gs.trade.add(card, 9999)
    return int(numpy.where(gs.to_array() == 9999)[0])
_TRADE_ROW_INDEX = { card : _get_trade_row_array_index(card) for card in cards.card_list }
def _get_player2_discard_index(card):
    gs = state.GameState()
    gs.player2.discard.add(card, 9999)
    gs.player2.deck.add(card, 1)
    return int(numpy.where(gs.to_array() == 9999)[0])
_P2_DISCARD_INDEX = { card : _get_player2_discard_index(card) for card in cards.card_list }
def _get_player2_deck_index(card):
    gs = state.GameState()
    gs.player2.deck.add(card, 9999 - gs.player2.deck.count(card))
    return int(numpy.where(gs.to_array() == 9999)[0])
_P2_DECK_INDEX = { card : _get_player2_deck_index(card) for card in cards.card_list }

def trade_row_ratings_new(model, move0, finish_move_cache=None, game_prob_cache=None, verbose=False, indent=''):
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

    # Get move end state if we do *not* buy anything (shouldn't do much,
    # but just in case)
    moves = move0.game.move
    move = finish_move_det(move0, [], finish_move_cache, verbose>1, indent+"> ")
    game_states = [ move.game.to_array() ]

    # Modify game states to include every individual card removed from
    # the trade deck and added to the player's deck + discard pile
    buyable = list(move0.game.trade.values()) + [cards.Explorer]
    for card in buyable:
        game_state = numpy.array(game_states[0])
        game_state[_P2_DISCARD_INDEX[card]] += 1
        game_state[_P2_DECK_INDEX[card]] += 1
        if card != cards.Explorer:
            game_state[_TRADE_ROW_INDEX[card]] -= 1
        game_states.append(game_state)

    # Calculate rating difference        
    ratings = nn.model_game_prob_array(model, game_states)
    base = ratings[0]
    return { card : base - ratings[i+1] for i, card in enumerate(buyable) }
    
def ratings_to_priority(ratings, min_rating = 0):
    """ Converts a dictionary of ratings into a priotity list
    """
    return sorted([ card for card in ratings.keys() if ratings[card] >= min_rating ],
                    key = lambda card: -ratings[card] )

def evaluate_trade_row_draw(move, model, acquire_priority, finish_move_cache=None, game_prob_cache=None, verbose=0, indent=''):
    """ Evaluate all possibilities for trade row draws in given position.
    
    """

    # Get move end state - we might have enough money to buy something after all.
    move = finish_move_det(move, acquire_priority, finish_move_cache, verbose>1, indent+"> ")
    if move.game.is_over():
        return nn.model_game_prob(model, move.game)
    base_state = move.game.to_array()
    game_states = []; game_state_count = []
    game_states_buy = []; buy_indices = []
    
    # Check cache
    game_hash = hash(move.game)
    if game_prob_cache is not None and game_hash in game_prob_cache:
        if verbose:
            print(indent+"(trade row found in cache: {})".format(game_hash))
        return game_prob_cache[game_hash]

    # Determine number of cards to fill
    cards_to_fill = 5 - move.game.trade.count()
    if verbose:
        print(indent+"Cards to fill: ", cards_to_fill)
    t = time.time()
    if cards_to_fill == 0:
        return model_movestate_prob(model, move, game_prob_cache, verbose, indent)
        
    # Reduce number of possibilities a bit by using pretty likely filler cards
    for filler_card in [cards.BlobWheel, cards.ImperialFighter, cards.FederationShuttle]: 
        if cards_to_fill <= 2:
            break
        if filler_card in move.game.cards:
            move.game.cards.remove(filler_card)
            move.game.trade.add(filler_card)
            cards_to_fill -= 1
        
    # Construct states where this card is on the trade row
    for i, (card_set, set_count) in enumerate(move.game.cards.subsets(cards_to_fill, True)):
        game_state = numpy.array(base_state)
        for card, count in card_set.items():
            game_state[_TRADE_ROW_INDEX[card]] += count
        game_states.append(game_state)
        game_state_count.append(set_count)
        # Some cards we might alternatively buy. Note that because of the behaviour of
        # finish_move_det we will likely not have much buying power left.
        for card, count in card_set.items():
            if move.trade >= card.cost:
                game_state_buy = numpy.array(game_state)
                game_state_buy[_P2_DISCARD_INDEX[card]] += 1
                game_state_buy[_P2_DECK_INDEX[card]] += 1
                game_states_buy.append(game_state_buy)
                buy_indices.append(len(game_states)-1)
    t2 = time.time()
    
    # Evaluate
    ratings = nn.model_game_prob_array(model, game_states + game_states_buy)
    
    if verbose:
        print(indent + "Options: {}, {} buy ({:.2f} s + {:.2f} s)".format(len(game_states), len(game_states_buy), t2-t, time.time() - t2))
    
    # Determine where it is a good idea to buy
    buy_start_ix = len(game_states)
    for i,j in enumerate(buy_indices):
        if ratings[buy_start_ix+i] < ratings[j]:
            ratings[j] = ratings[buy_start_ix+i]
    
    # Calculate weighted sum
    average = numpy.sum(ratings[:buy_start_ix] * numpy.array(game_state_count)) / numpy.sum(game_state_count)
    if game_prob_cache is not None:
        game_prob_cache[game_hash] = average
    return average
           
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
    def lazy_init():

        # Rate trade row, convert into priority list
        #trade_ratings = trade_row_ratings(model, move,
        #    finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
        trade_ratings = trade_row_ratings_new(model, move,
            finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
        #for card in trade_ratings:
        #    assert(numpy.abs(trade_ratings[card] - trade_ratings_new[card]) < 1e-5)
        if verbose:
            print(indent + "Trade ratings: ",
                ", ".join([ "{}:{:.2f}".format(card.__name__, rating)
                            for card, rating in trade_ratings.items() ]))
        trade_priority = ratings_to_priority(trade_ratings)

        # Determine the rating of doing nothing. Except if we
        # have a forced action, of course.
        if move.forced_actions:
            if verbose: print(indent + '... forced {}, skipping evaluation'.format(move.forced_actions[0].describe()))
            start_rating = 1
        elif not move.hand.empty():
            # Always force hand to be played entirely.
            start_rating = 1
        else:
            if verbose: print(indent + "Initial...")
            start_rating = model_movestate_prob_det(model, move, [],
                finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
            if verbose: print(start_rating)
        return trade_priority, start_rating

    # Exactly a draw-card-onto-trade-row action left? Use a special (faster)
    # heuristic to deal with the many cases to consider.
    if not skip_forced_rand and len(move.forced_actions) == 1 and isinstance(move.forced_actions[0], actions.DrawTrade):
        trade_priority, best_rating = lazy_init()
        rating = evaluate_trade_row_draw(move, model, trade_priority, finish_move_cache, game_prob_cache, nested_verbose, nested_indent)
        act = move.forced_actions[0]
        choice = move.game.cards.random()
        if take_action:
            move.take_action(act, choice)
        if verbose: print(indent+"Average trade row draw rating: {}".format(rating))
        return [act], [(act, choice)], rating
    
    possible = move.possible_actions()
    for act in possible:

        # Find and evaluate choices
        choices, is_rand = act.choices(move)
        rating_sum = 0
        rating_count = 0
        rand_continuations = {} # How to proceed for every random choice
        rand_states = {}

        # Only one random choice? And allowed to skip evaluation? Not always the
        # case - if this is actually a nested choice, we actually might be able to
        # skip this via a different choice up-stream. So we need to return a rating
        # (and therefore evaluate all possibilities) to make an informed decision there.
        if skip_forced_rand and move.forced_actions and len(possible) == 1 and (is_rand or len(choices.values()) == 1):
            best_move = [(act, choices.random())]
            best_rand = is_rand
            break

        for choice, count in choices.items():

            # Need to initialise trade ratings and best_rating?
            # We only do that here so obvious decisions don't need
            # us to go through all of this trouble
            if best_rating is None:
                trade_priority, best_rating = lazy_init()

            # Determine rating of the choice
            if verbose: print(indent + act.describe(choice), "...")
            move2 = state.MoveState(move)
            move2.take_action(act, choice)
            
            # Forced action left? Execute recursively
            rating = None
            best_continuation = []
            if move2.forced_actions: # or any([act.automatic() for act in move2.possible_actions()]):
                _, best_continuation, rating = make_greedy_move(move2, model, True,
                    finish_move_cache, game_prob_cache, nested_verbose, nested_indent,
                    skip_forced_rand = skip_forced_rand and move.forced_actions)
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
        t = time.time()
        _, acts, rating = make_greedy_move(move, model, True,
            finish_move_cache, game_prob_cache, max(0,verbose-1))
        if not acts:
            if verbose: print("# Best: End turn {} ({:.2g} s)".format(rating, time.time()-t))
            break
        else:
            if verbose: print("# Best: {} {} ({:.2g} s)".format('; '.join([ act.describe(choice) for act, choice in acts]), rating, time.time()-t))
    move.finish()
