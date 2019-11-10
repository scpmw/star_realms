
from . import mset
from .cards import *
from .actions import *

import numpy

# Turns on some internal consistency checks, at expense of performance
# Need to pull this into proper test cases eventually
CONSISTENCY_CHECKS = False

""" Persistent player state """
class PlayerState(object):
    def __init__(self, state=None):
        if state is None:
            self.authority = 50
            self.discard_count = 0 # Number of cards to discard at start of round
            self.deck = mset.MSet(8 * [Scout] + 2 * [Viper])
            self.top_of_deck = []
            self.discard = mset.MSet()
            self.bases = mset.MSet()
        else:
            self.authority = state.authority
            self.discard_count = state.discard_count
            self.deck = mset.MSet(state.deck)
            self.top_of_deck = list(state.top_of_deck)
            self.discard = mset.MSet(state.discard)
            self.bases = mset.MSet(state.bases)
    def have_outpost(self):
        return any([is_outpost(card) for card in self.bases.elements()])
    def to_array(self):
        return numpy.concatenate((
            [self.authority, self.discard_count],
            (self.deck+self.discard+mset.MSet(self.top_of_deck)).to_array(card_list),
            self.discard.to_array(card_list),
            self.bases.to_array(base_list)))
    @staticmethod
    def from_array(arr):
        state = PlayerState()
        state.authority = arr[0]
        state.discard_count = arr[1]
        ncard = len(card_list)
        state.deck = mset.MSet.from_array(card_list, arr[2:ncard+2])
        state.discard = mset.MSet.from_array(card_list, arr[ncard+2:2*ncard+2])
        state.bases = mset.MSet.from_array(base_list, arr[2*ncard+2:2*ncard+2+len(base_list)])
        state.deck = state.deck - state.discard
        return state
    def describe(self):
        return "%d auth, %s {%s | %s}" % (
            self.authority, mset.describe_mset(self.bases), mset.describe_mset(self.deck), mset.describe_mset(self.discard))
    def __hash__(self):
        return hash((self.authority, self.discard_count, self.deck, tuple(self.top_of_deck), self.discard, self.bases))
""" Persistent game state """
class GameState(object):
    def __init__(self, state=None):
        if state is None:
            self.move = 0
            self.player1 = PlayerState()
            self.player2 = PlayerState()
            self.cards = mset.MSet(card_set)
            self.trade = mset.MSet()
            self.scrapped = mset.MSet() # Needed?
        else:
            self.copy_from(state)
    def copy_from(self, state):
        self.move = state.move
        self.player1 = PlayerState(state.player1)
        self.player2 = PlayerState(state.player2)
        self.cards = mset.MSet(state.cards)
        self.trade = mset.MSet(state.trade)
        self.scrapped = mset.MSet(state.scrapped)
    def is_over(self):
        return self.player1.authority <= 0 or self.player2.authority <= 0
    def move_players(self):
        if self.move % 2 == 0:
            return self.player1, self.player2
        else:
            return self.player2, self.player1
    def next_player(self):
        return self.move % 2
    def to_array(self):
        p1, p2 = self.move_players()
        return numpy.concatenate((
            p1.to_array(), p2.to_array(), self.trade.to_array(card_list)
        ))
    @staticmethod
    def from_array(arr):
        gs = GameState()
        gs.move = 2
        ps_len = len(gs.player1.to_array())
        gs.player1 = PlayerState.from_array(arr[:ps_len])
        gs.player2 = PlayerState.from_array(arr[ps_len:2*ps_len])
        gs.trade = mset.MSet.from_array(card_list, arr[2*ps_len:])
        gs.cards = gs.cards - gs.trade
        for plr in [gs.player1, gs.player2]:
            gs.cards = gs.cards - plr.deck - plr.discard - plr.bases
        return gs
    def describe(self):
        return "Trade row: %s\nPlr1%s: %s\nPlr2%s: %s" % (
            mset.describe_mset(self.trade),
            "(m)" if self.next_player() == 0 else "", self.player1.describe(),
            "(m)" if self.next_player() == 1 else "", self.player2.describe())
    def __hash__(self):
        return hash((self.player1, self.player2, self.cards, self.trade))
""" Player move state """
class MoveState(object):
    def __init__(self, game):

        if isinstance(game, MoveState):
            self.copy_from(game, True)
        else:
            assert isinstance(game, GameState)
            self.game = game
            self.player, self.opponent = self.game.move_players()

            self.hand = mset.MSet() # Cards on hand
            self.played = mset.MSet() # Played ships, minus scrapped
            self.played_all = mset.MSet() # Played cards (for Blob World)
            self.factions = mset.MSet() # Number of played cards with faction

            self.trade = 0
            self.combat = 0
            self.actions = [PlayCard(), Acquire(), DestroyBase()]
            self.forced_actions = []

            # Special stuff
            self.acquire_onto_deck = 0 # Trade Federation
            self.per_ship_combat = 0 # Fleet HQ...

            # Draw trade row and hand cards
            for _ in range(5 - game.trade.count()):
                self.forced_actions.append(DrawTrade())
            draw_count = 3 if self.game.move == 0 else 5
            for _ in range(draw_count):
                self.forced_actions.append(DrawCard())
            discard_count = min(draw_count, self.player.discard_count)
            for _ in range(discard_count):
                self.forced_actions.append(DiscardCard())
            self.player.discard_count -= discard_count
            # Play bases
            for base in self.player.bases.elements():
                base.play(self, base)
                self.factions.add(base.faction)
    def copy_from(self, state, copy_gamestate=False):
        if copy_gamestate:
            self.game = GameState(state.game)
        else:
            self.game.copy_from(state.game)
        self.player, self.opponent = self.game.move_players()
        self.hand = mset.MSet(state.hand)
        self.played = mset.MSet(state.played)
        self.played_all = mset.MSet(state.played_all)
        self.factions = mset.MSet(state.factions)
        self.trade = state.trade
        self.combat = state.combat
        self.actions = list(state.actions)
        self.forced_actions = list(state.forced_actions)
        self.acquire_onto_deck = state.acquire_onto_deck
        self.per_ship_combat = state.per_ship_combat
    def __hash__(self):
        # Order doesn't matter on actions
        action_hashes = tuple([hash(act) for act in self.actions])
        forced_action_hashes = hash(tuple(self.forced_actions))
        return hash((self.game, self.hand, self.played, self.played_all,
                     self.factions, self.trade, self.combat,
                     action_hashes, forced_action_hashes,
                     self.acquire_onto_deck, self.per_ship_combat))
    def describe(self):
        desc = self.game.describe()
        desc += "\nTrade: {}, Combat: {}".format(self.trade, self.combat)
        if not self.hand.empty():
            desc += "\nHand: {}".format(mset.describe_mset(self.hand))
        if not self.played.empty():
            desc += "\nPlayed: {}".format(mset.describe_mset(self.played))
        if self.possible_actions():
            desc += "\n{} actions: {}".format(
                "Forced" if self.forced_actions else "Possible",
                ", ".join([str(act) for act in self.possible_actions()]))
        if self.acquire_onto_deck:
            desc += "(next {} ships acquired onto deck)".format(self.acquire_onto_deck)
        return desc
    def possible_actions(self):
        """ Returns currently possible actions. If actios are forced,
            other actions will not be returned until action gets taken. """
        # First check all forced actions, in given order. Skip impossible actions.
        while self.forced_actions:
            act = self.forced_actions[0]
            if act.possible(self):
                return [act]
            self.forced_actions = self.forced_actions[1:]
        # Otherwise return remaining actions
        possible = []
        for act in self.actions:
            if act.possible(self):
                if act.automatic():
                    return [act]
                possible.append(act)
            elif CONSISTENCY_CHECKS:
                choices, _ = act.choices(self)
                assert choices.empty(), "{} - possible {}, but choices {}!".format(act, act.possible(self), choices)
        return possible
    def force_action(self, act):
        """ Force an action to appear next. Takes priority over previously
            added forced actions, if any """
        self.forced_actions.insert(0,act)
    def take_action(self, act, choice):
        if not act.permanent():
            # Remove from forced actions
            if act in self.forced_actions:
                self.forced_actions.remove(act)
            else:
                self.actions.remove(act)
        # Apply
        act.apply(self, choice)
    def draw(self, card):
        """ Have player draw a card from their draw pile.
            Shuffles discard pile if necessary """
        # "Shuffle" discard pile if we can't draw
        if len(self.player.top_of_deck) > 0:
            assert self.player.top_of_deck[0] == card
            del self.player.top_of_deck[0]
        else:
            if self.player.deck.empty():
                self.player.deck = self.player.discard
                self.player.discard = mset.MSet()
            self.player.deck.remove(card)
        self.hand.add(card)
    def play(self, card):
        """ Plays a card the standard way from the hand. """
        self.hand.remove(card)
        self.played_all.add(card)
        if is_base(card):
            self.player.bases.add(card)
        else:
            self.played.add(card)
            self.combat += self.per_ship_combat
        card.play(self, card)
        self.factions.add(card.faction)
    def scrap_played(self, card):
        """ Scrap a played card (ship or base) """
        self.factions.remove(card.faction)
        if is_base(card):
            self.player.bases.remove(card)
        else:
            self.played.remove(card)
        self.game.scrapped.add(card)
    def scrap_trade(self, card):
        """ Scrap a card from the trade row """
        self.game.trade.remove(card)
        self.game.scrapped.add(card)
    def acquire(self, card):
        """ Buy a card from the trade row """
        if self.acquire_onto_deck > 0:
            self.player.top_of_deck.insert(0,card)
            self.acquire_onto_deck -= 1
        else:
            self.player.discard.add(card)
        if card != Explorer:
            self.game.trade.remove(card)
            self.force_action(DrawTrade())
    def finish(self):
        """ Finish move. Residual damage gets applied to opponent. """
        for act in self.forced_actions:
            if len(act.choices(self)[0]) > 0:
                assert False, "forced action left!"
        # Do damage, except if outposts are left
        if not self.opponent.have_outpost():
            self.opponent.authority -= self.combat
        self.player.discard = self.player.discard + self.played + self.hand
        # Advance to next move
        self.hand = self.played = mset.MSet()
        self.game.move += 1
        self.player, self.opponent = self.game.move_players()
