
import itertools

from . import mset

class Action(object):
    def __init__(self):
        self._hash = hash((self.__class__, tuple(sorted(self.__dict__.items(), key=lambda kv: kv[0]))))
        pass
    def choices(self, state):
        """ @return choices, random? """
        return mset.MSet([None]), True
    def permanent(self):
        """ Can be taken multiple times? """
        return False
    def possible(self, state):
        """ Possible to do? """
        return not self.choices(state)[0].empty()
    def automatic(self):
        """ Take action immediately without querying """
        return False
    def style(self):
        """ Style to use for buttons in UI """
        return ''
    def __str__(self):
        if self.__doc__ is not None:
            return self.__doc__.strip()
        else:
            return self.__class__.__name__
    def __hash__(self):
        return self._hash
        # return hash((self.__class__, tuple(sorted(self.__dict__.items(), key=lambda kv: kv[0]))))

class Noop(Action):
    """ Skip """
    def apply(self, state, choice):
        pass
    def automatic(self):
        return True
    def possible(self, state):
        return True
class EndMove(Action):
    """ End Move """ # Pseudo-Action
    def apply(self, state, choice):
        pass
    def style(self):
        return "info"
    def possible(self, state):
        return True
    
class Combined(Action):
    """ A number of actions in (forced) sequence """
    def __init__(self, *actions):
        self.actions = actions
        super().__init__()
    def choices(self, state):
        for act in enumerate(actions):
            choices, is_rand = act.choices(state)
            if not choices.empty():
                return choices, is_rand
        return MSet()
    def apply(self, state, choice):
        for i, act in enumerate(actions):
            if act.possible(state):
                act.apply(state, choice)
                break
        for act in self.actions[:i:-1]:
            state.force_action(act)
    def __str__(self):
        return " + ".join([str(act) for act in self.actions])
    def possible(self, state):
        return any([act.possible(state) for act in self.actions])

class Choice(Action):
    """ Choice between different actions """
    def __init__(self, *actions):
        self.actions = actions
        super().__init__()
    def choices(self, state):
        return mset.MSet([act for act in self.actions if act.possible(state)]), False
    def apply(self, state, choice):
        state.force_action(choice)
    def __str__(self):
        return "Choice" # choices will be described anyway
    def automatic(self):
        """ Take action immediately without querying """
        return all([act.automatic() for act in self.actions])
    def possible(self, state):
        return any([act.possible(state) for act in self.actions])

class Ally(Action):
    """ Action predicated on ally presence """
    def __init__(self, faction, action):
        self.faction = faction
        self.action = action
        super().__init__()
    def choices(self, state):
        if state.factions.count(self.faction) > 1:
            return self.action.choices(state)
        return mset.MSet(), True
    def apply(self, state, choice):
        self.action.apply(state, choice)
    def automatic(self):
        return self.action.automatic()
    def __str__(self):
        return "Ally %s: %s" % (self.faction, self.action.__str__())
    def style(self):
        return self.action.style()
    def possible(self, state):
        return state.factions.count(self.faction) > 1 and self.action.possible(state)

class BaseCount(Action):
    """ Action predicated on base count """
    def __init__(self, base_count, action):
        self.base_count = base_count
        self.action = action
        super().__init__()
    def choices(self, state):
        if state.player.bases.count() >= self.base_count:
            return self.action.choices(state)
        return mset.MSet(), True
    def apply(self, state, choice):
        self.action.apply(state, choice)
    def automatic(self):
        return self.action.automatic()
    def __str__(self):
        return "Basecount %d: %s" % (self.base_count, self.action.__str__())    
    def style(self):
        return self.action.style()
    def possible(self, state):
        return state.player.bases.count() >= self.base_count and self.action.possible(state)

from . import cards
        
class Acquire(Action):
    """ Acquire card """
    def choices(self, state):
        options = [cards.Explorer] + list(state.game.trade.values())
        return mset.MSet( [ card for card in options
                       if card.cost <= state.trade ] ), False
    def apply(self, state, choice):
        state.acquire(choice)
        state.trade -= choice.cost
    def possible(self, state):
        if state.trade >= 2:
            return True
        return super().possible(state)
    def permanent(self):
        return True

class AcquireOntoDeck(Action):
    """ Next acquired card on top of deck """
    def apply(self, state, choice):
        state.acquire_onto_deck += 1
    def style(self):
        return 'success'
    def automatic(self):
        return True
    def possible(self, state):
        return True

class AcquireShipFree(Action):
    """ Acquire ship for free """
    def choices(self, state):
        return mset.MSet( [ card for card in state.game.trade.values()
                            if cards.is_ship(card) ] ), False
    def apply(self, state, choice):
        state.acquire(choice)
    def style(self):
        return 'success'
    def possible(self, state):
        return any([ cards.is_ship(card) for card in state.game.trade.values() ])
class DrawCard(Action):
    """ Draw to hand """
    def choices(self, state):
        if len(state.player.top_of_deck) > 0:
            return mset.MSet([state.player.top_of_deck[0]]), "DrawCard"
        elif state.player.deck.empty():
            return state.player.discard, "DrawCard"
        else:
            return state.player.deck, "DrawCard"
    def apply(self, state, choice):
        state.draw(choice)
    def style(self):
        return 'success'
    def possible(self, state):
        return state.player.top_of_deck or \
            not state.player.discard.empty() or \
            not state.player.deck.empty()
class DiscardCard(Action):
    """ Discard a card """
    def choices(self, state):
        return state.hand, False
    def apply(self, state, choice):
        state.hand.remove(choice)
        state.player.discard.add(choice)
    def style(self):
        return 'danger'
    def possible(self, state):
        return not state.hand.empty()

class DrawTrade(Action):
    """ Draw to trade row """
    def choices(self, state):
        return state.game.cards, "DrawTrade"
    def apply(self, state, choice):
        state.game.trade.add(choice)
        state.game.cards.remove(choice)
    def possible(self, state):
        return not state.game.cards.empty()
class PlayCard(Action):
    """ Play card """
    def choices(self, state):
        return state.hand, False
    def apply(self, state, choice):
        state.play(choice)
    def permanent(self):
        return True
    def possible(self, state):
        return not state.hand.empty()
class PlayAll(Action):
    """ Play all cards """
    def apply(self, state, choice):
        # Stop if forced actions are added, as this is when
        # the "PlayCard" action would become unavailable
        while not state.hand.empty() and len(state.forced_actions) == 0:
            print(state.hand.get_ix(0))
            state.play(state.hand.get_ix(0))
    def permanent(self):
        return True
    def choices(self, state):
        if state.hand.empty():
            return mset.MSet(), True
        else:
            return super(PlayAll, self).choices(state)
    def possible(self, state):
        return not state.hand.empty()
    def style(self):
        return 'info'

class Scrap(Action):
    """ Scrap played card """
    def __init__(self, card, action):
        self.card = card
        self.action = action
        super().__init__()
    def choices(self, state):
        return self.action.choices(state)
    def apply(self, state, choice):
        state.scrap_played(self.card)
        self.action.apply(state, choice)
    def __str__(self):
        return "Scrap %s to %s" % (self.card.__str__(), self.action)
    def style(self):
        return 'danger'
    def possible(self, state):
        return self.action.possible(state)
class ScrapTrade(Action):
    """ Scrap from trade row """
    def choices(self, state):
        return state.game.trade, False
    def apply(self, state, choice):
        state.scrap_trade(choice)
        state.force_action(DrawTrade())
    def possible(self, state):
        return not state.game.trade.empty()
class ScrapHand(Action):
    """ Scrap from hand """
    def choices(self, state):
        return state.hand, False
    def apply(self, state, choice):
        state.hand.remove(choice)
        state.game.scrapped.add(choice)
    def style(self):
        return 'danger'
    def possible(self, state):
        return not state.hand.empty()        
class ScrapHandDiscard(Action):
    """ Scrap hand/discard pile """
    def choices(self, state):
        return state.hand + state.player.discard, False
    def possible(self, state):
        return not state.hand.empty() or not state.player.discard.empty()
    def apply(self, state, choice):
        if choice in state.player.discard:
            state.player.discard.remove(choice)
        else:
            state.hand.remove(choice)
        state.game.scrapped.add(choice)
    def style(self):
        return 'danger'
class ScrapDraw(ScrapHandDiscard):
    """ Scrap+draw card """
    def __init__(self, count=1):
        self.count = count
        super().__init__()
    def apply(self, state, choice):
        super(ScrapDraw, self).apply(state, choice)
        state.force_action(DrawCard())
        if self.count > 1:
            state.force_action(Choice(Noop(), ScrapDraw(self.count-1)))
    def style(self):
        return 'danger'
class DiscardDraw(DiscardCard):
    """ Discard+draw card """
    def __init__(self, count=1):
        self.count = count
        super().__init__()
    def apply(self, state, choice):
        super(DiscardDraw, self).apply(state, choice)
        state.force_action(DrawCard())
        if self.count > 1:
            state.force_action(Choice(Noop(), DiscardDraw(self.count-1)))
    def style(self):
        return 'danger'
class DrawScrapHand(DrawCard):
    """ Draw Card, then scrap from hand """
    def apply(self, state, choice):
        super(DrawScrapHand, self).apply(state, choice)
        state.force_action(ScrapHand())

class DestroyBase(Action):
    def choices(self, state):
        if state.opponent.have_outpost():
            return mset.MSet([ card for card in state.opponent.bases.values()
                               if cards.is_outpost(card) and card.base <= state.combat ]), False
        else:
            return mset.MSet([ card for card in state.opponent.bases.values()
                               if card.base <= state.combat ]), False
    def apply(self, state, choice):
        state.combat -= choice.base
        state.opponent.bases.remove(choice)
        state.opponent.discard.add(choice)
    def permanent(self):
        return True

class DestroyBaseFree(Action):
    def choices(self, state):
        if state.opponent.have_outpost():
            return mset.MSet([ card for card in state.opponent.bases.values()
                               if cards.is_outpost(card) ]), False
        return state.opponent.bases, False
    def apply(self, state, choice):
        state.opponent.bases.remove(choice)
        state.opponent.discard.add(choice)
    def style(self):
        return 'success'
    def possible(self, state):
        return not state.opponent.bases.empty()

class Trade(Action):
    def __init__(self, trade):
        self.trade = trade
        super().__init__()
    def apply(self, state, _choice):
        state.trade += self.trade
    def automatic(self):
        return True
    def __str__(self):
        return "+%d Trade" % self.trade
    def possible(self, state):
        return True
class Authority(Action):
    def __init__(self, authority):
        self.authority = authority
        super().__init__()
    def apply(self, state, _choice):
        state.player.authority += self.authority
    def automatic(self):
        return True
    def __str__(self):
        return "+%d Authority" % self.authority
    def possible(self, state):
        return True
class Combat(Action):
    def __init__(self, combat):
        self.combat = combat
        super().__init__()
    def apply(self, state, _choice):
        state.combat += self.combat
    def automatic(self):
        return True
    def __str__(self):
        return "+%d Combat" % self.combat
    def possible(self, state):
        return True

class DrawPerPlayed(Action):
    def __init__(self, faction):
        self.faction = faction
        super().__init__()
    def apply(self, state, _choice):
        for card, n in state.played_all.items():
            if card.faction == self.faction:
                for _ in range(n):
                    state.force_action(DrawCard())
    def style(self):
        return 'success'
    def possible(self, state):
        return True
class CopyShip(Action):
    def __init__(self, card):
        # Save back what ship is doing the copying for
        # adjusting scrap-ship actions
        self.card = card
        super().__init__()
    def choices(self, state):
        return mset.MSet([ ship for ship in state.played.values() if ship != cards.StealthNeedle ]), False
    def apply(self, state, choice):
        choice.play(state, self.card)
        state.factions.add(choice.faction)
        # Note we do not add to played_all, assuming copying a Blob card
        # would not count for drawing an additional card with Blob World.
    def possible(self, state):
        return any([ ship != cards.StealthNeedle for ship in state.played.values() ])
        
class OpponentDiscardCard(Action):
    """ Opponent discards a card """
    def apply(self, state, _choice):
        state.opponent.discard_count += 1
    def automatic(self):
        return True
    def possible(self, state):
        return True
