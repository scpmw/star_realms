
import os.path
import re
import urllib.request

from . import mset, factions
from .actions import *

card_set = mset.MSet()
card_list = []
base_list = []
def card(faction, count=0, cost=0, base=None, outpost=None, url=None):
    assert base is not None or outpost is None
    def card_decorate(cls):
        # Append to lists
        cls.index = len(card_list)
        card_list.append(cls)
        if base is not None:
            base_list.append(cls)
        card_set.add(cls, count)
        # Set card properties
        cls.cost = cost
        cls.faction = faction
        cls.base = base
        cls.outpost = outpost
        image_base = os.path.join('imgs', '-'.join(re.split("([A-Z]+[a-z]*)", cls.__name__)[1::2]) if url is None else url)
        if cls.base is not None:
            cls.url = "https://www.starrealms.com/wp-content/uploads/2017/12/%s-300x214.jpg" % image_base
        else:
            cls.url = "https://www.starrealms.com/wp-content/uploads/2017/12/%s-214x300.jpg" % image_base
        cls.image = image_base + ".jpg"
        cls.__str__ = lambda: cls.__name__
        return cls
    return card_decorate
def is_ship(card):
    return card.base is None
def is_base(card):
    return card.base is not None
def is_outpost(card):
    return card.outpost is not None

@card(factions.neutral, cost=2)
class Explorer():
    @staticmethod
    def play(state, card):
        state.trade += 2
        state.actions.append(Scrap(card, Combat(2)))
@card(factions.neutral)
class Scout():
    @staticmethod
    def play(state, card):
        state.trade += 1        
@card(factions.neutral)
class Viper():
    @staticmethod
    def play(state, card):
        state.combat += 1

@card(factions.blob, count=1, cost=6)
class BattleBlob():
    @staticmethod
    def play(state, card):
        state.combat += 8
        state.actions.append(Ally(factions.blob, DrawCard()))
        state.actions.append(Scrap(card, Combat(4)))
@card(factions.blob, count=2, cost=2, url="BattlePod")
class BattlePod():
    @staticmethod
    def play(state, card):
        state.combat += 4
        state.force_action(ScrapTrade())
        state.actions.append(Ally(factions.blob, Combat(2)))
@card(factions.blob, count=1, cost=6)
class BlobCarrier():
    @staticmethod
    def play(state, card):
        state.combat += 7
        state.actions.append(Ally(factions.blob, AcquireShipFree()))
@card(factions.blob, count=2, cost=4)
class BlobDestroyer():
    @staticmethod
    def play(state, card):
        state.combat += 6
        state.actions.append(Ally(factions.blob, DestroyBaseFree()))
        state.actions.append(Ally(factions.blob, ScrapTrade()))
@card(factions.blob, count=3, cost=1)
class BlobFighter():
    @staticmethod
    def play(state, card):
        state.combat += 3
        state.actions.append(Ally(factions.blob, DrawCard()))
@card(factions.blob, base=5, count=3, cost=3)
class BlobWheel():
    @staticmethod
    def play(state, card):
        state.combat += 1
        state.actions.append(Scrap(card, Trade(3)))
@card(factions.blob, base=7, count=1, cost=8)
class BlobWorld():
    @staticmethod
    def play(state, card):
        state.actions.append(Choice(Combat(5), DrawPerPlayed(factions.blob)))
@card(factions.blob, count=1, cost=7)
class Mothership():
    @staticmethod
    def play(state, card):
        state.combat += 6
        state.force_action(DrawCard())
        state.actions.append(Ally(factions.blob, DrawCard()))
@card(factions.blob, count=2, cost=3)
class Ram():
    @staticmethod
    def play(state, card):
        state.combat += 5
        state.actions.append(Ally(factions.blob, Combat(2)))
        state.actions.append(Scrap(card, Trade(3)))
@card(factions.blob, count=1, cost=5, base=5, url="The-Hive")
class Hive():
    @staticmethod
    def play(state, card):
        state.combat += 3
        state.actions.append(Ally(factions.blob, DrawCard()))
@card(factions.blob, count=3, cost=2)
class TradePod():
    @staticmethod
    def play(state, card):
        state.trade += 3
        state.actions.append(Ally(factions.blob, Combat(2)))
@card(factions.machine_cult, count=1, cost=5)
class BattleMech():
    @staticmethod
    def play(state, card):
        state.combat += 4
        state.force_action(ScrapHandDiscard())
        state.actions.append(Ally(factions.machine_cult, DrawCard()))
@card(factions.machine_cult, count=2, cost=3, base=5, outpost=True)
class BattleStation():
    @staticmethod
    def play(state, card):
        state.actions.append(Scrap(card, Combat(5)))
@card(factions.machine_cult, count=1, cost=8, base=6, outpost=True)
class BrainWorld():
    @staticmethod
    def play(state, card):
        state.actions.append(ScrapDraw(2))
@card(factions.machine_cult, count=1, cost=6, base=5, outpost=True)
class Junkyard():
    @staticmethod
    def play(state, card):
        state.actions.append(ScrapHandDiscard())
@card(factions.machine_cult, count=1, cost=7, base=6, outpost=True)
class MachineBase():
    @staticmethod
    def play(state, card):
        state.force_action(DrawScrapHand())
@card(factions.machine_cult, count=1, cost=5, base=6, outpost=True)
class MechWorld():
    @staticmethod
    def play(state, card):
        for f in factions.factions:
            if f != factions.machine_cult:
                state.factions.add(f)
@card(factions.machine_cult, count=3, cost=2)
class MissileBot():
    @staticmethod
    def play(state, card):
        state.combat += 2
        state.force_action(ScrapHandDiscard())
        state.actions.append(Ally(factions.machine_cult, Combat(2)))
@card(factions.machine_cult, count=1, cost=6)
class MissileMech():
    @staticmethod
    def play(state, card):
        state.combat += 6
        state.force_action(DestroyBaseFree())
        state.actions.append(Ally(factions.machine_cult, DrawCard()))
@card(factions.machine_cult, count=2, cost=4)
class PatrolMech():
    @staticmethod
    def play(state, card):
        state.force_action(Choice(Trade(3), Combat(5)))
        state.actions.append(Ally(factions.machine_cult, ScrapHandDiscard()))
@card(factions.machine_cult, count=1, cost=4)
class StealthNeedle():
    @staticmethod
    def play(state, card):
        state.force_action(CopyShip(StealthNeedle))
@card(factions.machine_cult, count=3, cost=3)
class SupplyBot():
    @staticmethod
    def play(state, card):
        state.trade += 2
        state.force_action(ScrapHandDiscard())
        state.actions.append(Ally(factions.machine_cult, Combat(2)))
@card(factions.machine_cult, count=3, cost=2)
class TradeBot():
    @staticmethod
    def play(state, card):
        state.trade += 1
        state.force_action(ScrapHandDiscard())
        state.actions.append(Ally(factions.machine_cult, Combat(2)))

@card(factions.star_empire, count=1, cost=6)
class Battlecruiser():
    @staticmethod
    def play(state, card):
        state.combat += 5
        state.force_action(DrawCard())
        state.actions.append(Ally(factions.star_empire, OpponentDiscardCard()))
        state.actions.append(Scrap(card, Combined(DrawCard(), DestroyBaseFree())))
@card(factions.star_empire, count=2, cost=2)
class Corvette():
    @staticmethod
    def play(state, card):
        state.combat += 1
        state.force_action(DrawCard())
        state.actions.append(Ally(factions.star_empire, Combat(2)))
@card(factions.star_empire, count=1, cost=7)
class Dreadnaught():
    @staticmethod
    def play(state, card):
        state.combat += 7
        state.force_action(DrawCard())
        state.actions.append(Scrap(card, Combat(5)))
@card(factions.star_empire, count=1, cost=8, base=8)
class FleetHQ():
    @staticmethod
    def play(state, card):
        state.per_ship_combat += 1 # Pretty hacky
@card(factions.star_empire, count=3, cost=1)
class ImperialFighter():
    @staticmethod
    def play(state, card):
        state.combat += 2
        state.opponent.discard_count += 1
        state.actions.append(Ally(factions.star_empire, Combat(2)))
@card(factions.star_empire, count=3, cost=3)
class ImperialFrigate():
    @staticmethod
    def play(state, card):
        state.combat += 4
        state.opponent.discard_count += 1
        state.actions.append(Ally(factions.star_empire, Combat(2)))
        state.actions.append(Scrap(card, DrawCard()))
@card(factions.star_empire, count=2, cost=4, base=4, outpost=True)
class RecyclingStation():
    @staticmethod
    def play(state, card):
        state.actions.append(Choice(Trade(1), DiscardDraw(2)))
@card(factions.star_empire, count=1, cost=6, base=6, outpost=True)
class RoyalRedoubt():
    @staticmethod
    def play(state, card):
        state.combat += 3
        state.actions.append(Ally(factions.star_empire, OpponentDiscardCard()))
@card(factions.star_empire, count=2, cost=4, base=4, outpost=True)
class SpaceStation():
    @staticmethod
    def play(state, card):
        state.combat += 2
        state.actions.append(Ally(factions.star_empire, Combat(2)))
        state.actions.append(Scrap(card, Trade(4)))
@card(factions.star_empire, count=3, cost=3)
class SurveyShip():
    @staticmethod
    def play(state, card):
        state.trade += 1
        state.force_action(DrawCard())
        state.actions.append(Scrap(card, OpponentDiscardCard()))
@card(factions.star_empire, count=1, cost=5, base=4, outpost=True)
class WarWorld():
    @staticmethod
    def play(state, card):
        state.combat += 3
        state.actions.append(Ally(factions.star_empire, Combat(4)))

@card(factions.trade_federation, count=2, cost=4, base=4)
class BarterWorld():
    @staticmethod
    def play(state, card):
        state.actions.append(Choice(Authority(2), Trade(2)))
        state.actions.append(Scrap(card, Combat(5)))
@card(factions.trade_federation, count=1, cost=7, base=6)
class CentralOffice():
    @staticmethod
    def play(state, card):
        state.trade += 2
        state.acquire_onto_deck += 1
        state.actions.append(Ally(factions.trade_federation, DrawCard()))
@card(factions.trade_federation, count=1, cost=8)
class CommandShip():
    @staticmethod
    def play(state, card):
        state.player.authority += 4
        state.combat += 5
        state.force_action(DrawCard())
        state.force_action(DrawCard())
        state.actions.append(Ally(factions.trade_federation, DestroyBaseFree()))
@card(factions.trade_federation, count=3, cost=2)
class Cutter():
    @staticmethod
    def play(state, card):
        state.player.authority += 4
        state.trade += 2
        state.actions.append(Ally(factions.trade_federation, Combat(4)))
@card(factions.trade_federation, count=1, cost=5, base=5, outpost=True)
class DefenseCenter():
    @staticmethod
    def play(state, card):
        state.actions.append(Choice(Authority(3), Combat(2)))
        state.actions.append(Ally(factions.trade_federation, Combat(2)))
@card(factions.trade_federation, count=2, cost=3)
class EmbassyYacht():
    @staticmethod
    def play(state, card):
        state.player.authority += 3
        state.trade += 2
        # state.actions.append(BaseCount(2, Combined(DrawCard(), DrawCard())))
        # ... it seems the correct interpretation is that the bases must be there
        # when the embassy yacht gets played
        if state.player.bases.count() >= 2:
            state.force_action(DrawCard())
            state.force_action(DrawCard())
@card(factions.trade_federation, count=3, cost=1)
class FederationShuttle():
    @staticmethod
    def play(state, card):
        state.trade += 2
        state.actions.append(Ally(factions.trade_federation, Authority(4)))
@card(factions.trade_federation, count=1, cost=6)
class Flagship():
    @staticmethod
    def play(state, card):
        state.combat += 5
        state.force_action(DrawCard())
        state.actions.append(Ally(factions.trade_federation, Authority(5)))
@card(factions.trade_federation, count=2, cost=4)
class Freighter():
    @staticmethod
    def play(state, card):
        state.trade += 4
        state.actions.append(Ally(factions.trade_federation, AcquireOntoDeck()))
@card(factions.trade_federation, count=1, cost=6, base=6, outpost=True, url="Port-of-Call")
class PortOfCall():
    @staticmethod
    def play(state, card):
        state.trade += 3
        state.actions.append(Scrap(card, Combined(DrawCard(), DestroyBaseFree())))
@card(factions.trade_federation, count=1, cost=5)
class TradeEscort():
    @staticmethod
    def play(state, card):
        state.player.authority += 4
        state.combat += 4
        state.actions.append(Ally(factions.trade_federation, DrawCard()))
@card(factions.trade_federation, count=2, cost=3, base=4, outpost=True)
class TradingPost():
    @staticmethod
    def play(state, card):
        state.actions.append(Choice(Authority(1), Trade(1)))
        state.actions.append(Scrap(card, Combat(3)))

def load_cards():
    """ Load card faces from StarRealms website """
    for card in card_list:
        if not os.path.isfile(card.image):
            print("Loading %s..." % card.url)
            urllib.request.urlretrieve(card.url, card.image)

def describe_cardset_html(cardset, width="150px"):
    return "<p>" + " ".join(["<img src='%s' style='display:inline;width:%s'></img>" % (c.image, width)
                      for c in sorted(mset.MSet(cardset).elements(), key=lambda c: c.index)]) + "</p>"

