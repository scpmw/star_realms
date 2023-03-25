
import unittest
import collections
import random
import numpy
import itertools
import functools
import operator
import math

class MSet(object):
    """ Utility multi set class with a bunch of special capabilities for
        handling drawing elems and similar game-specific stuff. There's
        likely something like this somewhere in a library, but I don't
        feel like looking for it. """
    
    def __init__(self, elems={}, key_map=None):
        """
        Construct a multi-set
        :param elems: Elements, either as list, dictionary or MSet
        :param key_map: Pair of maps from elements to indices, and the inverse
           All indices must be unique and 0 <= ix < len(key_map[0])
        """
        if isinstance(elems, MSet):
            self._hash = elems._hash
            self._key_map = elems._key_map
            if elems._key_map is None:
                assert isinstance(elems._elems, collections.Counter)
                self._elems = collections.Counter(elems._elems)
            else:
                assert isinstance(elems._elems, numpy.array)
                self._elems = numpy.array(elems._elems)
        elif isinstance(elems, list):
            self._key_map = key_map
            if key_map is None:
                self._elems = collections.Counter()
            else:
                self._elems = numpy.zeros(len(key_map[0]), dtype=int)
            self._hash = hash(MSet)
            for elem in elems:
                self.add(elem)
        else:
            #assert all([v > 0 for v in elems.values()])
            self._key_map = key_map
            if key_map is None:
                self._elems = collections.Counter(elems)
                h = hash(MSet)
                for kv in self._elems.items():
                    h ^= hash(kv)
                self._hash = h
            else:
                self._elems = numpy.zeros(len(key_map[0]), dtype=int)
                self._hash = hash(MSet)
                for elem, count in elems:
                    self.add(elem, count)
    def _to_ix(self, elem):
        return self._key_map[0][elem]
    def _from_ix(self, ix):
        return self._key_map[1][ix]
    def __repr__(self):
        return "MSet(%s)" % repr(dict(self._elems))
    def __contains__(self, elem):
        return elem in self._elems
    def __eq__(self, other):
        return self._elems == other._elems
    def __len__(self):
        return len(self._elems)
    def add(self, elem, count=1):
        assert count >= 0
        if count == 0:
            return
        if self._key_map is not None:
            old = self._elems[self._to_ix(elem)]
            self._elems[self._to_ix(elem)] = old + count
            self._hash ^= hash((elem, old)) ^ hash((elem, old+count))
        else:
            if elem in self._elems:
                old = self._elems[elem]
                self._elems[elem] = old + count
                self._hash ^= hash((elem, old)) ^ hash((elem, old+count))
            else:
                self._elems[elem] = count
                self._hash ^= hash((elem, count))
    def remove(self, elem):
        if self._key_map is not None:
            old = self._elems[self._to_ix(elem)]
            self._elems[self._to_ix(elem)] = old - 1
            self._hash ^= hash((elem, old)) ^ hash((elem, old-1))
        else:
            old = self._elems[elem]
            assert old > 0
            if old == 1:
                del self._elems[elem]
                self._hash ^= hash((elem, old))
            else:
                self._elems[elem] = old - 1
                self._hash ^= hash((elem, old)) ^ hash((elem, old-1))
    def afilter(self, f):
        return MSet({ k:v for k,v in self._elems.items() if f(k) })
    def __add__(self, mset):
        return MSet(self._elems + mset._elems)
    def __sub__(self, mset):
        return MSet(self._elems - mset._elems)
    def unique_count(self):
        if self._key_map is None:
            return len(self._elems)
        else:
            return numpy.sum(self._elems != 0)
    def empty(self):
        if self._key_map is None:
            return self.unique_count() == 0
        else:
            return numpy.max(self._elems) == 0
    def count(self, elem=None):
        if elem is not None:
            if self._key_map is None:
                return self._elems.get(elem, 0)
            else:
                return self._elems[self._to_ix(elem)]
        else:
            if self._key_map is None:
                return sum(self._elems.values())
            else:
                return numpy.sum(self._elems)
    def values(self):
        """ Return iterator of unique values"""
        
        return self._elems.keys()
    def items(self):
        return self._elems.items()
    def elements(self):
        """ Return all values as a list"""
        return self._elems.elements()

    def get_ix(self, n, default=None):
        if self._key_map is None:
            for elem, c in self._elems.items():
                if n < c:
                    return elem
                n -= c
        else:
            for i, c in enumerate(self._elems):
                if n < c:
                    return self._from_ix(i)
                n -= c
        return default
    def random(self):
        """ Get a random element """
        return self.get_ix(random.randrange(self.count()))
    def draw(self):
        """ Draw a random element """
        elem = self.random()
        self.remove(elem)
        return elem
    def to_array(self, elems):
        return numpy.array([self._elems.get(elem,0) for elem in elems])
    @staticmethod
    def from_array(elems, arr):
        assert(len(elems) == len(arr))
        return MSet({e:int(v) for e,v in zip(elems, arr) if v != 0})
    def to_list(self):
        out = []
        for elem, c in self._elems.items():
            for i in range(c):
                out.append(elem)
        return out
    def __hash__(self):
        # Yeah, this class is not immutable - be careful, alright?
        # On the bright side, this is pretty fast.
        return self._hash
    def subsets(self, r, subset_count=False):
        """ Returns all sub-multisets of this multiset that have the given size

        :param r: Subset size
        :param subet_count: Return how many ways there are to get the given subset
        :returns: Iterator over subsets
        """
        assert r >= 0
        # Handle empty set
        if r == 0:
            if subset_count:
                yield MSet(), 1
            else:
                yield MSet()
            return
        l = len(self._elems)
        # Implementation inspired by itertools.combinations
        indices = r * [0]
        vals = list(self.values())
        counts = list(self._elems.values())
        def binom(n, k):
            return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
        while True:
            # Construct a valid index tuple by increasing any 
            # indices that repeat more often than allowed
            ix = indices[0]; repeats = 1
            if ix >= l:
                return
            elems = {}
            count = 1
            impossible = False
            for i in range(1,r):
                ix2 = max(ix, indices[i])
                # A repeat?
                if ix == ix2:
                    # Too many repeats? Set to next valid index
                    if repeats == counts[ix]:
                        elems[vals[ix]] = repeats
                        ix2 += 1
                        repeats = 1
                        # If we run over the length we will need to backtrack
                        if ix2 >= l:
                            impossible = True
                            break
                    else:
                        # Just count repeat
                        repeats += 1
                else:
                    # Determine number of possible ways we can get
                    # this number of repeats given the number of existing
                    # elements
                    count *= binom(counts[ix], repeats)
                    elems[vals[ix]] = repeats
                    repeats = 1
                ix = indices[i] = ix2
            if not impossible:
                # Add last index
                count *= binom(counts[ix], repeats)
                elems[vals[ix]] = repeats
                # Yield a new multi set
                if subset_count:
                    yield MSet(elems), count
                else:
                    yield MSet(elems)
            # Find index to increment
            found = False
            for i in range(r):
                if indices[r-i-1]+1 >= l:
                    indices[r-i-1] = 0
                else:
                    indices[r-i-1] += 1
                    found = True
                    break
            # Done
            if not found:
                return

def mset_product(*msets):
    """ Cartesion product of multisets """
    return MSet({tuple([k for k,_ in kvs]) : functools.reduce(operator.mul, [v for _,v in kvs]) for
                 kvs in itertools.product(*[MSet(ms)._elems.items() for ms in msets])})

def describe_mset(choices):
    return ", ".join(["%dx%s" % (n,c.__str__()) for c,n in MSet(choices).items()])

class TestMSet(unittest.TestCase):
    def setUp(self):
        self.key_map = None
    def test_init(self):
        cs = MSet(key_map=self.key_map); cs.add(1)
        assert not cs.empty()
        assert cs.unique_count() == 1
        assert cs.draw() == 1
        assert cs.empty()
        assert cs.unique_count() == 0
        assert cs == MSet()
        assert cs == MSet(key_map=self.key_map)
    def test_reproduce_elems(self):
        elems = [ random.randrange(100) for _ in range(200) ]
        stack = MSet(elems, key_map=self.key_map)
        assert stack.count() == 200
        self.assertEqual(stack.unique_count(), len(set(elems)))
        assert not stack.empty()
        for i, elem in enumerate(elems):
            self.assertEqual(hash(stack), hash(MSet(stack._elems)))
            self.assertEqual(hash(stack), hash(MSet(stack._elems, key_map=self.key_map)))
            assert elem in stack
            self.assertEqual(stack.count(), 200 - i)
            stack.remove(elem)
        for elem in elems:
            assert elem not in stack
        assert stack.empty()
        for i, elem in enumerate(elems):
            self.assertEqual(stack.count(), 2 * i)
            stack.add(elem); stack.add(elem)
            assert elem in stack
            self.assertEqual(hash(stack), hash(MSet(stack._elems)))
            self.assertEqual(hash(stack), hash(MSet(stack._elems, key_map=self.key_map)))
        self.assertEqual(stack.unique_count(), len(set(elems)))
        for i, elem in enumerate(elems):
            stack.remove(elem)
        self.assertEqual(stack, MSet(elems))
        self.assertEqual(stack, MSet(elems, key_map=self.key_map))
        self.assertEqual(hash(stack), hash(MSet(elems)))
        self.assertEqual(hash(stack), hash(MSet(elems, key_map=self.key_map)))
    def test_product(self):
        stack1 = MSet([1,1,2])
        assert mset_product(stack1).count() == stack1.count()
        stack2 = MSet([1,2,2,2,3])
        prod = MSet({(1,1):2, (1,2):6, (1,3):2, (2,1): 1, (2,2): 3, (2,3): 1})
        assert mset_product(stack1, stack2) == MSet(prod)
        prod = MSet({(1, 1, 1): 4, (1, 1, 2): 12, (1, 1, 3): 4, (1, 2, 1): 2, (1, 2, 2): 6, (1, 2, 3): 2,
                     (2, 1, 1): 2, (2, 1, 2): 6,  (2, 1, 3): 2, (2, 2, 1): 1, (2, 2, 2): 3, (2, 2, 3): 1})
        assert mset_product(stack1, stack1, stack2) == MSet(prod)
    def test_to_array(self):
        prod = MSet({'a': 2, 'b':5, 'd': 4})
        for elems, expected in [ (['a','b','c','d'], [2,5,0,4]),
                                 (['d','c','a','b'], [4,0,2,5]) ]:
            assert (prod.to_array(elems) == expected).all()
            assert MSet.from_array(elems, prod.to_array(elems)) == prod
    def test_subsets(self):
        for i in range(6):
            assert list(MSet(range(i)).subsets(i, True)) == [(MSet(range(i)), 1)]
            for j in range(i+1):
                assert list(MSet(range(i)).subsets(j, True)) == \
                    list([(MSet(comb), 1) for comb in itertools.combinations(range(i), j) ])
                for elems in itertools.product([1,2,3],repeat=i):
                    assert MSet([MSet(es) for es in itertools.combinations(elems, j)]) == \
                        MSet(dict(MSet(elems).subsets(j, True)))

class TestMSetKeyMap(TestMSet):
    def setUp(self):
        self.key_map = ({ n : n for n in range(200) }, { n : n for n in range(200) })
        