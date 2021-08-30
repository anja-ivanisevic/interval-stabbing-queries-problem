import matplotlib.pyplot as plt
import random
from collections import Counter
import time
import math

NEG_INF = -10000
POS_INF = 10000

intervalNames = {}
names = []

class Interval:
    def __init__(self, leftEdge = None, rightEdge =  None):
        self.leftEdge = leftEdge
        self.rightEdge = rightEdge




class TreeNodeIntervalTree:
    def __init__(self, leftEdge = None, rightEdge = None):
        self.left = None
        self.right = None
        self.value = Interval(leftEdge, rightEdge)
        self.max = rightEdge
        self.height = 1

# Implementacija Intervalnog stabla
class IntervalTree(object):
    def searchIntervalTree(self, tree, point, S):
        if S is None:
            S = set()

        if tree is None:
            return S

        if tree.value.leftEdge <= point <= tree.value.rightEdge:
            S.add(tree.value)

        if tree.left and tree.left.max >= point:
            S.update(self.searchIntervalTree(tree.left, point, S))
        if tree.right and point >= tree.value.leftEdge:
            S.update(self.searchIntervalTree(tree.right, point, S))

        return S

    def createTree(self, intervals):
        tree = None
        for interval in intervals:
            tree = self.insertNode(tree, interval)
        return tree

    def insertNode(self, tree, interval):
        if not tree:
            return TreeNodeIntervalTree(interval.leftEdge, interval.rightEdge)

        if interval.leftEdge < tree.value.leftEdge:
            tree.left = self.insertNode(tree.left, interval)
        else:
            tree.right = self.insertNode(tree.right, interval)

        if interval.rightEdge > tree.max:
            tree.max = interval.rightEdge

        tree.height = 1 + max(self.calculateHeight(tree.left), self.calculateHeight(tree.right))
        balance = self.calculateBalance(tree)

        if balance > 1 and interval.leftEdge < tree.left.value.leftEdge:
            return self.rightRotate(tree)

        if balance < -1 and interval.leftEdge >= tree.right.value.leftEdge:
            return self.leftRotate(tree)

        if balance > 1 and interval.leftEdge >= tree.left.value.leftEdge:
            tree.left = self.leftRotate(tree.left)
            return self.rightRotate(tree)

        if balance < -1 and interval.leftEdge < tree.right.value.leftEdge:
            tree.right = self.rightRotate(tree.right)
            return self.leftRotate(tree)

        return tree

    def leftRotate(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self.calculateHeight(z.left), self.calculateHeight(z.right))
        y.height = 1 + max(self.calculateHeight(y.left), self.calculateHeight(y.right))
        z.max = max(self.calculateMax(z.left), self.calculateMax(z.right))
        y.max = max(self.calculateMax(y.left), self.calculateMax(y.right))

        return y

    def rightRotate(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self.calculateHeight(z.left), self.calculateHeight(z.right))
        y.height = 1 + max(self.calculateHeight(y.left), self.calculateHeight(y.right))
        z.max = max(self.calculateMax(z.left), self.calculateMax(z.right))
        y.max = max(self.calculateMax(y.left), self.calculateMax(y.right))

        return y

    def calculateMax(self, tree):
        if not tree:
            return 0
        return tree.max

    def calculateHeight(self, tree):
        if not tree:
            return 0
        return tree.height

    def calculateBalance(self, tree):
        if not tree:
            return 0
        return self.calculateHeight(tree.left) - self.calculateHeight(tree.right)





def calculateSizeOfSegmentTree(x):
    return pow(2, math.ceil(math.log2(x)))

def getElementaryIntervals(intervals):
    array = []
    for i in intervals.keys():
        interval = intervals[i]
        array.append(interval.leftEdge)
        array.append(interval.rightEdge)
    array.sort()
    elementaryIntervals = []
    elementaryIntervals.append(Interval(NEG_INF, array[0]))
    for x in range(len(array) - 1):
        elementaryIntervals.append(Interval(array[x], array[x]))
        elementaryIntervals.append(Interval(array[x], array[x + 1]))
    elementaryIntervals.append(Interval(array[len(array) - 1], POS_INF))
    return elementaryIntervals

class TreeNodeSegmentTree:
    def __init__(self, interval):
        self.left = None
        self.right = None
        self.interval = interval
        self.markers = set()

# Implementacija Segmentnog stabla
class SegmentTree(object):
    def buildBinaryTreeUtil(self, elementaryIntervals, low, high, size):
        if low > high or low >= size:
            return None

        if low == high or high - low == 1:
            node = TreeNodeSegmentTree(elementaryIntervals[low])
            return node

        i = low + (high - low) // 2
        newInterval = None
        if high > size:
            newInterval = Interval(elementaryIntervals[low].leftEdge, elementaryIntervals[size - 1].rightEdge)
        else:
            newInterval = Interval(elementaryIntervals[low].leftEdge, elementaryIntervals[high - 1].rightEdge)
        node = TreeNodeSegmentTree(newInterval)
        node.left = self.buildBinaryTreeUtil(elementaryIntervals, low, i, size)
        node.right = self.buildBinaryTreeUtil(elementaryIntervals, i, high, size)
        return node

    def buildBinaryTree(self, elementaryIntervals):
        size = len(elementaryIntervals)
        return self.buildBinaryTreeUtil(elementaryIntervals, 0, calculateSizeOfSegmentTree(size), size)

    def checkIfFirstIntervalContainsSecond(self, interval1, interval2):
        if interval1.leftEdge <= interval2.leftEdge and interval1.rightEdge >= interval2.rightEdge:
            return True
        else:
            return False

    def addMarkerForInterval(self, tree, interval, name):
        if tree is None:
            return

        if self.checkIfFirstIntervalContainsSecond(interval, tree.interval):
            tree.markers.add(name)
            return

        self.addMarkerForInterval(tree.left, interval, name)
        self.addMarkerForInterval(tree.right, interval, name)

    def addMarkers(self, tree, intervals):
        for i in intervals.keys():
            self.addMarkerForInterval(tree, intervals[i], i)

    def buildSegmentTree(self, intervals):
        elementaryIntervals = getElementaryIntervals(intervals)
        tree = self.buildBinaryTree(elementaryIntervals)
        self.addMarkers(tree, intervals)
        return tree

    def searchSegmentTree(self, tree, point):
        if tree is None:
            return set()

        markers = set()
        if tree.interval.leftEdge <= point <= tree.interval.rightEdge:
            markers.update(tree.markers)
            markers.update(self.searchSegmentTree(tree.left, point))
            markers.update(self.searchSegmentTree(tree.right, point))
        return markers





class TreeNodeIntervalSkipList(object):
    def __init__(self, level, value):
        self.level = level  # nivo na kom se cvor nalazi
        self.forward = [None] * (level + 1)  # pokazivaci na naredne cvorove u listi
        self.markers = [set() for i in range(level + 1)]  # niz setova markera za grane koje izlaze iz cvora
        self.owners = Counter()  # multi set identifiera intervala kojima je granica vrednost cvora
        self.eqMarkers = set()  # set markera za intervale koji imaju marker na stranici koja zavrsava u ovom cvoru
        self.value = value  # vrednost cvora (ne inverval vec jedna granica intervala)

# Implementacija Intervalne Skip liste
class IntervalSkipList(object):
    def __init__(self, maxLevel, p):
        self.p = p
        self.maxLevel = maxLevel
        self.level = 0
        self.header = self.createNode(self.maxLevel, None)

    def createNode(self, level, value):
        n = TreeNodeIntervalSkipList(level, value)
        return n

    def getLevel(self, x):
        level = 0
        while random.random() < self.p and level < self.maxLevel:
            level += 1
        return level

    def insertNode(self, x):
        update = [None] * (self.maxLevel + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < x:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.value != x:
            newLevel = self.getLevel(x)

            if newLevel > self.level:
                for i in range(self.level + 1, newLevel + 1):
                    update[i] = self.header
                self.level = newLevel

            newNode = self.createNode(newLevel, x)

            for i in range(newLevel + 1):
                newNode.forward[i] = update[i].forward[i]
                update[i].forward[i] = newNode

            return True, newNode, update
        elif current and current.value == x:
            return False, current, update
        else:
            print("Error")
            return False, None, update

    def searchElement(self, key):
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < key:
                current = current.forward[i]

        current = current.forward[0]
        if current and current.value == key:
            return current
        else:
            return None

    def searchIntervals(self, key):
        current = self.header
        S = set()

        for i in range(self.level, 0, -1):
            while current.forward[i] and current.forward[i].value < key:
                current = current.forward[i]
            S.update(current.markers[i])

        while current.forward[0] and current.forward[0].value < key:
            current = current.forward[0]

        if current.forward[0] is None or current.forward[0].value != key:
            S.update(current.markers[0])
        else:
            S.update(current.forward[0].eqMarkers)

        return S

    def findIntervalWithName(self, m):
        return intervalNames[m]

    def containsInterval(self, m, a, b):
        interval = self.findIntervalWithName(m)
        if a is not None and a.value is not None and interval.leftEdge <= a.value and \
                b is not None and b.value is not None and interval.rightEdge >= b.value:
            return True
        else:
            return False

    def intervalContainsPoint(self, I, x):
        if I.leftEdge <= x <= I.rightEdge:
            return True
        else:
            return False

    def searchAndRemoveMarker(self, i, a, b, m):
        # izbrisi markere na nivou i
        current = a
        while current.forward[i] and current.forward[i].value <= b.value:
            current.forward[i].eqMarkers.discard(m)
            current.markers[i].discard(m)
            current = current.forward[i]

    def adjustMarkersOnInsert(self, N, updated):
        # Faza 1: popravi markere na granama koje izlaze iz N
        promoted = set()
        newPromoted = set()

        for i in range(0, N.level, 1):
            pom = set(updated[i].markers[i])
            for m in pom:
                if N.forward[i + 1] and self.containsInterval(m, N, N.forward[i + 1]):
                    # promovisi m
                    self.searchAndRemoveMarker(i, N, N.forward[i + 1], m)
                    newPromoted.add(m)
                elif N.forward[i]:
                    N.markers[i].add(m)
                    N.forward[i].eqMarkers.add(m)

            pom = set(promoted)
            for m in pom:
                if not self.containsInterval(m, N, N.forward[i + 1]):
                    N.markers[i].add(m)
                    N.forward[i].eqMarkers.add(m)
                    promoted.discard(m)
                else:
                    # nastavi da promovises m
                    # obrisi m
                    self.searchAndRemoveMarker(i, N, N.forward[i + 1], m)

            promoted.update(newPromoted)
            newPromoted = set()

        Nlevel = N.level
        N.markers[Nlevel] = promoted
        if N.forward[Nlevel] is not None and updated[Nlevel] is not None and updated[Nlevel].markers[Nlevel] is not None:
            N.markers[Nlevel].update(updated[Nlevel].markers[Nlevel])
            N.forward[Nlevel].eqMarkers.update(N.markers[Nlevel])

        # Faza 2: popravi markere na granama koje ulaze u N
        promoted = set()
        newPromoted = set()

        for i in range(0, N.level, 1):
            pom = set(updated[i].markers[i])
            for m in pom:
                if self.containsInterval(m, updated[i + 1], N):
                    newPromoted.add(m)
                    self.searchAndRemoveMarker(i, updated[i + 1], N, m)
                else:
                    N.eqMarkers.add(m)

            pom = set(promoted)
            for m in pom:
                if self.containsInterval(m, updated[i], N) and not self.containsInterval(m, updated[i + 1], N):
                    # dodaj m
                    updated[i].markers[i].add(m)
                    N.eqMarkers.add(m)
                    promoted.discard(m)
                else:
                    # obrisi m
                    self.searchAndRemoveMarker(i, updated[i + 1], N, m)

            promoted.update(newPromoted)
            newPromoted = set()

        top = N.level
        if updated[top]:
            if updated[top].markers[top] is None:
                updated[top].markers[top] = set()
            updated[top].markers[top].update(promoted)
            N.eqMarkers.update(updated[top].markers[top])

    def placeMarkers(self, I, name):

        x = self.searchElement(I.leftEdge)

        if self.intervalContainsPoint(I, x.value):
            x.eqMarkers.add(name)

        i = 0
        while x and x.forward[i] and self.containsInterval(name, x, x.forward[i]):
            while i <= x.level - 1 and self.containsInterval(name, x, x.forward[i + 1]):
                i = i + 1
            x.markers[i].add(name)
            x = x.forward[i]
            if x and self.intervalContainsPoint(I, x.value):
                x.eqMarkers.add(name)

        while x and x.value < I.rightEdge:
            while i > 0 and not self.containsInterval(name, x, x.forward[i]):
                i = i - 1
            x.markers[i].add(name)
            x = x.forward[i]
            if x and self.intervalContainsPoint(I, x.value):
                x.eqMarkers.add(name)

    def insertInterval(self, I, name):
        newNode, nodeA, updateA = self.insertNode(I.leftEdge)
        if newNode:
            self.adjustMarkersOnInsert(nodeA, updateA)
        newNode, nodeB, updateB = self.insertNode(I.rightEdge)
        if newNode:
            self.adjustMarkersOnInsert(nodeB, updateB)

        self.placeMarkers(I, name)

    def displayList(self):
        print("Skip list with markers")
        head = self.header

        for lvl in range(self.level + 1):
            print("Level {}: ".format(lvl), end=" ")
            node = head.forward[lvl]
            while node:
                print("Key: {}, Markers: {}".format(node.value, node.markers[lvl]), end="    ")
                node = node.forward[lvl]
            print("")

    def displayListWithEqMarkers(self):
        print("Skip list eqMarkers")
        head = self.header

        for lvl in range(self.level + 1):
            print("Level {}: ".format(lvl), end=" ")
            node = head.forward[lvl]
            while node:
                print("Key: {}, EqMarkers: {}".format(node.value, node.eqMarkers), end="    ")
                node = node.forward[lvl]
            print("")

    def checkIfMarkersAreOk(self, list):
        isFine = True

        # pomocna struktura u koju upisujemo sve grane koje treba da imaju eqMarkere na osnovu markera
        eqMarkers = {}
        nodes = {}

        while list.forward[0]:
            for i in range(list.level + 1):
                if len(list.markers[i]) != 0:
                    value = list.forward[i].value
                    nodes[value] = list.forward[i]
                    if value in eqMarkers:
                        eqMarkers[value].update(list.markers[i])
                    else:
                        eqMarkers[value] = set(list.markers[i])
                for m in list.markers[i]:
                    if m not in list.forward[i].eqMarkers:
                        isFine = False
            list = list.forward[0]

        for key in eqMarkers.keys():
            # da li nam se razlikuju ocekivan skup eqMarkera od onog dobijenog u skip listi
            diff = eqMarkers[key].symmetric_difference(nodes[key].eqMarkers)
            for d in diff:
                # ako je leva granica intervala ta tacka onda ne postoji oznaka za interval u listi markera
                # postoji samo u eqMarkerima i samim tim treba da imamo razliku u setovima
                if intervalNames[d].leftEdge != key:
                    isFine = False

        print("Interval skip list structure is populated {}".format("CORRECTLY" if isFine else "WITH MISTAKE"))


def createIntervalsToBeInserted(n, points):
    intervals_dict = {}
    intervals_list = []
    intervalNames.clear()
    names.clear()
    for i in range(0, n):
        left = random.randint(0, n)
        right = random.randint(left, n)
        interval = Interval(left, right)
        name = str(left) + "-" + str(right)
        intervals_dict[name] = interval
        intervals_list.append(interval)
        intervalNames[name] = interval
        names.append(name)
        points.append(left)
        points.append(right)
    return intervals_dict, intervals_list

def printIntervalsIntervalTree(intervals):
    for i in intervals:
        print("({}, {})".format(i.leftEdge, i.rightEdge), end=" ")
    print()

def printIntervalsSegmentTree(intervals):
    for i in intervals.keys():
        print("{} = ({}, {})".format(i, intervals[i].leftEdge, intervals[i].rightEdge))

def printIntervalsIntervalSkipList():
    print("List of intervals")
    for i in intervalNames.keys():
        print("{} = ({}, {})".format(i, intervalNames[i].leftEdge, intervalNames[i].rightEdge))






def testIntervalTree():
    intervals = [Interval(5, 10), Interval(6, 11), Interval(4, 6), Interval(1, 3), Interval(6, 6)]
    point = 10
    intervalTree = IntervalTree()
    iTree = intervalTree.createTree(intervals)
    S = intervalTree.searchIntervalTree(iTree, point, None)
    print("*** Interval Tree ***")
    print("Intervals:")
    printIntervalsIntervalTree(intervals)
    print("Intervals that overlap point {} are:".format(point))
    printIntervalsIntervalTree(S)

def testSegmentTree():
    intervals = {
        "a": Interval(2, 5),
        "b": Interval(1, 17),
        "c": Interval(8, 12),
        "d": Interval(100, 120),
        "e": Interval(80, 121),
        "f": Interval(82, 88),
        "g": Interval(18, 19)
    }
    point = 88

    segmentTree = SegmentTree()
    sTree = segmentTree.buildSegmentTree(intervals)
    S = segmentTree.searchSegmentTree(sTree, point)
    print("*** Segment Tree ***")
    print("Intervals:")
    printIntervalsSegmentTree(intervals)
    print("Intervals that overlap point {} are:".format(point))
    if S is not None:
        print(S)

def testIntervalSkipList():
    printIntervalsIntervalSkipList()

    list = IntervalSkipList(6, 0.5)
    createIntervalsToBeInserted(10, [])
    for name in names:
        list.insertInterval(intervalNames[name], name)

    point = random.randint(0, 10)
    S = list.searchIntervals(point)
    list.displayList()
    list.displayListWithEqMarkers()

    if S is not None:
        print("Intervals that overlap point {} are {}".format(point, S))

    list.checkIfMarkersAreOk(list.header)


def timePerformance():
    searchingResultsIntervalTree = [0] * 100     # niz u kome cuvamo vreme pretrage elementa
    searchingResultsSegmentTree = [0] * 100
    searchingResultsIntervalSkipList = [0] * 100
    insertIntervalTree = [0] * 100
    insertIntervalSkipList = [0] * 100
    points = []

    n = 100
    for i in range(0, 100):
        points.clear()
        intervalTree = IntervalTree()
        segmentTree = SegmentTree()
        intervalSkipList = IntervalSkipList(100, 0.5)
        iTree = None
        intervals_dict, intervals_list = createIntervalsToBeInserted(n, points)
        for interval in intervals_list:
           iTree = intervalTree.insertNode(iTree, interval)
        sTree = segmentTree.buildSegmentTree(intervals_dict)
        for name in names:
            intervalSkipList.insertInterval(intervalNames[name], name)

        for j in range(0, 100):
            index = random.randint(0, len(points) - 1)
            q = points[index]

            left = random.randint(0, n)
            right = random.randint(left, n)
            newInterval = Interval(left, right)
            name = str(left) + "-" + str(right)
            intervalNames[name] = newInterval

            startSearching = time.clock()
            intervalTree.insertNode(iTree, newInterval)
            endSearching = time.clock()
            insertIntervalTree[i] = insertIntervalTree[i] + endSearching - startSearching

            startSearching = time.clock()
            intervalSkipList.insertInterval(newInterval, name)
            endSearching = time.clock()
            insertIntervalSkipList[i] = insertIntervalSkipList[i] + endSearching - startSearching


            startSearching = time.clock()
            S1 = intervalTree.searchIntervalTree(iTree, q, None)
            endSearching = time.clock()
            searchingResultsIntervalTree[i] = searchingResultsIntervalTree[i] + endSearching - startSearching

            startSearching = time.time()
            segmentTree.searchSegmentTree(sTree, q)
            endSearching = time.time()
            searchingResultsSegmentTree[i] = searchingResultsSegmentTree[i] + endSearching - startSearching

            startSearching = time.clock()
            intervalSkipList.searchIntervals(q)
            endSearching = time.clock()
            searchingResultsIntervalSkipList[i] = searchingResultsIntervalSkipList[i] + endSearching - startSearching

        if searchingResultsIntervalTree[i] != 0:
            searchingResultsIntervalTree[i] = searchingResultsIntervalTree[i] / 100
        if searchingResultsSegmentTree[i] != 0:
            searchingResultsSegmentTree[i] = searchingResultsSegmentTree[i] / 100
        if searchingResultsIntervalSkipList[i] != 0:
            searchingResultsIntervalSkipList[i] = searchingResultsIntervalSkipList[i] / 100
        if insertIntervalTree[i] != 0:
            insertIntervalTree[i] = insertIntervalTree[i] / 100
        if insertIntervalSkipList[i] != 0:
            insertIntervalSkipList[i] = insertIntervalSkipList[i] / 100

        n = n + 100

    print("Insert interval tree")
    print(insertIntervalTree)
    print("Insert interval skip list")
    print(insertIntervalSkipList)
    print("Search interval tree")
    print(searchingResultsIntervalTree)
    print("Search interval skip list")
    print(searchingResultsIntervalSkipList)

    x = range(0, 10000, 100)

    # Pravljenje grafika za umetanje novog elementa
    plt.xlabel('Broj elemenata u strukturi podataka')
    plt.ylabel('Vreme potrebno za umetanje novog elementa')
    plt.plot(x, insertIntervalTree, 'C0', label="intervalno stablo")
    plt.plot(x, insertIntervalSkipList, 'C2', label="intervalna skip lista")
    plt.legend(title="Legenda")
    plt.tight_layout()
    plt.savefig('inserting')

    plt.clf()

    # Pravljenje grafika za pretragu struktura podataka
    plt.xlabel('Broj elemenata u strukturi podataka')
    plt.ylabel('Vreme potrebno za pretragu')
    plt.plot(x, searchingResultsIntervalTree, 'C0', label="intervalno stablo")
    plt.plot(x, searchingResultsIntervalSkipList, 'C2', label="intervalna skip lista")
    plt.plot(x, searchingResultsSegmentTree, 'C1', label="segmentno stable")
    plt.legend(title="Legenda")
    plt.tight_layout()
    plt.savefig('searching')

    plt.clf()

    # Pravljenje grafika za pretragu segmentnog stabla
    plt.xlabel('Broj elemenata u segmentnom stablu')
    plt.ylabel('Vreme potrebno za pretragu segmentnog stabla')
    plt.plot(x, searchingResultsSegmentTree, label="segmentno stablo")
    plt.tight_layout()
    plt.savefig('searchingSegmentTree')

    plt.clf()

    # Pravljenje grafika za umetanje novog elementa u intervalno stablo
    plt.xlabel('Broj elemenata intervalnog stabla')
    plt.ylabel('Vreme potrebno za umetanje novog elementa')
    plt.plot(x, insertIntervalTree, label="intervalno stablo")
    plt.tight_layout()
    plt.savefig('insertingIntervalTree')

    plt.clf()

    # Pravljenje grafika za umetanje novog elementa u intervalnu skip listu
    plt.xlabel('Broj elemenata intervalne skip liste')
    plt.ylabel('Vreme potrebno za umetanje novog elementa')
    plt.plot(x, insertIntervalSkipList, label="intervalna skip lista")
    plt.tight_layout()
    plt.savefig('insertingIntervalSkipList')

    plt.clf()

    # Pravljenje grafika za pretragu intervalnog stabla
    plt.xlabel('Broj elemenata intervalnog stabla')
    plt.ylabel('Vreme potrebno za pretragu intervalnog stabla')
    plt.plot(x, searchingResultsIntervalTree, label="intervalno stablo")
    plt.tight_layout()
    plt.savefig('searchingIntervalTree')

    plt.clf()

    # Pravljenje grafika za pretragu intervalne skip liste
    plt.xlabel('Broj elemenata intervalne skip liste')
    plt.ylabel('Vreme potrebno za pretragu intervalne skip liste')
    plt.plot(x, searchingResultsIntervalSkipList, label="intervalna skip lista")
    plt.tight_layout()
    plt.savefig('searchingIntervalSkipList')

    plt.clf()
    
    
    

def main():
    testIntervalTree()
    testSegmentTree()
    testIntervalSkipList()
    timePerformance()

main()