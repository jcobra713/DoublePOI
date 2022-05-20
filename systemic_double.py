import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd
import scipy.stats as stats
from itertools import permutations
import itertools
import random


class distance:
    # class used for arrival time and distance calculations
    vP = None
    tP = None

    def __init__(self, vP, tP):
        self.vP = vP
        self.tP = tP

    def travelTime(self, src, dest, start, finish): # src and dest are the indexes / start and finish are bools that determine the situation
        if start and finish:
            return spd.pdist([self.vP[2*src], self.vP[2*src+1]])
        elif start:
            return spd.pdist([self.vP[2*src], self.tP[dest]])
        elif finish:
            return spd.pdist([self.tP[dest], self.vP[2*src+1]])
        else:
            return spd.pdist([self.tP[src], self.tP[dest]])

    def cost(self, vehicle, path):
        distance = 0
        currentPosition = self.vP[2 * vehicle]
        sp = spd.pdist([self.vP[2 * vehicle], self.vP[2 * vehicle + 1]])
        for t in path:
            distance += spd.pdist([currentPosition, self.tP[int(t)]])  # Check if t is target or target index!!!
            currentPosition = self.tP[int(t)]
        distance += spd.pdist([currentPosition, self.vP[2 * vehicle + 1]])
        return distance - sp


class nashAssigner:
    N = None  # number of vehicles
    M = None  # number of targets
    R = None  # max reward per sample
    tau = None
    dist = None  # object of class used for arrival time and distance calculations
    vP = None
    arrivalTime = None  # dictionary that maintains sequence of vehicle numbers and their arrival times
    decisionTime = None
    targets = None
    utilities = None
    iterations = None

    collection = None

    #CHANGED TO ACCOMODATE 2 TARGETS
    def __init__(self, N, M, R, tau, dist=None, arrivalTime={}, decisionTime=0):  # initialize
        self.N = N
        self.M = M
        self.R = R
        self.tau = tau
        self.dist = dist
        self.arrivalTime = arrivalTime
        self.decisionTime = decisionTime

        # initalize arrival times at all targets to empty sets
        for t in range(self.M): self.arrivalTime[t] = []
        self.targets = np.zeros((self.N, 2), dtype='int')  # initalize by assigning random targets
        self.collection = list(permutations(range(0, M), 2))
        singleTList = [(val, val) for val in range(0, M)]
        self.collection.extend(singleTList)
        # assign random targets for initialization
        for i in range(self.N):
            index = self.randomAllocation()
            at = np.zeros(2)
            at[0] = self.decisionTime + self.dist.travelTime(src=i, dest=index[0], start=True, finish=False)
            if index[0] != index[1]:
                at[1] = self.dist.travelTime(src=index[0], dest=index[1], start=False, finish=False) + at[0]
                self.arrivalTime[index[1]].append((i, at[1]))
            self.arrivalTime[index[0]].append((i, at[0]))
            self.targets[i] = index

        tar = str(self.targets)
        tar = tar.replace('\n', ",")
        print(tar)
        # sort the arrival times at all targets
        for t in self.arrivalTime:
            self.arrivalTime[t].sort(key=lambda x: x[1], reverse=True)
        self.utilities = np.zeros(N)

    def randomAllocation(self):
        randNum = np.random.randint(low=0, high=len(self.collection))
        return self.collection[randNum]


    # CHANGING
    def getUtility(self, si, path):  # returns utility of vehicle si to go to target t (dependent on state of other vehicles)
        reward2 = 0
        at = np.zeros(2)
        at[0] = self.decisionTime + self.dist.travelTime(src=si, dest=path[0], start=True, finish=False)  # calculate arrival time of si at t
        at[1] = self.dist.travelTime(src=path[0], dest=path[1], start=False, finish=False) + at[0]
        # calculate reward according to previous arrival time at t
        try:
            pat = next(x[1] for x in self.arrivalTime[path[0]] if x[1] < at[0])  # get the previous arrival time at t
            reward1 = self.R * (1 - np.e ** (-(at[0] - pat) / self.tau))
        except StopIteration:
            reward1 = self.R
        # Have to implement to be compatible with the single target situations
        if path[0] != path[1]:
            try:
                pat2 = next(x[1] for x in self.arrivalTime[path[1]] if x[1] < at[1])
                reward2 = self.R * (1 - np.e ** (-(at[1] - pat2) / self.tau))
            except StopIteration:
                reward2 = self.R

        cost = self.dist.cost(vehicle=si, path=path)  # calculate cost
        return reward1 + reward2 - cost

    def getReward(self, si, path):  # returns reward given to vehicle si to go to target t (dependent on state of other vehicles)
        reward2 = 0
        at = np.zeros(2)
        at[0] = self.decisionTime + self.dist.travelTime(src=si, dest=path[0], start=True,
                                                         finish=False)  # calculate arrival time of si at t
        at[1] = self.dist.travelTime(src=path[0], dest=path[1], start=False, finish=False) + at[0]
        # calculate reward according to previous arrival time at t
        try:
            pat = next(x[1] for x in self.arrivalTime[path[0]] if x[1] < at[0])  # get the previous arrival time at t
            reward1 = self.R * (1 - np.e ** (-(at[0] - pat) / self.tau))
        except StopIteration:
            reward1 = self.R
        # Have to implement to be compatible with the single target situations
        if path[0] != path[1]:
            try:
                pat2 = next(x[1] for x in self.arrivalTime[path[1]] if x[1] < at[1])
                reward2 = self.R * (1 - np.e ** (-(at[1] - pat2) / self.tau))
            except StopIteration:
                reward2 = self.R

        return reward1 + reward2

    def getBestTarget(self, si):
        u = np.zeros(len(self.collection))
        i = 0
        for path in self.collection:
            u[i] = self.getUtility(si, path)
            i += 1
        return self.collection[np.argmax(u)]

    def pair(self, si, path):  # pair vehicle si with target t
        tc = self.targets[si]  # current target
        if tc[0] == path[0] and tc[1] == path[1]: return  # move along, nothing to do here
        at = np.zeros(2)
        # remove si and its arrival time from current target
        sip1 = next(x for x in self.arrivalTime[tc[0]] if x[0] == si)  # vehicle, arrival time pair
        self.arrivalTime[tc[0]].remove(sip1)
        at[0] = self.decisionTime + self.dist.travelTime(src=si, dest=path[0], start=True, finish=False)
        self.arrivalTime[path[0]].append((si, at[0]))
        self.arrivalTime[path[0]].sort(key=lambda x: x[1], reverse=True)
        if tc[0] != tc[1]:  # If the previous path is a single target, it doesn't need to be removed twice
            sip2 = next(x for x in self.arrivalTime[tc[1]] if x[0] == si)  # vehicle, arrival time pair
            self.arrivalTime[tc[1]].remove(sip2)

        if path[0] != path[1]:
            # add si and its arrival time at the new target
            at[1] = self.dist.travelTime(src=path[0], dest=path[1], start=False, finish=False) + at[0]
            self.arrivalTime[path[1]].append((si, at[1]))
            self.arrivalTime[path[1]].sort(key=lambda x: x[1], reverse=True)
        # assign the new target
        self.targets[si] = path

    def detach(self, si):  # detaches the vehicle si from its current target
        tc = self.targets[si]
        if tc[0] == np.nan: return
        sip1 = next(x for x in self.arrivalTime[tc[0]] if x[0] == si)  # vehicle, arrival time pair
        self.arrivalTime[tc[0]].remove(sip1)
        if tc[0] != tc[1]:
            sip2 = next(x for x in self.arrivalTime[tc[1]] if x[0] == si)
            self.arrivalTime[tc[1]].remove(sip2)
        self.targets[si] = np.nan

    def successor(self, si, path):  # get the vehicle arriving after si at target t
        at = np.zeros(2)
        at[0] = self.decisionTime + self.dist.travelTime(src=si, dest=path[0], start=True, finish=False)
        at[1] = self.dist.travelTime(src=path[0], dest=path[1], start=False, finish=False) + at[0]
        try:
            succ1 = next(x[0] for x in reversed(self.arrivalTime[path[0]]) if x[1] > at[0])
        except StopIteration:
            succ1 = None
        try:
            succ2 = next(x[0] for x in reversed(self.arrivalTime[path[1]]) if x[1] > at[1])
        except StopIteration:
            succ2 = None
        return (succ1, succ2) # Check to see if returning correctly!!!

    def closestTargetTo(self, si):
        distances = np.zeros(self.M)
        for t in range(self.M):
            distances[t] = self.dist.travelTime(vehicle=si, target=t, source=True)
        return np.argmin(distances)

    # function takes in a given configuration and returns target assignments
    def getAssignments(self):
        # Returns: (targets, utilities)
        # targets[x] is the target allocation for vehicle x. If no target is assigned to x, then targets[x] will be equal to np.nan
        # utilities[x] contains the utility of the vehicle x
        count = self.N
        iteration = 0
        while count:
            # print("Count: ", count, "    Iteration: ", iteration)
            count = 0
            iteration = iteration + 1
            if iteration > 25: break
            # print('running iteration # %d'%iteration)
            S = set(range(self.N))
            startingAssignment = np.copy(self.targets)
            transitionSequence = [startingAssignment]
            cycleCount = 0
            while S:
                if cycleCount % 130 == 0:
                    startingAssignment = np.copy(self.targets)
                    transitionSequence = [startingAssignment]
                si = random.sample(S, 1)[0]
                S.remove(si)
                t1 = self.targets[si]
                tk = self.getBestTarget(si)  # arg max_l u(si,tl)
                # print("Vehicle: ", si, "    ", t1, "    ", tk)
                if t1[0] != tk[0] or t1[1] != tk[1]:
                    count += 1
                    self.pair(si, tk)
                    sj = self.successor(si, tk)
                    # print(si,t1,tk,sj)
                    if sj[0] != None:
                        S.add(sj[0])
                    if sj[1] != None and sj[1] != sj[0]:
                        S.add(sj[1])
                    transitionSequence.append(np.copy(self.targets))
                    endingAssignment = np.copy(self.targets)
                    test = startingAssignment == endingAssignment
                    if np.sum(startingAssignment == endingAssignment) == self.N*2 and count > 0:  # detected a cycle
                        # print('detected a cycle')
                        # for each transition, find the cost to go to the new location
                        edgeCosts = []
                        for i in range(len(transitionSequence) - 1):
                            # find the vehicle which is transition
                            tVehicle = np.where(transitionSequence[i] != transitionSequence[i + 1])
                            # find the new target
                            tTarget = transitionSequence[i + 1][tVehicle[0][0]]
                            edgeCosts.append(self.dist.cost(vehicle=tVehicle[0][0],
                                                            path=tTarget))  # Check if tTarget is passing as a tuple
                        self.R = max(edgeCosts) - 0.01
                        print('adusted reward to %f' % self.R)
                cycleCount += 1
        # print(self.targets)
        self.iterations = iteration
        # calculate utilities at the Nash equilibrium
        for s in range(self.N):
            self.utilities[s] = self.getUtility(s, self.targets[s])
            if self.utilities[s] < 0:
                self.detach(s)

        return (self.targets, self.utilities)

    def greedyAssignments(self):
        # Returns: (targets, utilities)
        S = set(range(self.N))
        while S:
            si = S.pop()
            ti = self.closestTargetTo(si)
            self.pair(si, ti)
        for s in range(self.N):
            self.utilities[s] = self.getUtility(s, self.targets[s])
            if self.utilities[s] < 0:
                self.targets[s] = np.nan
        return (self.targets, self.utilities)

    def getTargetUtility(self, t):
        if not self.arrivalTime[t]: return 0
        ats = [p[1] for p in self.arrivalTime[t]]   # Changed from p[1][0] to p[1]
        ot = ats[0] + np.mean(ats)  # end of observation period
        ats = [ot] + ats
        ats.append(self.decisionTime)
        ats = np.array(ats)
        its = ats[:-1] - ats[1:]  # time intervals
        its = its / np.sum(its)  # normalize to make probabilities
        return stats.entropy(its, base=2)  # /np.log2(len(its))

    def getVehicleReward(self, n):
        if self.targets[n][0] < 0: return 0
        return self.getReward(n, self.targets[n])


def getTargetAssignments(nA):
    print('final target allocation: (vehicle,target,utility)')
    allocations = []
    for x in range(nA.N):
        allocations.append('(%d,%.0f,%.3f)' % (x, nA.targets[x], nA.utilities[x]))
    return allocations


def getArrivalTimes(nA):
    print('arrival times at targets')
    arrivalTimes = []
    for x in nA.arrivalTime:
        times = ['(%d,%.2f)' % (p[0], p[1][0]) for p in nA.arrivalTime[x]]
        arrivalTimes.append(times)
    return arrivalTimes


def getTargetUtilities(nA):
    targetUtilities = []
    for t in range(nA.M):
        targetUtilities.append(nA.getTargetUtility(t))
    return targetUtilities


def getPlatformReward(nA):
    targetRewards = 0
    for n in range(nA.N):
        targetRewards += (nA.getVehicleReward(n))
    return targetRewards


def getVUtility(nA):
    vehicleU = []  # Utility per Reward
    value = 0
    for t in range(nA.N):
        if nA.targets[t][0] == nA.targets[t][0] and nA.targets[t][0] >= 0:
            value = nA.getUtility(t, nA.targets[t])
            vehicleU.append(value[0])  # Vehicle Utility per Dollar given out
        else:
            print("Vehicle ", t, " target = nan")
    return vehicleU

def getLocalizedPlacement(N, M):
    vP = np.zeros(shape=(2 * N, 2))
    for i in range(N):
        a = np.random.random((2, 2))
        vP[2 * i:2 * i + 2, :] = (1 / (1.2 * (np.sum(a)))) * a
        vP[2 * i + 1, :] = 1 - vP[2 * i + 1, :]

    r = np.random.random(M) - 0.5  # random radii in [-0.5,0.5]
    theta = np.pi / 2 + (np.pi / 6) * np.random.random(M)  # random ange in [pi/2-pi/6, pi/2+pi/6]
    complexCrd = r * np.e ** (1j * theta)
    tP = np.zeros(shape=(M, 2))
    tP[:, 0] = np.real(complexCrd)
    tP[:, 1] = np.imag(complexCrd)
    tP = (0.5, 0.5) + tP

    return (vP, tP)

####################################
def printVp(vP):
    vehicle = 0
    count = 0
    for i in vP:
        if count % 2 == 0:
            vehicle += 1
        count += 1
        # print("Vehicle ", vehicle, " position: ", i, "    ", count % 2)

def printTp(tP):
    count = 0
    for i in tP:
        # print("Target ", count, " position: ", i)
        count += 1

##################################

def otherMain(N, M, vP, tP):
    printVp(vP)
    printTp(tP)
    dist = distance(vP,tP)
    R = 3
    tau = 1

    print("Running standard assignment")
    arrivalTime = {}  # dictionary that maintains sequence of vehicle numbers and their arrival times
    for t in range(M): arrivalTime[t] = []
    decisionTime = 0
    targets = np.zeros((N, 2), dtype='int')
    x = list(range(0, M))
    x = [p for p in itertools.product(x, repeat=2)]
    collection = [p for p in itertools.product(x, repeat=N)]
    nash = []
    # Check if nash equilibrium with getBestTarget on all vehicles. If tk = t1 then good for all
    # How far they are from greatest nash equilibrium



    targetUtilities = []
    permUtilities = []
    j = 0
    # print(collection[40000])
    for perm in collection:
        if j % 1000 == 0:
            print("j = ", j)
        j += 1
        skip = False
        arrivalTime = {}
        for t in range(M): arrivalTime[t] = []
        for i in range(N):
            # Actually Calculate the arrival times
            at = np.zeros(2)
            at[0] = dist.travelTime(src=i, dest=perm[i][0], start=True, finish=False)
            if perm[i][0] != perm[i][1]:
                at[1] = dist.travelTime(src=perm[i][0], dest=perm[i][1], start=False, finish=False) + at[0]
                arrivalTime[perm[i][1]].append((i, at[1]))
                arrivalTime[perm[i][1]].sort(key=lambda x: x[1], reverse=True)
            arrivalTime[perm[i][0]].append((i, at[0]))
            arrivalTime[perm[i][0]].sort(key=lambda x: x[1], reverse=True)
            targets[i] = perm[i]

        for i in range(N):
            # Check if Best Target
            u = np.zeros(len(x))
            q = 0
            for path in x:
                reward2 = 0
                at = np.zeros(2)
                at[0] = decisionTime + dist.travelTime(src=i, dest=path[0], start=True, finish=False)  # calculate arrival time of si at t
                at[1] = dist.travelTime(src=path[0], dest=path[1], start=False, finish=False) + at[0]
                # calculate reward according to previous arrival time at t
                try:
                    pat = next(x[1] for x in arrivalTime[path[0]] if x[1] < at[0])  # get the previous arrival time at t
                    reward1 = R * (1 - np.e ** (-(at[0] - pat) / tau))
                except StopIteration:
                    reward1 = R
                # Have to implement to be compatible with the single target situations
                if path[0] != path[1]:
                    try:
                        pat = next(x[1] for x in arrivalTime[path[1]] if x[1] < at[1])
                        reward2 = R * (1 - np.e ** (-(at[1] - pat) / tau))
                    except StopIteration:
                        reward2 = R
                cost = dist.cost(vehicle=i, path=path)  # calculate cost
                u[q] = reward1 + reward2 - cost
                q += 1
                # print("vehicle ", i, " path ", path, " at[0]: ", at[0], " at[1]: ", at[1], " reward1 and reward2: ", reward1, reward2)
            bestT = x[np.argmax(u)]
            if (perm[i][0] != bestT[0] or perm[i][1] != bestT[1]):
                skip = True
                break
                # Change to break
        if skip == True:
            continue

        for t in range(M): # Calculate the utility values for nash equilibriums
            if not arrivalTime[t]:
                targetUtilities.append(0)
                continue
            ats = [p[1] for p in arrivalTime[t]]  # Changed from p[1][0] to p[1]
            ot = ats[0] + np.mean(ats)  # end of observation period
            ats = [ot] + ats
            ats.append(decisionTime)
            ats = np.array(ats)
            its = ats[:-1] - ats[1:]  # time intervals
            its = its / np.sum(its)  # normalize to make probabilities
            targetUtilities.append(stats.entropy(its, base=2))
        nash.append(perm)
        permUtilities.append(np.mean(targetUtilities))
        targetUtilities.clear()
    # print(collection[np.argmax(permUtilities)])
    k = 0
    for n in nash:
        print("\n", n, " Platform Utility: ", permUtilities[k])
        k += 1
    return nash, permUtilities







def originalMain(N, M, vP, tP, nA):
    # printVp(vP)
    # printTp(tP)
    nOfIterations = []
    targetUtilities = []

    # R should be in the upper range of the diversion costs
    # print('running smart assignment')
    nA.getAssignments()  # get target assignments
    nOfIterations.append(nA.iterations)
    utility = np.mean(getTargetUtilities(nA))
    target = nA.targets
    # print(target)
    # plt.figure()
    # plt.plot(vP[:, 0], vP[:, 1], 'ro')
    # plt.plot(tP[:, 0], tP[:, 1], 'bo')
    # plt.show()
    # print(utility)


def largeVehicleRegimeTesting():
    n = 3
    nMax = 12
    Ns = range(n, nMax+1)
    M = 5

    nOfIterations = []
    targetUtilities = []
    platformReward = []
    platformUtilperReward = []
    averageVehicleUtility = []
    maxUtility = []
    minUtility = []

    while n <= nMax:
        print('running smart assignment, M=%d, N=%d' % (M, n))
        vP, tP = getLocalizedPlacement(n, M)
        dist = distance(vP, tP)
        nA = nashAssigner(N=n, M=M, R=3, tau=1, dist=dist)
        originalMain(n, M, vP, tP, nA)
        if nA.R != 3 or nA.iterations >= 26:
            print("Cycle Detected, Iteration Repeated")
            continue
        nOfIterations.append(nA.iterations)
        targetUtilities.append(np.mean(getTargetUtilities(nA)))
        platformReward.append(getPlatformReward(nA))
        platformUtilperReward.append(np.mean(getTargetUtilities(nA)) / getPlatformReward(nA))
        averageVehicleUtility.append(np.mean(getVUtility(nA)))
        maxUtility.append(np.amax(getTargetUtilities(nA)))
        minUtility.append(np.amin(getTargetUtilities(nA)))
        print(nA.iterations)
        n += 1

    with open('doubleTestRangeN3-14_M5_Sep29.csv', 'w') as out:
        for i, N in enumerate(Ns):
            # print(i, " ", N)
            out.write('%d,%d,%f,%f,%f,%f,%f,%f\n' % (N, nOfIterations[i], targetUtilities[i], platformReward[i], platformUtilperReward[i], maxUtility[i],
                                                        minUtility[i], averageVehicleUtility[i]))



def averageTest():
    n = 2
    nStart = n
    nMax = 9
    Ns = range(n, nMax+1)
    M = 4

    loop = 0
    loopMax = 20
    cycles = 0
    cycleIterationMax = 50

    nOfIterations = []
    targetUtilities = []
    platformReward = []
    platformUtilperReward = []
    averageVehicleUtility = []
    maxUtility = []
    minUtility = []

    averages = {}
    averages['nOfIterations'] = []
    averages['targetUtilities'] = []
    averages['platformReward'] = []
    averages['platformUtilperReward'] = []
    averages['averageVehicleUtility'] = []
    averages['maxUtility'] = []
    averages['minUtility'] = []

    while loop < loopMax:
        while n <= nMax:
            print('running smart assignment, M=%d, N=%d' % (M, n))
            vP, tP = getLocalizedPlacement(n, M)
            dist = distance(vP, tP)
            nA = nashAssigner(N=n, M=M, R=3, tau=1, dist=dist)
            originalMain(n, M, vP, tP, nA)
            while (nA.R != 3 or nA.iterations >= 26) and cycles <= cycleIterationMax:
                cycles = cycles + 1
                print("Cycle Detected, Iteration Repeated")
                nA = nashAssigner(N=n, M=M, R=3, tau=1, dist=dist)
                originalMain(n, M, vP, tP, nA)

            if cycles > cycleIterationMax:
                print("Too many cycles, skipping")
                cycles = 0
                continue

            nOfIterations.append(nA.iterations)
            targetUtilities.append(np.mean(getTargetUtilities(nA)))
            platformReward.append(getPlatformReward(nA))
            platformUtilperReward.append(np.mean(getTargetUtilities(nA)) / getPlatformReward(nA))
            averageVehicleUtility.append(np.mean(getVUtility(nA)))
            maxUtility.append(np.amax(getTargetUtilities(nA)))
            minUtility.append(np.amin(getTargetUtilities(nA)))
            cycles = 0
            # print(nA.iterations)
            n += 1

        '''with open('Data5/doubleTestRangeN3-9_M4_Oct5_' + str(loop) + '.csv', 'w') as out:
            for i, N in enumerate(Ns):
                # print(i, " ", N)
                out.write('%d,%d,%f,%f,%f,%f,%f,%f\n' % (
                N, nOfIterations[i], targetUtilities[i], platformReward[i], platformUtilperReward[i], maxUtility[i],
                minUtility[i], averageVehicleUtility[i]))'''


        n = nStart
        loop += 1
        print("Loop: ", loop)
        averages['nOfIterations'].append(nOfIterations.copy())
        averages['targetUtilities'].append(targetUtilities.copy())
        averages['platformReward'].append(platformReward.copy())
        averages['platformUtilperReward'].append(platformUtilperReward.copy())
        averages['averageVehicleUtility'].append(averageVehicleUtility.copy())
        averages['maxUtility'].append(maxUtility.copy())
        averages['minUtility'].append(minUtility.copy())

        nOfIterations.clear()
        targetUtilities.clear()
        platformReward.clear()
        platformUtilperReward.clear()
        averageVehicleUtility.clear()
        maxUtility.clear()
        minUtility.clear()

    itera = np.array(averages['nOfIterations'][0])
    tUtil = np.array(averages['targetUtilities'][0])
    formR = np.array(averages['platformReward'][0])
    utilPerR = np.array(averages['platformUtilperReward'][0])
    averageVUtil = np.array(averages['averageVehicleUtility'][0])
    maxUtil = np.array(averages['maxUtility'][0])
    minUtil = np.array(averages['minUtility'][0])

    for i in range(1, loopMax):
        itera += np.array(averages['nOfIterations'][i])
        tUtil += np.array(averages['targetUtilities'][i])
        formR += np.array(averages['platformReward'][i])
        utilPerR += np.array(averages['platformUtilperReward'][i])
        averageVUtil += np.array(averages['averageVehicleUtility'][i])
        maxUtil += np.array(averages['maxUtility'][i])
        minUtil += np.array(averages['minUtility'][i])

    itera = itera / loopMax
    tUtil = tUtil / loopMax
    formR = formR / loopMax
    utilPerR = utilPerR / loopMax
    averageVUtil = averageVUtil / loopMax
    maxUtil = maxUtil / loopMax
    minUtil = minUtil / loopMax

    '''with open('Data4/doubleTestRangeAveraged.csv', 'w') as out:
        for i, N in enumerate(Ns):
            # print(i, " ", N)
            out.write('%d,%d,%f,%f,%f,%f,%f,%f\n' % (
                N, itera[i], tUtil[i], formR[i], utilPerR[i], maxUtil[i],
                minUtil[i], averageVUtil[i]))'''








def nashTester():
    N = 15
    M = 6
    nashExist = []
    utilityMax = []
    nash = []

    iteration = 0
    while len(nashExist) < 1:
        iteration += 1
        nashExist.append(True)
        np.random.seed(550+iteration)
        print(550+iteration)
        vP, tP = getLocalizedPlacement(N, M)
        dist = distance(vP, tP)  # initialize distance object
        nA = nashAssigner(N=N, M=M, R=3, tau=1, dist=dist)  # initialize nash equilibrium algorithm
        originalMain(N, M, vP, tP, nA)
        algTarget = nA.targets
        if nA.R != 3 or nA.iterations >= 26:
            continue




    print(nashExist)
    print(utilityMax)
    print(nash)

    # maxUtilPercent = np.sum(utilityMax) / len(utilityMax)
    # with open('doubleTesting5V3P20Trials2.csv', 'w') as out:
    #     out.write('%f\n' %(maxUtilPercent))



# iterativeTargets, permUtilities = otherMain(N, M, vP, tP)
        # present = False
        #
        # for alloc in iterativeTargets:
        #     differences = algTarget == alloc
        #     if differences.all() == True:
        #         nashExist.append(True)
        #         break
        # if len(permUtilities) != 0:
        #     if np.amax(permUtilities) > np.mean(getTargetUtilities(nA)):
        #         utilityMax.append(False)
        #     else:
        #         utilityMax.append(True)
        #     nash.append(True)
        # else:
        #     nash.append(False)


def printLargeVehicleRegimeTestingData():
    with open('Data10/doubleTestRangeAveraged.csv','r') as inp:
        lines = inp.read().splitlines()

    Ns = []
    nOfIterations = []
    targetUtilities = []
    platformReward = []
    platformUperR = []
    vUtility = []
    maxUtility = []
    minUtility = []

    for line in lines:
        tokens = line.split(',')
        Ns.append(int(tokens[0]))
        nOfIterations.append(int(tokens[1]))
        targetUtilities.append(float(tokens[2]))
        platformReward.append(float(tokens[3]))
        platformUperR.append(float(tokens[4]))
        maxUtility.append(float(tokens[5]))
        minUtility.append(float(tokens[6]))
        vUtility.append(float(tokens[7]))


    plt.figure()
    plt.plot(Ns, platformUperR)
    plt.axis([np.amin(Ns)-1, np.amax(Ns)+1, np.amin(platformUperR), np.amax(platformUperR)])
    plt.xlabel('Number Of Vehicles')
    plt.ylabel('Platform Utility per Reward')
    plt.title('Platform Utility per Reward as N increases')

    plt.figure()
    plt.plot(Ns, platformReward)
    plt.axis([np.amin(Ns)-1, np.amax(Ns)+1, np.amin(platformReward), np.amax(platformReward)])
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Total Dollars Given')
    plt.title('Platform Reward given out as N increases')

    plt.figure()
    plt.plot(Ns, targetUtilities)
    plt.axis([np.amin(Ns)-1, np.amax(Ns)+1, np.amin(targetUtilities), np.amax(targetUtilities)])
    plt.xlabel('Number Of Vehicles')
    plt.ylabel('Platform Utility')
    plt.title('Platform Utility as N increases')

    plt.figure()
    plt.plot(Ns, vUtility)
    plt.axis([np.amin(Ns)-1, np.amax(Ns)+1, np.amin(vUtility), np.amax(vUtility)])
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Average Vehicle Utility")
    plt.title("Vehicle Utility as N increases")

    plt.figure()
    plt.plot(Ns, maxUtility)
    plt.plot(Ns, minUtility)
    plt.axis([np.amin(Ns)-1, np.amax(Ns)+1, np.amin(minUtility), np.amax(maxUtility)])
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Target Utility')
    plt.title('Max and Min Utility of targets as N increases')
    plt.show()



def main():
    # averageTest()
    #largeVehicleRegimeTesting()
    printLargeVehicleRegimeTestingData()



if __name__ == '__main__':
    main()