import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd
import scipy.stats as stats
from itertools import permutations
import itertools
import random

np.random.seed(4)
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

    def getReward(self, si, t):  # returns reward given to vehicle si to go to target t (dependent on state of other vehicles)
        at = self.decisionTime + self.dist.travelTime(vehicle=si, target=t, source=True)  # calculate arrival time of si at t
        # calculate reward according to previous arrival time at t
        try:
            pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at)  # get the previous arrival time at t
            reward = self.R * (1 - np.e ** (-(at - pat) / self.tau))
        except StopIteration:
            reward = self.R
        return reward

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
            print("Count: ", count, "    Iteration: ", iteration)
            count = 0
            iteration = iteration + 1
            if iteration > 25: break
            # print('running iteration # %d'%iteration)
            S = set(range(self.N))
            startingAssignment = np.copy(self.targets)
            transitionSequence = [startingAssignment]
            while S:
                si = S.pop()  # pick random vehicle
                # si = list(S)[np.random.randint(len(S))]
                # S.remove(si)
                t1 = self.targets[si]
                tk = self.getBestTarget(si)  # arg max_l u(si,tl)
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
                    # print("Length of transitions: ", len(transitionSequence))
            endingAssignment = np.copy(self.targets)
            if np.sum(startingAssignment == endingAssignment) == self.N and count > 0:  # detected a cycle
                print('detected a cycle')
                # for each transition, find the cost to go to the new location
                edgeCosts = []
                for i in range(len(transitionSequence) - 1):
                    # find the vehicle which is transition
                    tVehicle = np.where(transitionSequence[i] != transitionSequence[i + 1])
                    # find the new target
                    tTarget = transitionSequence[i + 1][tVehicle[0][0]]
                    edgeCosts.append(self.dist.cost(vehicle=tVehicle[0][0], path=tTarget)) # Check if tTarget is passing as a tuple
                self.R = max(edgeCosts) - 0.01
                print('adusted reward to %f' % self.R)
        # print(self.targets)
        self.iterations = iteration

        if self.targets[0][0] == 2 and self.targets[0][1] == 0 and self.targets[1][0] == 0 and self.targets[1][1] == 1 and self.targets[2][0] == 1 \
            and self.targets[2][1] == 1 and self.targets[3][0] == 0 and self.targets[3][1] == 2 and self.targets[4][0] == 0 and self.targets[4][1] == 2:
            print("This is the one")
            for i in range(0, 5):
                tk = self.getBestTarget(i)
                pass

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

    def getTargetReward(self, t):
        if not self.arrivalTime[t]: return 0
        targetReward = 0
        for si, at in self.arrivalTime[t]:
            targetReward += float(self.getReward(si, t))
        return targetReward


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
    targetRewards = []
    for t in range(nA.M):
        targetRewards.append(nA.getTargetReward(t))
    return targetRewards


def getVUtility(nA):
    vehicleU = []  # Utility per Reward
    value = None
    for t in range(nA.N):
        if nA.targets[t] == nA.targets[t]:
            value = nA.getUtility(t, nA.targets[t])
            vehicleU.append(value[0])  # Vehicle Utility per Dollar given out
        else:
            print("Vehicle ", t, " target = nan")
        # vehicleUpR.append(0.0)
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
        print("Vehicle ", vehicle, " position: ", i, "    ", count % 2)

def printTp(tP):
    count = 0
    for i in tP:
        print("Target ", count, " position: ", i)
        count += 1

##################################

def otherMain():
    N = 5
    M = 3
    vP, tP = getLocalizedPlacement(N, M)
    printVp(vP)
    printTp(tP)
    dist = distance(vP,tP)
    R = 3
    tau = 1

    arrivalTime = {}  # dictionary that maintains sequence of vehicle numbers and their arrival times
    for t in range(M): arrivalTime[t] = []
    decisionTime = 0
    targets = np.zeros((N, 2), dtype='int')

    x = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    collection = [p for p in itertools.product(x, repeat=5)]
    nash = []

    # Check if nash equilibrium with getBestTarget on all vehicles. If tk = t1 then good for all
    # How far they are from greatest nash equilibrium



    targetUtilities = []
    permUtilities = []
    j = 0
    # print(collection[40000])
    for perm in collection:
        if perm[0][0] == 2 and perm[0][1] == 0 and perm[1][0] == 0 and perm[1][1] == 1 and perm[2][0] == 1 and perm[2][1] == 1 \
            and perm[3][0] == 0 and perm[3][1] == 2 and perm[4][0] == 0 and perm[4][1] == 2:
            print("This is the one")
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
        print(perm)
        permUtilities.append(np.mean(targetUtilities))
    # print(collection[np.argmax(permUtilities)])
    for n in nash:
        print(n)
    pass







def originalMain():
    N = 5
    M = 3
    vP, tP = getLocalizedPlacement(N, M)
    printVp(vP)
    printTp(tP)
    dist = distance(vP, tP)  # initialize distance object

    nOfIterations = []
    targetUtilities = []

    # R should be in the upper range of the diversion costs
    print('running smart assignment')
    nA = nashAssigner(N=N, M=M, R=3, tau=1, dist=dist)  # initialize nash equilibrium algorithm
    nA.getAssignments()  # get target assignments
    nOfIterations.append(nA.iterations)
    utility = np.mean(getTargetUtilities(nA))
    target = nA.targets
    print(target)
    plt.figure()
    plt.plot(vP[:, 0], vP[:, 1], 'ro')
    plt.plot(tP[:, 0], tP[:, 1], 'bo')
    plt.show()
    print(utility)


def main():
    # originalMain()
    otherMain()



if __name__ == '__main__':
    main()