import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd
import scipy.stats as stats
import random


class distance:
    # class used for arrival time and distance calculations
    vP = None
    tP = None

    def __init__(self, vP, tP):
        self.vP = vP
        self.tP = tP

    # def travelTime(self, vehicle, target, source):  # return the distance between ith source/destination and jth target. s indicates source or destination (assuming unit speed). Return the first half if source is True, and return the second half if source is False.
    #     vehicle = int(vehicle)
    #     target = int(target)
    #     if source:
    #         return spd.pdist([self.vP[2 * vehicle], self.tP[target]])
    #     else:
    #         return spd.pdist([self.tP[target], self.vP[2 * vehicle + 1]])

# 



    # INSERT NEW FUNCTION THAT CALCULALES TRAVEL TIME BETWEEN ONE LOCATION GENERAL AND ANOTHER
    def travelTimeUpdate(self, vehicle, target, source, dest):
        vehicle = int(vehicle)
        if not np.isnan(target[0]): targetA=int(target[0])
        else: targetA = target[0]
        if not np.isnan(target[1]):  targetB=int(target[1])
        else: targetB = target[1]

        if source:
            return spd.pdist(self.vP[2*vehicle], self.tP[targetA])
        elif dest and np.isnan(targetB):
            return spd.pdist(self.tP[targetA], self.vP[2 * vehicle + 1])
        elif dest and not np.isnan(targetB):
            return spd.pdist(self.tP[targetB], self.vP[2 * vehicle + 1])
        else:
            return spd.pdist(self.tP[targetA], self.tP[targetB])


    # THIS WAS CHANGED TO USE 2 TARGETS
    def cost(self, vehicle, target):  # cost for vehicle to take the diversion to targets
        vehicle = int(vehicle)
        if not np.isnan(target[0]): targetA=int(target[0])
        else: targetA = target[0]
        if not np.isnan(target[1]):  targetB=int(target[1])
        else: targetB = target[1]
        sp = spd.pdist([self.vP[2 * vehicle], self.vP[2 * vehicle + 1]])
        if np.isnan(targetA) and np.isnan(targetB): # No targets are accessed
            c = 0
        elif not np.isnan(targetA) and np.isnan(targetB):
            d1 = self.travelTimeUpdate(vehicle=vehicle, target=targetA, source=True, dest=False)  # calculate distance from source to target
            d2 = self.travelTimeUpdate(vehicle=vehicle, target=targetA, source=False, dest=True)  # calculate distance from target to destination
            c = d1 + d2 - sp
        elif np.isnan(target) and not np.isnan(targetB):
            d1 = self.travelTimeUpdate(vehicle=vehicle, target=targetB, source=True, dest=False)  # calculate distance from source to target
            d2 = self.travelTimeUpdate(vehicle=vehicle, target=targetB, source=False, dest=True)  # calculate distance from target to destination
            c = d1 + d2 - sp
        else:
            d1 = self.travelTimeUpdate(vehicle=vehicle, target=targetA, source=True, dest=False)  # calculate distance from source to first target
            d2 = self.travelTimeUpdate(vehicle=targetA, target=targetB, source=False, dest=False) # calculate distance from first target to second target
            d3 = self.travelTimeUpdate(vehicle=vehicle, target=targetB, source=False, dest=True)  # calculate distance from second target to destination
            c = d1 + d2 + d3 - sp  # cost c
        return c




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
        self.targets = np.zeros(self.N, dtype='float')  # initalize by assigning random targets
        for i in range(self.N):  # assign random targets as initialization
            t1 = np.random.randint(low=0, high=self.M)  # random target
            t2 = np.random.randint(low=0, high=self.M) # random second target
            while (t2 == t1): # In case a vehicle gets both targets set to the same place
                t2 = np.random.randint(low=0, high=self.M)
            at = np.zeros(2)
            at[0] = self.decisionTime + self.dist.travelTime(vehicle=i, target=t1, source=True)
            at[1] = self.dist.travelTime(vehicle=t1, target=t2, source=False, dest=False) + at[0]
            self.arrivalTime[t1].append((i, at[0]))
            self.arrivalTime[t2].append((i, at[1]))
            self.targets[i] = [t1, t2]
        # sort the arrival times at all targets
        for t in self.arrivalTime:
            self.arrivalTime[t].sort(key=lambda x: x[1], reverse=True)
        self.utilities = np.zeros(N)



    # CHANGING
    def getUtility(self, si,t):  # returns utility of vehicle si to go to target t (dependent on state of other vehicles)
        at = self.decisionTime + self.dist.travelTime(vehicle=si, target=t, source=True)  # calculate arrival time of si at t
        # calculate reward according to previous arrival time at t
        try:
            pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at)  # get the previous arrival time at t
            reward = self.R * (1 - np.e ** (-(at - pat) / self.tau))
        except StopIteration:
            reward = self.R
        cost = self.dist.cost(vehicle=si, target=t)  # calculate cost
        return reward - cost

    def getUtilities(self):


    def getReward(self, si,
                  t):  # returns reward given to vehicle si to go to target t (dependent on state of other vehicles)
        at = self.decisionTime + self.dist.travelTime(vehicle=si, target=t,
                                                      source=True)  # calculate arrival time of si at t
        # calculate reward according to previous arrival time at t
        try:
            pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at)  # get the previous arrival time at t
            reward = self.R * (1 - np.e ** (-(at - pat) / self.tau))
        except StopIteration:
            reward = self.R
        return reward

    def getBestTarget(self, si):
        u = np.zeros(self.M)  # array to store utilities
        for t in range(self.M):  # for each target
            u[t] = self.getUtility(si, t)  # calculate utility
        return np.argmax(u)  # return best target

    def pair(self, si, t):  # pair vehicle si with target t
        tc = self.targets[si]  # current target
        if tc == t: return  # move along, nothing to do here
        # remove si and its arrival time from current target
        sip = next(x for x in self.arrivalTime[tc] if x[0] == si)  # vehicle, arrival time pair
        self.arrivalTime[tc].remove(sip)
        # add si and its arrival time at the new target
        at = self.decisionTime + self.dist.travelTime(vehicle=si, target=t, source=True)
        self.arrivalTime[t].append((si, at))
        self.arrivalTime[t].sort(key=lambda x: x[1], reverse=True)
        # assign the new target
        self.targets[si] = t

    def detach(self, si):  # detaches the vehicle si from its current target
        tc = self.targets[si]
        if tc == np.nan: return
        sip = next(x for x in self.arrivalTime[tc] if x[0] == si)  # vehicle, arrival time pair
        self.arrivalTime[tc].remove(sip)
        self.targets[si] = np.nan

    def successor(self, si, t):  # get the vehicle arriving after si at target t
        at = self.decisionTime + self.dist.travelTime(vehicle=si, target=t, source=True)
        try:
            succ = next(x[0] for x in reversed(self.arrivalTime[t]) if x[1] > at)
        except StopIteration:
            succ = None
        return succ

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
                if t1 != tk:
                    count += 1
                    self.pair(si, tk)
                    sj = self.successor(si, tk)
                    # print(si,t1,tk,sj)
                    if sj: S.add(sj)
                    transitionSequence.append(np.copy(self.targets))
            endingAssignment = np.copy(self.targets)
            if np.sum(startingAssignment == endingAssignment) == self.N and count > 0:  # detected a cycle
                print('detected a cycle')
                # for each transition, find the cost to go to the new location
                edgeCosts = []
                for i in range(len(transitionSequence) - 1):
                    # find the vechile which is tranisition
                    tVehicle, = np.where(transitionSequence[i] != transitionSequence[i + 1])
                    # find the new target
                    tTarget = transitionSequence[i + 1][tVehicle]
                    edgeCosts.append(self.dist.cost(vehicle=tVehicle, target=tTarget))
                self.R = max(edgeCosts) - 0.01
                print('adusted reward to %f' % self.R)
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
        ats = [p[1][0] for p in self.arrivalTime[t]]
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

