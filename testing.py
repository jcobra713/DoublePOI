import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd
import scipy.stats as stats
from itertools import permutations

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



