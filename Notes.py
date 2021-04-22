# Fix detach, succesor, getBestTarget to include single target situations
# Fix the tuple return for successor
# Fix the single element tuple allocation in the initialization. Single element e because (e, e) MIGHT NOT BE A PROBLEM






# INSERT NEW FUNCTION THAT CALCULALES TRAVEL TIME BETWEEN ONE LOCATION GENERAL AND ANOTHER
    # def travelTimeUpdate(self, vehicle, target, source, dest):
    #     vehicle = int(vehicle)
    #     if not np.isnan(target[0]): targetA=int(target[0])
    #     else: targetA = target[0]
    #     if not np.isnan(target[1]):  targetB=int(target[1])
    #     else: targetB = target[1]
    #
    #     if source:
    #         return spd.pdist(self.vP[2*vehicle], self.tP[targetA])
    #     elif dest and np.isnan(targetB):
    #         return spd.pdist(self.tP[targetA], self.vP[2 * vehicle + 1])
    #     elif dest and not np.isnan(targetB):
    #         return spd.pdist(self.tP[targetB], self.vP[2 * vehicle + 1])
    #     else:
    #         return spd.pdist(self.tP[targetA], self.tP[targetB])


#     def travelTime(self, vehicle, path):
#         distance = 0
#         currentPosition = self.vP[2*vehicle]
#         for t in path:
#             distance += spd.pdist([currentPosition, self.tP[t]]) # Check if t is target or target index!!!
#             currentPosition = self.tP[t]
#         distance += spd.pdist([currentPosition, self.vP[2*vehicle+1]])
#         return distance


# def __init__(self, N, M, R, tau, dist=None, arrivalTime={}, decisionTime=0):  # initialize
#     self.N = N
#     self.M = M
#     self.R = R
#     self.tau = tau
#     self.dist = dist
#     self.arrivalTime = arrivalTime
#     self.decisionTime = decisionTime
#     # initalize arrival times at all targets to empty sets
#     for t in range(self.M): self.arrivalTime[t] = []
#     self.targets = np.zeros(self.N, dtype='float')  # initalize by assigning random targets
#
#     collection = set(permutations(range(0, M), 2))
#     collection.extend(range(0, M))
#     for i in range(self.N):  # assign random targets as initialization
#         index = np.random.randint(low=0, high=len(collection))
#
#         t1 = np.random.randint(low=0, high=self.M)  # random target
#         t2 = np.random.randint(low=0, high=self.M)  # random second target
#         while (t2 == t1):  # In case a vehicle gets both targets set to the same place
#             t2 = np.random.randint(low=0, high=self.M)
#         at = np.zeros(2)
#         at[0] = self.decisionTime + self.dist.travelTime(vehicle=i, target=t1, source=True)
#         at[1] = self.dist.travelTime(vehicle=t1, target=t2, source=False, dest=False) + at[0]
#         self.arrivalTime[t1].append((i, at[0]))
#         self.arrivalTime[t2].append((i, at[1]))
#         self.targets[i] = [t1, t2]
#     # sort the arrival times at all targets
#     for t in self.arrivalTime:
#         self.arrivalTime[t].sort(key=lambda x: x[1], reverse=True)
#     self.utilities = np.zeros(N)