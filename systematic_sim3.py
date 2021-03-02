#import packages 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd 
import scipy.stats as stats

# np.random.seed(7)
#define class for distance function 
class distance: 
	#class used for arrival time and distance calculations
	vP = None 
	tP = None
	def __init__(self,vP,tP):
		self.vP = vP
		self.tP = tP 
	def travelTime(self,vehicle,target,source): #return the distance between ith source/destination and jth target. s indicates source or destination (assuming unit speed). Return the first half if source is True, and return the second half if source is False. 
		vehicle = int(vehicle)
		target = int(target)
		if source:
			return spd.pdist([self.vP[2*vehicle],self.tP[target]])
		else: 
			return spd.pdist([self.tP[target],self.vP[2*vehicle+1]])
			
			
	def cost(self,vehicle,target):#cost for vehicle to take the diversion to target
		vehicle = int(vehicle)
		target = int(target)
		d1 = self.travelTime(vehicle=vehicle,target=target,source=True) #calculate distance from source to target 
		d2 = self.travelTime(vehicle=vehicle,target=target,source=False) #calculate distance from target to destination 
		sp = spd.pdist([self.vP[2*vehicle], self.vP[2*vehicle+1]])
		c = d1+d2-sp#cost c
		return c


class nashAssigner:
	N = None #number of vehicles 
	M = None #number of targets 
	R = None #max reward per sample
	tau = None 
	dist = None #object of class used for arrival time and distance calculations
	vP = None 
	arrivalTime = None #dictionary that maintains sequence of vehicle numbers and their arrival times
	decisionTime = None 
	targets = None 
	utilities = None 
	iterations = None 
	
	def __init__(self,N,M,R,tau,dist=None,arrivalTime={},decisionTime=0): #initialize 
		self.N = N 
		self.M = M 
		self.R = R
		self.tau = tau 
		self.dist = dist 
		self.arrivalTime = arrivalTime
		self.decisionTime = decisionTime
		#initalize arrival times at all targets to empty sets
		for t in range(self.M): self.arrivalTime[t] = []
		self.targets = np.zeros(self.N,dtype='float') #initalize by assigning random targets 
		for i in range(self.N): #assign random targets as initialization
			t = np.random.randint(low=0,high=self.M) #random target 
			at = self.decisionTime + self.dist.travelTime(vehicle=i,target=t,source=True)
			self.arrivalTime[t].append((i,at))
			self.targets[i] = t
		#sort the arrival times at all targets 
		for t in self.arrivalTime:
			self.arrivalTime[t].sort(key=lambda x: x[1],reverse=True)
		self.utilities = np.zeros(N) 
			
	
	def getUtility(self,si,t): #returns utility of vehicle si to go to target t (dependent on state of other vehicles)
		at = self.decisionTime + self.dist.travelTime(vehicle=si,target=t,source=True)#calculate arrival time of si at t 
		#calculate reward according to previous arrival time at t
		try:
			pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at) #get the previous arrival time at t
			reward = self.R*(1-np.e**(-(at-pat)/self.tau))
		except StopIteration:
			reward = self.R 
		cost = self.dist.cost(vehicle=si,target=t)#calculate cost 
		return reward - cost


	def getReward(self,si,t): #returns reward given to vehicle si to go to target t (dependent on state of other vehicles)
		at = self.decisionTime + self.dist.travelTime(vehicle=si,target=t,source=True)#calculate arrival time of si at t
		#calculate reward according to previous arrival time at t
		try:
			pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at) #get the previous arrival time at t
			reward = self.R*(1-np.e**(-(at-pat)/self.tau))
		except StopIteration:
			reward = self.R
		return reward


	
	def getBestTarget(self,si):
		u = np.zeros(self.M)#array to store utilities 
		for t in range(self.M):#for each target 
			u[t] = self.getUtility(si,t)#calculate utility 
		return np.argmax(u)#return best target 
		
	
	def pair(self,si,t): #pair vehicle si with target t 
		tc = self.targets[si] #current target
		if tc == t: return #move along, nothing to do here
		#remove si and its arrival time from current target 
		sip = next(x for x in self.arrivalTime[tc] if x[0] == si) #vehicle, arrival time pair 
		self.arrivalTime[tc].remove(sip)
		#add si and its arrival time at the new target 
		at = self.decisionTime + self.dist.travelTime(vehicle=si,target=t,source=True)
		self.arrivalTime[t].append((si,at))
		self.arrivalTime[t].sort(key = lambda x: x[1], reverse=True)
		#assign the new target 
		self.targets[si] = t 
		
	def detach(self,si): #detaches the vehicle si from its current target 
		tc = self.targets[si] 
		if tc == np.nan: return 
		sip = next(x for x in self.arrivalTime[tc] if x[0] == si) #vehicle, arrival time pair 
		self.arrivalTime[tc].remove(sip)
		self.targets[si] = np.nan 


	
	def successor(self,si,t): #get the vehicle arriving after si at target t
		at = self.decisionTime + self.dist.travelTime(vehicle=si,target=t,source=True)
		try: 
			succ = next(x[0] for x in reversed(self.arrivalTime[t]) if x[1] > at)
		except StopIteration:
			succ = None 
		return succ 

	
	def closestTargetTo(self,si):
		distances = np.zeros(self.M) 
		for t in range(self.M):
			distances[t] = self.dist.travelTime(vehicle=si,target=t,source=True)
		return np.argmin(distances)	
	
	
	
	#function takes in a given configuration and returns target assignments 
	def getAssignments(self): 
		#Returns: (targets, utilities)
		#targets[x] is the target allocation for vehicle x. If no target is assigned to x, then targets[x] will be equal to np.nan		
		#utilities[x] contains the utility of the vehicle x 
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
				si = S.pop() #pick random vehicle 
				#si = list(S)[np.random.randint(len(S))]
				#S.remove(si)
				t1 = self.targets[si]
				tk = self.getBestTarget(si)#  arg max_l u(si,tl)
				if t1 != tk: 
					count += 1
					self.pair(si,tk) 
					sj = self.successor(si,tk)
					# print(si,t1,tk,sj)
					if sj: S.add(sj) 
					transitionSequence.append(np.copy(self.targets))
			endingAssignment = np.copy(self.targets) 
			if np.sum(startingAssignment==endingAssignment)==self.N and count > 0: #detected a cycle
				print('detected a cycle')
				#for each transition, find the cost to go to the new location 
				edgeCosts = []
				for i in range(len(transitionSequence)-1):
					#find the vechile which is tranisition
					tVehicle, = np.where(transitionSequence[i]!=transitionSequence[i+1])  
					#find the new target 
					tTarget = transitionSequence[i+1][tVehicle]
					edgeCosts.append(self.dist.cost(vehicle=tVehicle,target=tTarget))
				self.R = max(edgeCosts)-0.01
				print('adusted reward to %f'%self.R)
			# print(self.targets)	
				
			
		self.iterations = iteration
		#calculate utilities at the Nash equilibrium
		for s in range(self.N):
			self.utilities[s] = self.getUtility(s,self.targets[s])
			if self.utilities[s] < 0: 
				self.detach(s)
		return (self.targets, self.utilities)
			
	
	def greedyAssignments(self): 
		#Returns: (targets, utilities)
		S = set(range(self.N))
		while S:
			si = S.pop()
			ti = self.closestTargetTo(si) 
			self.pair(si,ti) 
		for s in range(self.N):
			self.utilities[s] = self.getUtility(s,self.targets[s])
			if self.utilities[s] < 0: 
				self.targets[s] = np.nan 
		return (self.targets, self.utilities)
		
	def getTargetUtility(self,t):
		if not self.arrivalTime[t]: return 0 
		ats = [p[1][0] for p in self.arrivalTime[t]]
		ot = ats[0] + np.mean(ats) #end of observation period 
		ats = [ot] + ats 
		ats.append(self.decisionTime)
		ats = np.array(ats)
		its = ats[:-1] - ats[1:] #time intervals 
		its = its/np.sum(its) #normalize to make probabilities
		return stats.entropy(its,base=2)#/np.log2(len(its))

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
		allocations.append('(%d,%.0f,%.3f)'%(x,nA.targets[x],nA.utilities[x]))
	return allocations
		

def getArrivalTimes(nA):
	print('arrival times at targets')
	arrivalTimes = []
	for x in nA.arrivalTime:
		times = ['(%d,%.2f)'%(p[0],p[1][0]) for p in nA.arrivalTime[x]] 
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
	vehicleU = [] # Utility per Reward
	value = None
	for t in range(nA.N):
		if nA.targets[t] == nA.targets[t]:
			value = nA.getUtility(t, nA.targets[t])
			vehicleU.append(value[0]) # Vehicle Utility per Dollar given out
		else:
			print("Vehicle ", t, " target = nan")
			# vehicleUpR.append(0.0)
	return vehicleU

def getRandomPlacement(N,M):
	vP = np.random.random((2*N,2))#sources and destinations
	tP = np.random.random((M,2))# targets 
	return (vP,tP)
	
	
def getLocalizedPlacement(N,M):
	vP = np.zeros(shape=(2*N,2))
	for i in range(N):
		a = np.random.random((2,2))
		vP[2*i:2*i+2,:] = (1/(1.2*(np.sum(a))))* a 
		vP[2*i+1,:] = 1-vP[2*i+1,:]
		
		
	
	
	r = np.random.random(M)-0.5 #random radii in [-0.5,0.5]
	theta = np.pi/2 + (np.pi/6)*np.random.random(M) #random ange in [pi/2-pi/6, pi/2+pi/6] 
	complexCrd = r*np.e**(1j*theta) 
	tP = np.zeros(shape=(M,2))
	tP[:,0] = np.real(complexCrd)
	tP[:,1] = np.imag(complexCrd)
	tP = (0.5,0.5) + tP 
	
	
	
	return (vP,tP)

def originalMain():
	N = 100
	M = 20
	vP,tP = getRandomPlacement(N,M)
	vP,tP = getLocalizedPlacement(N,M)
	dist = distance(vP,tP)#initialize distance object  
	
	# R should be in the upper range of the diversion costs 
	print('running smart assignment')
	nA = nashAssigner(N=N,M=M,R=5,tau=1,dist=dist) #initialize nash equilibrium algorithm
	nA.getAssignments()#get target assignments 
	smartUtilities = np.array(getTargetUtilities(nA))
	
	print('running greedy  assignment')
	nA.__init__(N=N,M=M,R=5,tau=1,dist=dist)
	nA.greedyAssignments()
	greedyUtilities =  np.array(getTargetUtilities(nA))
	
	plt.figure()
	plt.plot(vP[:,0], vP[:,1], 'ro')
	plt.plot(tP[:,0], tP[:,1], 'bo')
	
	plt.figure()
	plt.scatter(greedyUtilities,smartUtilities)
	m = np.maximum(greedyUtilities.max(), smartUtilities.max())+1
	plt.plot([0,m], [0,m], 'r')
	plt.xlabel('utitlities with greedy assignment')
	plt.ylabel('utilities with smart assignment')
	plt.show()


def largeTargetRegimeTesting():
	N = 20
	Ms = range(4,50)
	
	nOfIterations = []
	for M in Ms:
		print('running smart assignment, M=%d, N=%d'%(M,N))
		vP,tP = getLocalizedPlacement(N,M)
		dist = distance(vP,tP)#initialize distance object
		nA = nashAssigner(N=N,M=M,R=5,tau=1,dist=dist) #initialize nash equilibrium algorithm
		nA.getAssignments()#get target assignments 
		if nA.iterations >= 25: 
			with open('example_M_%d_N_%d.csv'%(M,N), 'w') as out: 
				out.write('%d,%d\n'%(2*N,M))
				for x in vP: out.write('%f,%f\n'%(x[0],x[1]))
				for x in tP: out.write('%f,%f\n'%(x[0],x[1]))
				
		nOfIterations.append((M,nA.iterations))
	
	nOfIterations = np.array(nOfIterations)
	plt.stem(nOfIterations[:,0], nOfIterations[:,1])
	plt.xlabel('M')
	plt.ylabel('number of iterations')
	plt.title('N=%d'%N)
	plt.show()
	

def largeVehicleRegimeTesting():
	n = 20
	nMax = 100
	Ns = range(n, nMax+1)
	M = 10
	
	nOfIterations = []
	targetUtilities = []
	platformReward = []
	platformUtilperReward = []
	averageVehicleUtility = []
	maxUtility = []
	minUtility = []
	while n <= nMax:
		print('running smart assignment, M=%d, N=%d'%(M,n))
		vP,tP = getLocalizedPlacement(n,M)
		dist = distance(vP,tP)#initialize distance object
		nA = nashAssigner(N=n,M=M,R=5,tau=1,dist=dist) #initialize nash equilibrium algorithm
		nA.getAssignments()#get target assignments
		if (nA.iterations >= 25):
			print("Repeated")
			continue
		elif np.sum(getPlatformReward(nA)) < 30:
			print("Cycle Detected, Iteration Repeated")
			continue
		nOfIterations.append(nA.iterations)
		targetUtilities.append(np.mean(getTargetUtilities(nA)))
		platformReward.append(np.sum(getPlatformReward(nA)))
		platformUtilperReward.append(np.mean(getTargetUtilities(nA))/np.sum(getPlatformReward(nA)))
		averageVehicleUtility.append(np.mean(getVUtility(nA)))
		maxUtility.append(np.amax(getTargetUtilities(nA)))
		minUtility.append(np.amin(getTargetUtilities(nA)))
		print(nA.iterations)
		n += 1

	
	with open('largeVehicleRegimeTesting_14.csv','w') as out:
		for i,N in enumerate(Ns):
			# print(i, " ", N)
			out.write('%d,%d,%f,%f,%f,%f,%f,%f\n'%(N, nOfIterations[i], targetUtilities[i], platformReward[i], platformUtilperReward[i], maxUtility[i], minUtility[i], averageVehicleUtility[i]))


def printLargeVehicleRegimeTestingData():
	with open('largeVehicleRegimeTesting_13.csv','r') as inp:
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


	fig,axs = plt.subplots(2,1,sharex=True)
	axs[0].set_ylim([0, 26])
	axs[0].plot(Ns,nOfIterations)
	axs[1].plot(Ns,targetUtilities)
	axs[1].set_xlabel('number of vehicles')
	axs[0].set_ylabel('number of iterations')
	axs[1].set_ylabel('average utility')
	axs[0].set_title('number of targets M=10')

	plt.figure()
	plt.plot(Ns, platformReward)
	plt.axis([np.amin(Ns), np.amax(Ns), np.amin(platformReward), np.amax(platformReward)+0.2])
	plt.xlabel('Number of vehicles')
	plt.ylabel('Total Dollars Given')
	plt.title('Platform Reward given out as N increases')

	fig2,axs2 = plt.subplots(2, 1, sharex=True)
	axs2[0].plot(Ns, platformUperR)
	axs2[1].plot(Ns, vUtility)
	axs2[0].axis([np.amin(Ns), np.amax(Ns), 0.0225, 0.04])
	axs2[1].axis([np.amin(Ns), np.amax(Ns), np.amin(vUtility), np.amax(vUtility)])
	axs2[0].set_xlabel('number of vehicles')
	axs2[0].set_ylabel('Platform Utility per Dollar')
	axs2[1].set_xlabel('number of vehicles')
	axs2[1].set_ylabel('Average Vehicle Utility')
	axs2[0].set_title('Reward as N increases')

	# plt.figure()
	# plt.plot(Ns, platformUperR)
	# plt.axis([np.amin(Ns), np.amax(Ns), np.amin(platformUperR), 0.04])
	# plt.xlabel('number of vehicles')
	# plt.ylabel('Platform Utility per Dollar')
	# plt.title('Platform U per R')

	plt.figure()
	plt.plot(Ns, maxUtility)
	plt.plot(Ns, minUtility)
	plt.axis([np.amin(Ns), np.amax(Ns), np.amin(minUtility)-0.1, np.amax(maxUtility)+0.2])
	plt.xlabel('number of vehicles')
	plt.ylabel('Target Utility')
	plt.title('Max and Min Utility of targets as N increases')
	plt.show()


def exampleNonConvergence():
	#read the locations from the input file 
	with open('example_M_5_N_20.csv', 'r') as inp: 
		lines = inp.read().splitlines()
	tokens = lines[0].split(',')
	N = int(tokens[0])
	M = int(tokens[1])
	positions = [ x.split(',') 	for x in lines[1:] ]
	vP = np.array(positions[:N])
	tP = np.array(positions[N:])
	N = int(N/2)
	dist = distance(vP,tP)#initialize distance object
		
	#perform nash assignment 
	nA = nashAssigner(N=N,M=M,R=5,tau=1,dist=dist) #initialize nash equilibrium algorithm
	nA.getAssignments()
	
	
def main():
	# largeVehicleRegimeTesting()
	printLargeVehicleRegimeTestingData()
	# originalMain()
	
	
if __name__ == '__main__':
	main()






