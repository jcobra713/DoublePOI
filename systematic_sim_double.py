#import packages 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.spatial.distance as spd 
import scipy.stats as stats


#define class for distance function 
class distance: 
	#class used for arrival time and distance calculations
	vP = None 
	tP = None
	def __init__(self,vP,tP):
		self.vP = vP
		self.tP = tP 
	def travelTime(self,vehicle,targetA,targetB=None,source=False): #return the distance between ith source/destination and jth target. s indicates source or destination (assuming unit speed). Return the first half if source is True, and return the second half if source is False. 
		vehicle = int(vehicle)
		target = int(targetA)
		if(targetB != None):
			return spd.pdist([self.tP[target],self.tP[targetB]])
		else:
			if source:
				return spd.pdist([self.vP[2*vehicle],self.tP[target]])
			else: 
				return spd.pdist([self.tP[target],self.vP[2*vehicle+1]])
			
			
	def cost(self,vehicle,target):#cost for vehicle to take the diversion to target
		vehicle = int(vehicle)
		if not np.isnan(target[0]): targetA = int(target[0])
		else: targetA = target[0]
		if not np.isnan(target[1]): targetB = int(target[1])
		else: targetB = target[1]
		sp = spd.pdist([self.vP[2*vehicle], self.vP[2*vehicle+1]])
		if np.isnan(targetA) and np.isnan(targetB):
			c = 0
		elif np.isnan(targetA):
			d1 = self.travelTime(vehicle=vehicle,targetA=targetB,source=True)
			d2 = self.travelTime(vehicle=vehicle,targetA=targetB,srouce=False)
			c= d1+d2-sp
		elif np.isnan(targetB):
			d1 = self.travelTime(vehicle=vehicle,targetA=targetA,source=True)
			d2 = self.travelTime(vehicle=vehicle,targetA=targetA,source=False)
			c= d1+d2-sp
		else:
			d1 = self.travelTime(vehicle=vehicle,targetA=targetA,source=True) #calculate distance from source to target 
			d2 = self.travelTime(vehicle=vehicle,targetA=targetA,targetB=targetB)
			d3 = self.travelTime(vehicle=vehicle,targetA=targetA,source=False) #calculate distance from target to destination 
			c = d1+d2+d3-sp#cost c
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
		self.targets = [] #initalize by assigning random targets 
		for t in range(self.N): self.targets.append([])
		for i in range(self.N): #assign random targets as initialization
			t = np.random.randint(low=0,high=self.M) #random target 
			tB = np.random.randint(low=0,high=self.M) #second random target
			while tB == t:
				tB = np.random.randint(low=0,high=self.M)
				
			at = np.zeros(2)
			at[0] = self.decisionTime + self.dist.travelTime(vehicle=i,targetA=t,source=True)
			at[1] = at[0]+self.dist.travelTime(vehicle=i,targetA=t,targetB=tB)
			self.arrivalTime[t].append((i,at[0]))
			self.arrivalTime[tB].append((i,at[1]))
			self.targets[i] = [t,tB]
		#sort the arrival times at all targets 
		for t in self.arrivalTime:
			self.arrivalTime[t].sort(key=lambda x: x[1],reverse=True)
		self.utilities = np.zeros(self.N) 
			
	
	def getUtility(self,si,t,tB): #returns utility of vehicle si to go to target t (dependent on state of other vehicles)
		at = np.zeros(2)
		if np.isnan(t) and np.isnan(tB): 
			return 0
		elif np.isnan(t):
			at[0] = 0
			at[1] = self.decisionTime + self.dist.travelTime(vehicle=si,targetA = tB,source = True)
			rewardA = 0;
			try:
				pat = next(x[1] for x in self.arrivalTime[tB] if x[1] < at[1])
				rewardB = self.R*(1-np.e**(-(at[1]-pat)/self.tau))
			except StopIteration:
				rewardB = self.R
		elif np.isnan(tB):
			at[0] = self.decisionTime + self.dist.travelTime(vehicle=si,targetA=t,source = True)
			at[1] = 0
			try:
				pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at[0]) #get the previous arrival time at t
				rewardA = self.R*(1-np.e**(-(at[0]-pat)/self.tau))
			except StopIteration:
				rewardA = self.R 
			rewardB = 0;
		else:
			at[0] = self.decisionTime + self.dist.travelTime(vehicle=si,targetA=t,source=True)#calculate arrival time of si at t 
			at[1] = at[0] + self.dist.travelTime(vehicle=si,targetA=t,targetB=tB)
			#calculate reward according to previous arrival time at t
			try:
				pat = next(x[1] for x in self.arrivalTime[t] if x[1] < at[0]) #get the previous arrival time at t
				rewardA = self.R*(1-np.e**(-(at[0]-pat)/self.tau))
			except StopIteration:
				rewardA = self.R 
			try:
				pat = next(x[1] for x in self.arrivalTime[tB] if x[1] < at[1])
				rewardB = self.R*(1-np.e**(-(at[1]-pat)/self.tau))
			except StopIteration:
				rewardB = self.R
		cost = self.dist.cost(vehicle=si,target=[t,tB])#calculate cost 
		return rewardA + rewardB - cost 
	
	
	def getBestTarget(self,si):
		u = []
		for i in range(self.M):
			u.append([])
			u[i] = np.zeros(self.M)#array to store utilities 
		for t in range(self.M):#for each target 
			for tB in range(self.M):
				if t == tB: 
					u[t][tB] = self.getUtility(si,t,float("nan"))
				u[t][tB] = self.getUtility(si,t,tB)#calculate utility 
		i = np.argmax(u)
		out = [int(np.floor(i/10)),i%10]
		if out[0] == out[1]: return [out[0],float("nan")]
		return out#return best target pair
		
	
	def pair(self,si,t,index): #pair vehicle si with target t 
		tc = self.targets[si][index] #current target]
		if tc == t: return #move along, nothing to do here
		if not np.isnan(tc):
			#remove si and its arrival time from current target 
			sip = next(x for x in self.arrivalTime[tc] if x[0] == si) #vehicle, arrival time pair 
			self.arrivalTime[tc].remove(sip)
		#add si and its arrival time at the new target 
		if np.isnan(t): 
			self.targets[si][index] = float("nan")
			return
		if index == 0:
			at = self.decisionTime + self.dist.travelTime(vehicle=si,targetA=t,source=True)
		elif index == 1:
			tE = self.targets[si][0]
			at = self.decisionTime + self.dist.travelTime(vehicle=si,targetA=tE,source=True) + self.dist.travelTime(vehicle=si,targetA=tE,targetB=t)
		self.arrivalTime[t].append((si,at[0]))
		self.arrivalTime[t].sort(key = lambda x: x[1], reverse=True)
		#assign the new target 
		self.targets[si][index] = t 
		
	def detach(self,si,index): #detaches the vehicle si from its current target 
		tc = self.targets[si][index]
		if np.isnan(tc): return 
		sip = next(x for x in self.arrivalTime[tc] if x[0] == si) #vehicle, arrival time pair 
		self.arrivalTime[tc].remove(sip)
		self.targets[si][index] = np.nan 


	
	def successor(self,si,t,index): #get the vehicle arriving after si at target t
		if np.isnan(t):return None
		if index == 0:
			at = self.decisionTime + self.dist.travelTime(vehicle=si,targetA=t,source=True)
		elif index == 1:
			tE = self.targets[si][0]
			at = self.decisionTime + self.dist.travelTime(vehicle=si,targetA=tE,source=True) + self.dist.travelTime(vehicle=si,targetA=tE,targetB=t)
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
				#print(tk) debug
				tick = False
				if t1[0] != tk[0] and tk[0] != t1[1]: 
					tick = True
					self.pair(si,tk[0],0) 
					sj = self.successor(si,tk[1],0)
					# print(si,t1,tk,sj)
					if sj: S.add(sj) 
					transitionSequence.append(np.copy(self.targets))
				elif t1[1] != tk[1] and tk[1] != t1[0]:
					tick = True
					self.pair(si,tk[1],1)
					sj = self.successor(si,tk[1],1)
					# print(si,t1,tk,sj)
					if sj: S.add(sj)
					transitionSequence.append(np.copy(self.targets))
				if tick: 
					count += 1
			endingAssignment = np.copy(self.targets)
			################################### testing with potential nan targets
			#if np.sum(startingAssignment==endingAssignment)==self.N and count > 0: #detected a cycle
			testSum = 0
			for t in range(len(startingAssignment)):
				testBool = False
				for i in range(len(startingAssignment[t])):
					if(startingAssignment[t][i]==endingAssignment[t][i]) or (np.isnan(startingAssignment[t][i]) and np.isnan(endingAssignment[t][i])):
						testBool = True
				if testBool: testSum += 1
			if testSum==self.N and count > 0:
			####################################
				print('detected a cycle')
				#for each transition, find the cost to go to the new location 
				edgeCosts = []
				sumCosts = 0
				for i in range(len(transitionSequence)-1):
					#find the vehicle which is tranisition
					tVehicle,*excess= np.where(transitionSequence[i]!=transitionSequence[i+1])  
					# print(tVehicle) #Debug
					#find the new target 
					for tVeh in tVehicle:
						tTarget = transitionSequence[i+1][tVeh]
						sumCosts += self.dist.cost(vehicle=tVeh,target=tTarget)
					edgeCosts.append(sumCosts)
				self.R = max(edgeCosts)-0.01
				print('adusted reward to %f'%self.R)
			# print(self.targets)
				
			
		self.iterations = iteration
		#calculate utilities at the Nash equilibrium
		for s in range(self.N):
			self.utilities[s] = self.getUtility(s,self.targets[s][0],self.targets[s][1])
			if self.utilities[s] < 0: 
				self.detach(s,0)
				self.detach(s,1)
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
		#print(self.arrivalTime[t]) #Debug
		if not self.arrivalTime[t]: return 0 
		ats = [p[1] for p in self.arrivalTime[t]]
		ot = ats[0] + np.mean(ats) #end of observation period 
		ats = [ot] + ats 
		ats.append(self.decisionTime)
		ats = np.array(ats)
		its = ats[:-1] - ats[1:] #time intervals 
		its = its/np.sum(its) #normalize to make probabilities
		return stats.entropy(its,base=2)#/np.log2(len(its))
		
		
		
		
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
	Ns = range(20,100)
	M = 10
	
	nOfIterations = []
	targetUtilities = []
	for N in Ns:
		print('running smart assignment, M=%d, N=%d'%(M,N))
		vP,tP = getLocalizedPlacement(N,M)
		dist = distance(vP,tP)#initialize distance object
		nA = nashAssigner(N=N,M=M,R=5,tau=1,dist=dist) #initialize nash equilibrium algorithm
		nA.getAssignments()#get target assignments 
		nOfIterations.append(nA.iterations)
		targetUtilities.append(np.mean(getTargetUtilities(nA)))
		print(nA.iterations)
	
	with open('largeVehicleRegimeTesting_1.csv','w') as out: 
		for i,N in enumerate(Ns):
			out.write('%d,%d,%f\n'%(N,nOfIterations[i],targetUtilities[i]))


def printLargeVehicleRegimeTestingData():
	with open('largeVehicleRegimeTesting_1.csv','r') as inp:
		lines = inp.read().splitlines()
	Ns = []
	nOfIterations = []
	targetUtilities = []
	for line in lines: 
		tokens = line.split(',')
		Ns.append(int(tokens[0]))
		nOfIterations.append(int(tokens[1]))
		targetUtilities.append(float(tokens[2]))
		
	fig,axs = plt.subplots(2,1,sharex=True) 
	axs[0].plot(Ns,nOfIterations)
	axs[1].plot(Ns,targetUtilities)
	axs[1].set_xlabel('number of vehicles')
	axs[0].set_ylabel('number of iterations')
	axs[1].set_ylabel('average utility')
	axs[0].set_title('number of targets M=10')
	plt.show()

def doublePoiTesting():
	N = 20
	M = 10
	verbose = False


	#print(nA.arrivalTime[0])
	percentSum = 0
	for i in range(100):
		nashCount = 0
		vP,tP = getLocalizedPlacement(N,M)
		dist = distance(vP,tP)
		nA = nashAssigner(N=N,M=M,R=5,tau=1,dist=dist)
		nA.getAssignments()
		if verbose:
			print(nA.targets)
		for veh in range(N):
			pair = nA.getBestTarget(veh)
			util = nA.getUtility(veh,pair[0],pair[1])
			if (pair[0] == nA.targets[veh][0] or (np.isnan(pair[0]) and np.isnan(nA.targets[veh][0]))) and (pair[1] == nA.targets[veh][1] or (np.isnan(pair[1]) and np.isnan(nA.targets[veh][1]))): nashCount += 1
			if verbose:
				print("%s %s %s" % (nA.targets[veh],pair,util))
		if verbose:
			for veh in range(N):
				print("Vehicle %s Utility: %s" %(veh,nA.getUtility(veh,nA.targets[veh][0],nA.targets[veh][1])))
		percent = nashCount*100/N
		if verbose:
			print("Optimal target Percentage: %s%%" % (percent))
		percentSum += percent
	print("Average Optimal Target Percentage: %s%%" % (percentSum/100))


	return

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
	doublePoiTesting()
	#largeVehicleRegimeTesting()
	#printLargeVehicleRegimeTestingData()
	
	
	
if __name__ == '__main__':
	main()






