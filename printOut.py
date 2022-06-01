import numpy as np
import matplotlib.pyplot as plt


def printLargeVehicleRegimeTestingData():
	with open('SinglePOINew/averagedSingleN5_20_R3.csv','r') as inp:
		lines = inp.read().splitlines()

	with open('Data10/doubleTestRangeAveraged.csv', 'r') as inp:
		linesDouble = inp.read().splitlines()

	# Setup for Single
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

	# Setup for Double
	Ns2 = []
	nOfIterations2 = []
	targetUtilities2 = []
	platformReward2 = []
	platformUperR2 = []
	vUtility2 = []
	maxUtility2 = []
	minUtility2 = []
	for line in linesDouble:
		tokens = line.split(',')
		Ns2.append(int(tokens[0]))
		nOfIterations2.append(int(tokens[1]))
		targetUtilities2.append(float(tokens[2]))
		platformReward2.append(float(tokens[3]))
		platformUperR2.append(float(tokens[4]))
		maxUtility2.append(float(tokens[5]))
		minUtility2.append(float(tokens[6]))
		vUtility2.append(float(tokens[7]))

	fig,axs = plt.subplots(2,1,sharex=True)
	axs[0].set_ylim([0, 26])
	axs[0].plot(Ns,nOfIterations, color='red')
	axs[0].plot(Ns2, nOfIterations2, color="blue")
	axs[1].plot(Ns, targetUtilities, color='red')
	axs[1].plot(Ns2,targetUtilities2, color="blue")
	axs[1].set_xlabel('number of vehicles')
	axs[0].set_ylabel('number of iterations')
	axs[1].set_ylabel('average utility')
	axs[0].set_title('number of targets M=10')

	plt.figure()
	plt.plot(Ns, platformReward, color='red')
	plt.plot(Ns2,platformReward2, color="blue")
	#plt.axis([np.amin(Ns), np.amax(Ns), np.amin(platformReward), np.amax(platformReward)+0.2])
	plt.xlabel('Number of vehicles')
	plt.ylabel('Total Dollars Given')
	plt.title('Platform Reward given out as N increases')

	fig2,axs2 = plt.subplots(2, 1, sharex=True)
	axs2[0].plot(Ns, platformUperR, color='red')
	axs2[0].plot(Ns2, platformUperR2, color="blue")
	axs2[1].plot(Ns, vUtility, color='red')
	axs2[1].plot(Ns2, vUtility2, color="blue")
	#axs2[0].axis([np.amin(Ns), np.amax(Ns), 0.0225, 0.04])
	#axs2[1].axis([np.amin(Ns), np.amax(Ns), np.amin(vUtility), np.amax(vUtility2)])
	axs2[0].set_xlabel('number of vehicles')
	axs2[0].set_ylabel('Platform Utility per Dollar')
	axs2[1].set_xlabel('number of vehicles')
	axs2[1].set_ylabel('Average Vehicle Utility')
	axs2[0].set_title('Reward as N increases')

	plt.figure()
	plt.plot(Ns, platformUperR, color='red')
	plt.plot(Ns2, platformUperR2, color="blue")
	#plt.axis([np.amin(Ns), np.amax(Ns), np.amin(platformUperR), 0.04])
	plt.xlabel('number of vehicles')
	plt.ylabel('Platform Utility per Dollar')
	plt.title('Platform U per R')

	plt.figure()
	plt.plot(Ns, maxUtility, color='red')
	plt.plot(Ns2, maxUtility2, color="blue")
	plt.plot(Ns, minUtility, color='red')
	plt.plot(Ns2, minUtility2, color="blue")
	#plt.axis([np.amin(Ns), np.amax(Ns), np.amin(minUtility)-0.1, np.amax(maxUtility2)+0.2])
	plt.xlabel('number of vehicles')
	plt.ylabel('Target Utility')
	plt.title('Max and Min Utility of targets as N increases')
	plt.show()



printLargeVehicleRegimeTestingData()
print("Details: M=4 N=5-40 R=3")