import numpy as np

def averager():
    loopMax = 5
    averages = {}

    averages['nOfIterations'] = []
    averages['targetUtilities'] = []
    averages['platformReward'] = []
    averages['platformUtilperReward'] = []
    averages['averageVehicleUtility'] = []
    averages['maxUtility'] = []
    averages['minUtility'] = []

    for i in range(0, 5):
        with open('doubleTestRangeN3-14_M5_Sep29_' + str(i) + '.csv', 'r') as inp:
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

        averages['nOfIterations'].append(nOfIterations)
        averages['targetUtilities'].append(targetUtilities)
        averages['platformReward'].append(platformReward)
        averages['platformUtilperReward'].append(platformReward)
        averages['averageVehicleUtility'].append(vUtility)
        averages['maxUtility'].append(maxUtility)
        averages['minUtility'].append(minUtility)

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

    itera = itera / 2
    tUtil = tUtil / 2
    formR = formR / 2
    utilPerR = utilPerR / 2
    averageVUtil = averageVUtil / 2
    maxUtil = maxUtil / 2
    minUtil = minUtil / 2

    with open('doubleTestRangeAveraged.csv', 'w') as out:
        for i, N in enumerate(Ns):
            # print(i, " ", N)
            out.write('%d,%d,%f,%f,%f,%f,%f,%f\n' % (
                N, itera[i], tUtil[i], formR[i], utilPerR[i], maxUtil[i],
                minUtil[i], averageVUtil[i]))



averager()