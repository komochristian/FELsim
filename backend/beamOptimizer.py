#Author: Christian Komo

'''
helpful resources
https://www.youtube.com/watch?v=G0yP_TM-oag
https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
https://docs.scipy.org/doc/scipy/reference/optimize.html
gpt"can you xplain the parameters of a constraint paramter in scipy.minimize"
'''
#  NOTE: nelder-mead method doesn't work if starting search point is zero (if variablesValues = [0,0,0...]
#  NOTE: After testing, COBYLA and some other methods may try a test value of 0 for current, length, etc, that may throw back a difference of NAN
#        (because beamline object divides by 0). Have to figure out how to bound x variable values automatically so computer doesn't use weird values like 0, 
#        negative numbers (YOU MUST USE bounds for now so program doesn't use negative/zero numbers
#  NOTE: "measure" in self.objectives has to be a function call that returns a single value with a parameter of a
#         2d list of particles and each indice can only appear once as a key

import scipy.optimize as spo
from beamline import *
from ebeam import beam
import numpy as np
from schematic import *
import timeit
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time

# NOTE: EACH BEAMOPTIMIZER OBJECT AFTER INSTANTIATION SHOULD ONLY BE USED TO 
# RUN A CALC() FUNCTION ONE TIME, CODE HAS NOT BEEN MODIFIED YET BEYOND ONE CALC() FUNCTION CALL

# For each indice of the beam segment in parameter, no proper error handling yet for invalid values, repeating indices with same objective, have to test...

#  Currently can only optimize one x variable for each segment indice, should we implement more than one variable in the future?

#   Do we begin plotting iterations at 0 or 1???

class beamOptimizer():
    def __init__(self, beamline, matrixVariables):
        '''
        Constructor for beamline and particle values to optimize over for given y objectives and x variables

        Parameters
        ----------
        beamline: list[beamline]
            list of beamline objects representing accelerator beam
        matrixVariables: np.array(list[float][float])
            2D numPy list of particle elements
        '''
        ebeam = beam()
        #  Methods included in class to return staistical information, add to dictionary if additional methods are created
        self.OBJECTIVEMETHODS = {"std": ebeam.std,"epsilon":ebeam.epsilon,"alpha":ebeam.alpha,
                                 "beta":ebeam.beta,"gamma":ebeam.gamma,"phi":ebeam.phi,
                                 "envelope": ebeam.envelope, "dispersion": ebeam.disper}
        
        self.matrixVariables = matrixVariables
        self.beamline = beamline
    
    def _optiSpeed(self, variableVals):
        '''
        Simulates particle movement through a beamline, calculates y objective accuracy statistic for 
        particles after simulating with new x variables, and returns how accurate algorithm got by 
        using mean squared error stat

        Parameters
        ----------
        variableVals: list[float]
            test x value(s) to optimize, indice of each value corresponds to indice of variable in 
            variablesToOptimize

        Returns
        -------
        difference: float
            mean squared error statistic of how accurate all test objective values are to their goal
        '''
        particles = self.matrixVariables
        segments = self.beamline
        mse = []
        numGoals = 0
        
        #  Loop through beamline indices
        for i in range(len(segments)):
            #  Check if indice is in segmentVar (x variable dictionary)
            if i in self.segmentVar:
                try:
                    #  Adjust the segment's x variable value according to its mathematical relationship
                    yFunc = self.segmentVar.get(i)[2]
                    varIndex = self.variablesToOptimize.index(self.segmentVar.get(i)[0]) #  Get the index of the x variable to use with 
                    newValue = yFunc(variableVals[varIndex])
                    param = self.segmentVar.get(i)[1]
                    particles = np.array(segments[i].useMatrice(particles, **{param: newValue})) # Apply matrice transformation with changed segment attribute value
                except TypeError as e:
                    raise ValueError(f"segment {i} has no parameter {param}")
            else:
                particles = np.array(segments[i].useMatrice(particles))  #  apply matrice transformation with static segment values
            #  Check if indice in objective dictionary
            if i in self.objectives:
                for goalDict in self.objectives[i]:
                    # add sum piece to calculate statistical accuracy with MSE
                    stat = (goalDict["measure"][1](particles, goalDict["measure"][0]))
                    goalDict["measured"] = stat
                    mse.append(((stat-goalDict["goal"])**2)*goalDict["weight"])
                    numGoals = numGoals+1
                    stringForm = "indice " + str(i) + ": " + goalDict["measure"][0] + " " + goalDict["measure"][1].__name__
                    self.trackGoals[stringForm].append(stat) #  for plotting in calc()

        #  Calculate MSE
        difference = (np.sum(mse))/numGoals

        #  For plotting purposes in calc()
        self.trackVariables.append(variableVals)
        self.plotMSE.append(difference)
        self.plotIterate.append((self.iterationTrack) + 1)
        self.iterationTrack = self.iterationTrack + 1

        return difference
    
    def calc(self, method, segmentVar, startPoint, objectives, plotProgress = False, plotBeam = False, printResults = False):
        '''
        optimizes beamline segment attribute values so y values are close to objective values as possible.
        Post optimization plotting supported

        Parameters
        ----------
        method: str
            name of minimization method/algorithm
        segmentVar: dict
            dictionary, each key is an indice corresponding to its value of a list of 
            x variable parameters 
        objectives: dict
            dictionary, each key is an indice corresponding to its value of a list of
            y objectives. In each list are dictionaries corresponding to the parameters of that 
            y objective 
        startPoint: dict
            dictionary, each key is an x variable corresponding to another dictionary of
            that variables' bounds, starting search point, and other parameters
        plotProgress: bool
            plot x variable and y objective values as a function of iterations
        plotBeam: bool
            plot beamline simulation with new x variables post-optimization
        printResults: bool
            output data in terminal

        Returns
        -------
        result: OptimizeResult
            Object containing resulting information about optimization process and results
        '''
        #  Variables for plotting purposes later
        self.plotMSE = []
        self.plotIterate = []
        self.trackVariables = []
        self.iterationTrack = 0
        self.trackGoals = {}

        #  Initialize set-list of x variables to optimize
        self.segmentVar = segmentVar
        checkSet = set()
        self.variablesToOptimize = []
        for indice in self.segmentVar:
            if (indice < 0 or indice >= len(self.beamline)):
                raise IndexError(str(indice) + " is out of bounds for segmentVar dictionary")
            checkSet.add(self.segmentVar.get(indice)[0])
        #  This entire program relies on checking the indice of the variables in this list-set.
        #  A little sketch, will work for now but better implementation is needed
        self.variablesToOptimize = list(checkSet)

        #  Initialize objectives dictionary with measurement methods
        self.objectives = objectives
        for key, value in self.objectives.items():
            if (key not in range(len(self.beamline))):
                raise TypeError("Invalid indice: indice " + str(key) + " in objectives dict is out of bounds" )
            for goal in value:
                if goal["measure"][1] in self.OBJECTIVEMETHODS:
                    goal["measure"][1] = self.OBJECTIVEMETHODS[goal["measure"][1]]
                elif isinstance(goal["measure"][1], str):
                    raise TypeError("Invalid method name: No such method name exists in OBJECTIVESMETHOD dict")
                #  Used to keep track of data plotting through optimization
                #  Very rudementary, since looking for plotting of an objective relies on finding the same string name. Will have to improve in future
                self.trackGoals.update({"indice " + str(key) + ": " + goal["measure"][0] + " "  + goal["measure"][1].__name__: []})

        #  Create x variables' bounds and  start point list. 
        #  Order corresponding to x variable order in variablesToOptimize
        self.variablesValues = [] 
        self.bounds = []
        for i in self.variablesToOptimize:
            self.variablesValues.append(1) 
            self.bounds.append((None, None))
        for var in startPoint:
            index = self.variablesToOptimize.index(var)
            if "start" in startPoint.get(var): self.variablesValues[index] = startPoint.get(var).get("start")
            if "bounds" in startPoint.get(var): self.bounds[index] = startPoint.get(var).get("bounds")

        # Time speed to minimize difference of objective function
        startTime = time.perf_counter()  
        result = spo.minimize(self._optiSpeed, self.variablesValues, method=method, bounds=self.bounds)
        endTime = time.perf_counter()

        # print out new values for each beam segment's attribute
        output = "\nx variables:"
        for indice in self.segmentVar:
                variable = self.segmentVar.get(indice)[0]
                index = self.variablesToOptimize.index(variable)
                yFunc = self.segmentVar.get(indice)[2]
                newVal = yFunc(result.x[index])
                segAttr = self.segmentVar.get(indice)[1]
                setattr(self.beamline[indice], segAttr, newVal)
                if printResults:
                    output += "\nindice " + str(indice) + " new " + segAttr + " value: " + str(newVal)
        if printResults:
            output += "\n\ny objectives:\n"
            for indice, value in self.objectives.items():
                for obj in value:
                    output += "indice " + str(indice) + ": " + obj["measure"][0] + " "  + obj["measure"][1].__name__ + " value of " + str(obj["measured"]) + "\n"
            output += "Final difference: " + str(result.fun) + "\n"
            output += "\nTotal time: " + str(endTime-startTime) + " s\n"
            output +="Total iterations: " + str(self.iterationTrack) + "\n"
            print(output)

        # Plot the progress of y objectives and x variables as a function of iterations
        if plotProgress:
            fig, ax = plt.subplots(2,1)
            handles = []

            # plot MSE line
            mseLine, =ax[1].plot(self.plotIterate, self.plotMSE, label = 'Mean Squared Error', color = 'black')
            ax[1].set_xlabel('Iterations')
            ax[1].set_yscale('log')
            ax[1].set_ylabel('Mean Squared Error')
            ax[1].set_title("MSE and objectives vs Iterations")
            ax[1].tick_params(axis='y')
            handles.append(mseLine)

            # Plot y goals
            ax2 = ax[1].twinx()
            mini = 0
            for i, key in enumerate(self.trackGoals):
                valLine, = ax2.plot(self.plotIterate, self.trackGoals[key], label = key)
                handles.append(valLine)
                tempMin = abs(min(self.trackGoals[key]))
                if i == 0 or mini>tempMin:
                    mini = tempMin
            ax2.set_yscale('symlog', linthresh=10**(np.ceil(np.log10(mini))))
            ax2.set_ylabel('Objective functions')
            ax2.legend(handles = handles, loc = 'upper right')

            # Plot x variables + sec/iteration
            tempTrackVari = np.array(self.trackVariables)
            handles = []
            for i in range(len(tempTrackVari[0])):
                varLine, = ax[0].plot(self.plotIterate, tempTrackVari[:,i], label = self.variablesToOptimize[i])
                handles.append(varLine)
            ax[0].set_xlabel('Iterations')
            ax[0].set_ylabel('Variable values')
            ax[0].set_title('Variable Values through each Iteration')
            timeLine = mlines.Line2D([], [], color = 'white', label=f'{round((endTime-startTime)/self.iterationTrack,4)} s/iteration')
            handles.append(timeLine)
            ax[0].legend(handles = handles, loc = 'upper right')

           
            plt.tight_layout()
            plt.show()

        # Plot beam simulation with new values
        if plotBeam:
            schem = draw_beamline()
            tempPart = self.matrixVariables
            schem.plotBeamPositionTransform(tempPart, self.beamline)

        return result
   
    def testSpeed(self, iterations):
        '''
        Test the speed of an optimization algorithm (more iterations = more accurate)

        Parameters
        ----------
        iterations: float
            Number of times to run calc() function with 

        Returns
        -------
        timeResult: float
            average number of seconds to execute calc() function once
        '''
        timeResult = (timeit.timeit(self.calc,number = iterations))/iterations
        return timeResult
    
    def testFuncEval(self, iterations):
        '''
        Test the number of function evalutions an optimization algorithm performs (more iterations = more accurate)

        Parameters
        ----------
        iterations: float
            Number of times to run calc() function with 

        Returns
        -------
        timeResult: float
            average number of function evalutions per calc() call
        evalType: str
            type of evaluation that is being returned
        '''
        iterationsTotal = 0
        evalType = ''
        for i in range(iterations):
            result = self.calc()
            try: 
                iterationsTotal += result.nfev
                evalType = "# of function evaluations"
            except AttributeError: 
                try: 
                    iterationsTotal += result.njev
                    evalType = "# of Jacobian evaluations"
                except AttributeError: 
                    iterationsTotal += result.nhev
                    evalType = "# of Hessian evaluations"
        return iterationsTotal/iterations, evalType   
        
    def testFuncIt(self, iterations):
        '''
        Test the number of iterations of an optimization algorithm (more iterations = more accurate)

        Parameters
        ----------
        iterations: float
            Number of times to run calc() function with 
        
        Returns
        -------
        avIterate: float
            average number of function iterations per calc() call
        '''
        iterationsTotal = 0
        for i in range(iterations):
            result = self.calc()
            try: iterationsTotal += result.nit
            except AttributeError: 
                raise AttributeError("algorithm does not track iterations")
        avIterate = iterationsTotal/iterations   
        return avIterate
    
