from beamline import *
from schematic import draw_beamline
from ebeam import *
import sympy as sp
import sys
from tqdm import tqdm
import warnings

'''
helpful resources
https://stackoverflow.com/questions/38104025/sympy-solveset-returns-conditionset-when-it-should-be-returning-a-numerical
'''

class AlgebraicOpti():
    def __init__(self):
        self.DDOF = 1
        self.SEARCHINTERVAL = 1
        self.UNIVARIATE_SEARCH_RANGE = (0.00001,10)
        self.BIVARIATE_SEARCH_RANGE = 10
        self.BIVARIATE_SEARCH_POINT = (5,5)
        

    def getDistSigmai(self, particles):
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        sigmaI = [[0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0]]
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = twiss.loc[axis]
            epsilon = ax["$\epsilon$ ($\pi$.mm.mrad)"]
            alphaEpsilon = ax[r"$\alpha$"]*epsilon
            beta = ax[r"$\beta$ (m)"]
            gamma = ax[r"$\gamma$ (rad/m)"]
    
            sigmaI[i*2][i*2] = beta*epsilon
            sigmaI[i*2][i*2+1] = -(alphaEpsilon)
            sigmaI[i*2+1][i*2] = -(alphaEpsilon)
            sigmaI[i*2+1][i*2+1] = gamma*epsilon
        return sp.Matrix(sigmaI)
    
    #[epsilon, alpha, beta, gamma] for each axis
    def getTwissSigmai(self, xTwiss, yTwiss, zTwiss):
        twiss = [xTwiss,yTwiss,zTwiss]
        sigmaI = [[0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0]]
        for i in range(3):
             ax = twiss[i]
             epsilon = ax[0]
             alphaEpsilon = -(ax[1]*epsilon)
             betaEpsilon = ax[2]*epsilon
             gammaEpsilon = ax[3]*epsilon
             sigmaI[i*2][i*2] = betaEpsilon
             sigmaI[i*2][i*2+1] = -(alphaEpsilon)
             sigmaI[i*2+1][i*2] = -(alphaEpsilon)
             sigmaI[i*2+1][i*2+1] = gammaEpsilon
        return sp.Matrix(sigmaI)
 
    def getM(self, beamline: list, xVar: dict):
        resultArr = None
        try:
             resultArr = beamline[-1].getSymbolicMatrice(**xVar[len(beamline) - 1])
        except KeyError:
            resultArr = beamline[-1].getSymbolicMatrice()
        i = len(beamline) - 2
        while (i >= 0):
                try:
                    resultArr = resultArr*beamline[i].getSymbolicMatrice(**xVar[i])
                except KeyError:
                    resultArr = resultArr*beamline[i].getSymbolicMatrice()
                i -= 1
        return resultArr
    
    def getSigmaF(self, transferM, sigmaI):
        '''
        returns linear equations representing final beam matrix
        '''  
        mTransposed = transferM.T
        return transferM*sigmaI*mTransposed
    

  
    #  assumed that particles/ twiss are beginning beamline parameters

    # is it possible to have multiple parameter variables for a single beam element? 

    # if values plotted, should we return them or print them?
    def findSymmetricObjective(self, beamline, xVar, startParticles = None, twiss = None, plotBeam = None, latex = False,
                               plotEquation = False):
        '''
        returns the final sigma beam matrix from beamline and twiss/particle parameters. 
        final sigma matrix made as to return a symmetric beamline regarding each twiss parameter

        Parameters
        ----------
        beamline: list[beamline]
            list of beamline objects representing accelerator beam
        xVar: dict
            dictionary of beamline element indices and their parameter values to optimize
        startParticles = np.array(list[float][float])
            2D numPy list of particle elements
        twiss: dict
            twiss values for each dimensional plane
        plotBeam: list
            indice address of equation in final sigma matrix to plot
        latex: bool
            whether to return equations in Latex form

        Returns
        -------
        latexList: list[list[str]]
            6x6 2D list of equations in latesx form represented as strings
        sigObg: sympy.Matrix
            6x6 2D Symypy matrix of equations representing transformed beam matrix
        '''
        # Initialize particles/twiss parameters for finding sigma f matrix
        sigi = None
        eq = None
        if not startParticles is None:
              sigi = self.getDistSigmai(startParticles)
        elif not twiss is None:
            objList = []
            for ind, axis in enumerate(['x','y','z']):
                objList.append(twiss[axis])
            sigi = self.getTwissSigmai(objList[0],objList[1],objList[2])
        else:
             raise ValueError(
                 "Please enter twiss or startParticles parameter")
        if (not twiss is None) and (not startParticles is None):
             raise ValueError(
                 "Please enter one parameter for either twiss or startParticles only")
        
        # Create sigmaf matrice
        mMat = self.getM(beamline, xVar)
        sigObg = self.getSigmaF(mMat,sigi)
        for i in range(len(sigObg)):
             sigObg[i] -= sigi[i]
        
        #  REMOVE FLAG AND CLEAN CODE UP
        #  Plotting found optimized values
        if plotBeam and twiss is not None:
            raise ValueError(
                 "plotBeam cannot be used with twiss parameter")
        if plotBeam is not None:
            flag = True
            eq = sigObg[plotBeam[0], plotBeam[1]]
            numVar = eq.free_symbols
            roots = None
            variables = None
            if len(numVar) == 1: 
                roots, variables = self.getRootsUni(eq)
            elif len(numVar) == 2:
                roots, variables = self.getRootsMulti(eq)
            elif len(numVar) > 2:
                raise ValueError("Too many variables in equation")
            else:
                warnings.warn("No variables detected in equation at specified sigma final matrix position, plotting skipped")
                flag = False
            if flag:
                try:
                    #  The root pair used is the first one in the list if a equation has only one solution,
                    #  might make this more noticable to change in the future if user wants to use not only first root
                    test = roots[0]
                except IndexError:
                    warnings.warn(
                        "No root found, plotting skipped")
                    flag = False
                if flag:
                    for i, usedRoot in enumerate(roots):
                        print(f"root used for plotting: {usedRoot}")  # For testing purposes, might return this in the future
                        for i in xVar:
                            for paramName, key in xVar[i].items():
                                variableIndex = variables.index(key)
                                if (len(numVar) == 1):
                                    setattr(beamline[i], paramName, float(usedRoot))
                                else:    
                                    setattr(beamline[i], paramName, float(usedRoot[variableIndex]))
                        schem = draw_beamline()
                        schem.plotBeamPositionTransform(startParticles,beamline)
                            
                            
        if plotEquation and len(numVar) == 1:
            sp.plot(eq) 
        elif plotEquation and len(numVar) == 2:
            sp.plot_implicit(eq)
        
        # Return sigma f matrix in LaTex format
        if latex:
            latexList = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
            for i in range(len(latexList)):
                for ii in range(len(latexList[i])):
                    latexList[i][ii] = sp.latex(sigObg[i,ii])
            return latexList
        
        return sigObg  # return the objective functions, solutions at zeros
         
    def getRootsUni(self, equation):
        '''
        returns the roots of an equation using the multi-start method,
        function to be used when finding segment parameter values for a sigmaf equation. 
        Function intended for univariate equations
        
        Parameters
        ----------
        equation: sympy.add
            sympy equation to find roots at

        Returns
        -------
        rootList: list[float]
            estimated list of zeros of equation in specified interval
        nameList: list[str]
            list of variable name
        '''
        rootSet = set()
        ind = self.UNIVARIATE_SEARCH_RANGE[0]
        tempSet = equation.free_symbols
        if not len(tempSet) == 1:
            raise ValueError("Wrong amount of variables, please use univariate equation")
        var = tempSet.pop()
        total_intervals = self.UNIVARIATE_SEARCH_RANGE[1]
        with tqdm(total=total_intervals, desc="Finding roots...", 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar: #  loading bar
            while ind <= self.UNIVARIATE_SEARCH_RANGE[1]:
                try:
                    val = sp.nsolve(equation, var ,ind)
                    if val > 0:
                        rootSet.add(val)
                except ValueError: #  if root not found, ignore error
                    pass
                pbar.update(self.SEARCHINTERVAL)
                ind += self.SEARCHINTERVAL
        rootList = list(rootSet)
        nameList = [var.name]
        return rootList, nameList
    

    # problems with x = int and y = int: the starting point is at 5 for both of these. If a negative value
    # is detected and thrown out, despite maybe another postive solution existing, could be discarded

    # could implement looking at more starting points in the future than just at 5,5
    def getRootsMulti(self, equation):
        '''
        returns the roots of an implicit equation.
        Function to be used when finding segment parameter values for a sigmaf equation. 
        Function intended for bivariate implicit equations
        
        Parameters
        ----------
        equation: sympy.add
            sympy equation to find roots at

        Returns
        -------
        rootList: list[tuple]
            estimated list of zero pairs of equation in specified interval
        nameList: list[str]
            ordered list of variable names matching with the order of root pairs.
        '''
        ans = None
        rootSet = set()
        sett = equation.free_symbols
        if not len(sett) == 2:
            raise ValueError("Wrong amount of variables, please use bivariate equation")
        
        
        #  This is necessary so that the solutions given in rootList and variable order
        #  in nameList are the matched accordingly
        x = sett.pop()
        y = sett.pop()
        variableOrder = (x,y)
        nameList = []
        for i in variableOrder:
            nameList.append(i.name)

        # checkLine = sp.Eq(y-x, 0)
        # ind = self.UNIVARIATE_SEARCH_RANGE[0]
        # total_intervals = self.UNIVARIATE_SEARCH_RANGE[1]
        # with tqdm(total=total_intervals, desc="Finding roots...",
        #           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        #     while ind <= total_intervals:
        #         try:
        #             ans = sp.nsolve((equation, checkLine), (x, y), (ind, ind))
        #             if (ans[0] > 0 and ans[1] > 0):
        #                 tup = tuple(tuple(row) for row in ans.tolist())
        #                 rootSet.add(tup)
        #         except ValueError:
        #             pass
        #         pbar.update(self.ROOTINTERVAL)
        #         ind = ind + self.ROOTINTERVAL

        searchRange = self.BIVARIATE_SEARCH_RANGE
        with tqdm(total=searchRange*2, desc="Finding roots...",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar: #  loading bar
               
            while (searchRange > 0):
                try:
                    checkLine = sp.Eq(y, searchRange)
                    ans = sp.nsolve((equation, checkLine), variableOrder, self.BIVARIATE_SEARCH_POINT)
                    if (ans[0] > 0 and ans[1] > 0):
                            tup = tuple(row[0] for row in ans.tolist())
                            rootSet.add(tup)
                except ValueError: #  if root not found, ignore error
                    pass
                searchRange -= self.SEARCHINTERVAL
                pbar.update(self.SEARCHINTERVAL)

            searchRange = self.BIVARIATE_SEARCH_RANGE
            while (searchRange > 0):
                try:
                    checkLine = sp.Eq(x, searchRange)
                    ans = sp.nsolve((equation, checkLine), variableOrder, self.BIVARIATE_SEARCH_POINT)
                    if (ans[0] > 0 and ans[1] > 0):
                            tup = tuple(row[0] for row in ans.tolist())
                            rootSet.add(tup)
                except ValueError: #  if root not found, ignore error
                    pass
                searchRange -= self.SEARCHINTERVAL
                pbar.update(self.SEARCHINTERVAL)

        rootList = list(rootSet)
        return rootList, nameList
