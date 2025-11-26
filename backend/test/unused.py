'''
old code for circulating list in schematic
'''
# circularList = cycle(lineTwissData)
#             circularDataNames = cycle(twissDataNames)
#             def nextL(event):
#                 data = (next(circularList))
#                 for linei in range(len(lineList)): 
#                     line = lineList[linei]
#                     line.set_ydata(data[linei])
#                 ax6.relim()
#                 ax6.autoscale_view()
#                 ax6.set_ylabel(next(circularDataNames))  #  WORK AND IMPROVE THIS
#                 plt.draw()

'''
old main testing code for findSymmetricObjective 
'''
# eq = finm[3,2]
# # print(eq)
# sett = eq.free_symbols
# I = sett.pop()
# I2 = sett.pop()
# # p = plot.plot3d(eq, (I, -3, 3), (I2, -3, 3), zlim = (-10,10))
# newEq = sp.Eq(eq, 0)
# # p1 = plot.plot_implicit(newEq, (I, -10, 10), (I2, -10, 10))
# # print(alg.getRootsUni(eq))
# solutions, var = alg.getRootsMulti(eq)
# print(var)
# print(solutions)
# schem.plotBeamPositionTransform(beam_dist,line_E)
# line_E[1].current = solutions[0][0]
# line_E[3].current = solutions[0][0]
# line_E[5].current = solutions[0][0]
# line_E[6].current = solutions[0][0]
# schem.plotBeamPositionTransform(beam_dist,line_E, spacing = 5, defineLim=False)
# newEq = sp.Eq(eq, 0)
# p1 = plot.plot_implicit(newEq, (I, -10, 10), (I2, -10, 10),show = False)
# # p2 = plot.plot_implicit(testEq,(I, -10, 10), (I2, -10, 10),show = False)
# # p1.append(p2[0])
# p1.show()

''' old ebeam func'''
# '''
# plots 6d and twiss data with only particle distribution data
# '''
# import matplotlib.pyplot as plt
# import sympy as sp
# import numpy as np
# def plot_6d(self, dist_6d, title):
#     #
#     # Not used anymore and remove?
#     #

#     num_pts = 60  # Used for implicit plot of the ellipse
#     ddof = 1  # Unbiased Bessel correction for standard deviation calculation

#     dist_avg, dist_cov, twiss = self.cal_twiss(dist_6d, ddof=ddof)

#     # Define SymPy symbols for plotting
#     x_sym, y_sym = sp.symbols('x y')

#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     x_labels = [r'Position $x$ (mm)', r'Position $y$ (mm)', r'Energy $\Delta$ $E$ (keV)', r'Position $x$ (mm)']
#     y_labels = [r'Phase $x^{\prime}$ (mm)', r'Phase $y^{\prime}$ (mm)', r'Time $\Delta$ $t$ (ns)',
#                 r'Position $y$ (mm)']

#     for i, axis in enumerate(['x', 'y', 'z']):

#         # Access Twiss parameters for the current axis
#         twiss_axis = twiss.loc[axis]

#         # Plot the contour where Z = 0 (the ellipse)
#         ax = axes[i // 2, i % 2]
#         ax.scatter(dist_6d[:, 2 * i], dist_6d[:, 2 * i + 1], s=15, alpha=0.7)
#         X, Y, Z = self.ellipse_sym(dist_avg[2 * i], dist_avg[2 * i + 1], twiss_axis, n=1, num_pts=num_pts)
#         ax.contour(X, Y, Z, levels=[0], colors='black', linestyles='--')
#         X, Y, Z = self.ellipse_sym(dist_avg[2 * i], dist_avg[2 * i + 1], twiss_axis, n=6, num_pts=num_pts)
#         ax.contour(X, Y, Z, levels=[0], colors='black', linestyles='--')

#         # Construct the text string from the Twiss parameters
#         twiss_txt = '\n'.join(f'{label}: {np.round(value, 2)}' for label, value in twiss_axis.items())
#         props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)
#         ax.text(0.05, 0.95, twiss_txt, transform=ax.transAxes, fontsize=12,
#                 verticalalignment='top', bbox=props)

#         ax.set_title(f'{axis} - Phase Space')
#         ax.set_xlabel(x_labels[i])
#         ax.set_ylabel(y_labels[i])
#         ax.grid(True)

#     # Plot for 'x, y - Space'
#     ax = axes[(i + 1) // 2, (i + 1) % 2]
#     ax.scatter(dist_6d[:, 0], dist_6d[:, 2], s=15, alpha=0.7)

#     ax.set_title(f'x, y - Space')
#     ax.set_xlabel(x_labels[i + 1])
#     ax.set_ylabel(y_labels[i + 1])
#     ax.grid(True)

#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()

'''
old schematic function
'''
# def appendToList(self, xStd, yStd, xMean, yMean, x_axis, interval, matrixVariables):

#         # Not used anymore
#         # To be removed
#         # Due to the use of pandas frame from the 6d distribution, we do no need to recalculate the standard devs
#         '''
#         Append updated values to five different arrays, used for plotBeamPositionTransform

#         Parameters
#         ----------
#         xStd: list[float]
#             standard deviation of particles' x position for each distance interval
#         yStd: list[float]
#             standard deviation of particles' y position for each distance interval
#         xMean: list[float]
#             average of particles' x position for each distance interval
#         yMean: list[float]
#             average of particles' y position for each distance interval
#         x_axis: list[float]
#             contains distance intervals to measure particle data over
#         interval: float
#             the amount between each distance interval
#         matrixVariables: np.array(list[float][float])
#             2D numPy list of particle elements to measure data from

#         Returns
#         -------
#         xStd: list[float]
#             updated standard deviation of x position list
#         yStd: list[float]
#             updated standard deviation of y position list
#         xMean: list[float]
#             updated average of x position list
#         yMean: list[float]
#             updated average of y positiion list
#         x[axis]: list[float]
#             updated distance interval list
#         '''
#         xStd.append(np.std(matrixVariables[:,0]))
#         yStd.append(np.std(matrixVariables[:,2]))
#         xMean.append(np.mean(matrixVariables[:,0]))
#         yMean.append(np.mean(matrixVariables[:,2]))
#         x_axis.append(round(x_axis[-1]+interval, self.DEFAULTINTERVALROUND))
#         return xStd, yStd, xMean, yMean, x_axis


'''Old x variable initialization for beamOptimizer calc(), keep temporarily in case of bugs with new one. delete in future'''
        #
        # for item in self.segmentVar:
        #     if (item < 0 or item >= len(self.beamline)):
        #         raise IndexError
        #     varItem = self.segmentVar.get(item)[0]
        #     if varItem not in checkSet:
        #         self.variablesToOptimize.append(varItem)
        #         checkSet.add(varItem)

# Old lattice function
def getSymbolicMatrice(self):
        raise NameError("getSymbolicMatrice not defined in child class")


# Old Lattice matrice func
def useMatrice(self, values, matrice = None):
        ''''
        Simulates the movement of given particles through its child segment

        Parameters
        ----------
        values: np.array(list[float][float])
            2D numPy list of particle elements from which to simulate data
        matrice: np.array(list[float][float])
            6x6 matrice of child segment to simulate particle data with
        '''
        if (matrice is None): 
            raise NameError("useMatrice not defined in child class")
        newMatrix = []
        for array in values:
            tempArray = np.matmul(matrice, array)
            newMatrix.append(tempArray.tolist())
        return newMatrix #  return 2d list

'''old drift lattice'''
# def getSymbolicMatrice(self, length = None):
#         if length is None: l = self.length
#         else: l = symbols(length, real = True)
#         M56 = (l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         mat = Matrix([[1, l, 0, 0, 0, 0],
#                       [0, 1, 0, 0, 0, 0],
#                       [0, 0, 1, l, 0, 0],
#                       [0, 0, 0, 1, 0, 0],
#                       [0, 0, 0, 0, 1, M56],
#                       [0, 0, 0, 0, 0, 1]])
#         return mat
    
# #Matrix multiplecation, values is a 2 dimensional numPy array, each array is 6 elements long
# #values = np.array([[x, x', y, y', z, z'],...])
# #Note: 1x6 array is multiplied correctly with 6x6 array
# def useMatrice(self, values, length = -1):
#     if length <= 0:
#         length = self.length
#     M56 = (length * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#     return super().useMatrice(values,(np.array([[1, length, 0, 0, 0, 0],
#                                                 [0, 1, 0, 0, 0, 0],
#                                                 [0, 0, 1, length, 0, 0],
#                                                 [0, 0, 0, 1, 0, 0],
#                                                 [0, 0, 0, 0, 1, M56],
#                                                 [0, 0, 0, 0, 0, 1]])))

'''old qpf class functions'''
# def getSymbolicMatrice(self, length = None, current = None):
#         if current is None: I = self.current
#         else: I = symbols(current, real = True)
#         if length is None: l = self.length
#         else: l = symbols(length, real = True)
        

#         self.k = sp.Abs((self.Q*self.G*I)/(self.M*self.C*self.beta*self.gamma))
#         self.theta = sp.sqrt(self.k)*l

#         M11 = sp.cos(self.theta)
#         M21 = (-(sp.sqrt(self.k)))*sp.sin(self.theta)
#         M22 = sp.cos(self.theta)
#         M33 = sp.cosh(self.theta)
#         M43 = sp.sqrt(self.k)*sp.sinh(self.theta)
#         M44 = sp.cosh(self.theta)
#         M56 = (l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         if I == 0:
#             M12 = l
#             M34 = l
#         else:
#             M34 = sp.sinh(self.theta)*(1/sp.sqrt(self.k))
#             M12 = sp.sin(self.theta)*(1/sp.sqrt(self.k))

#         mat =  Matrix([[M11, M12, 0, 0, 0, 0],
#                         [M21, M22, 0, 0, 0, 0],
#                         [0, 0, M33, M34, 0, 0],
#                         [0, 0, M43, M44, 0, 0],
#                         [0, 0, 0, 0, 1, M56],
#                         [0, 0, 0, 0, 0, 1]])
        
#         return mat
               
#     '''
#     performs a transformation to a 2d np array made of 1x6 variable matrices

#     values: np.array([list[int],...])
#     '''
#     def useMatrice(self, values, length = -1, current = -1):
#         if length <= 0:
#             length = self.length
#         if current < 0:
#             current = self.current

#         #   Necessary because code had problems working with numpy arrays
#         if isinstance(current, np.ndarray):
#             current = current[0]

#         self.k = np.abs((self.Q*self.G*current)/(self.M*self.C*self.beta*self.gamma))
#         self.theta = np.sqrt(self.k)*length

#         M11 = np.cos(self.theta)
#         M21 = (-(np.sqrt(self.k)))*np.sin(self.theta)
#         M22 = np.cos(self.theta)
#         M33 = np.cosh(self.theta)
#         M43 = np.sqrt(self.k)*np.sinh(self.theta)
#         M44 = np.cosh(self.theta)
#         M56 = (length * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         if current == 0:
#             M12 = length
#             M34 = length
#         else:
#             M34 = np.sinh(self.theta) * (1 / np.sqrt(self.k))
#             M12 = np.sin(self.theta) * (1 / np.sqrt(self.k))

#         return super().useMatrice(values, np.array([[M11, M12, 0, 0, 0, 0],
#                                                     [M21, M22, 0, 0, 0, 0],
#                                                     [0, 0, M33, M34, 0, 0],
#                                                     [0, 0, M43, M44, 0, 0],
#                                                     [0, 0, 0, 0, 1, M56],
#                                                     [0, 0, 0, 0, 0, 1]]))


''' old qpd class functions'''
# def getSymbolicMatrice(self, length = None, current = None):
#         if current is None: I = self.current
#         else: I = symbols(current, real = True)
#         if length is None: l = self.length
#         else: l = symbols(length, real = True)
        
#         self.k = sp.Abs((self.Q*self.G*I)/(self.M*self.C*self.beta*self.gamma))
#         self.theta = sp.sqrt(self.k)*l

#         M11 = sp.cosh(self.theta)
#         M21 = sp.sqrt(self.k)*sp.sinh(self.theta)
#         M22 = sp.cosh(self.theta)
#         M33 = sp.cos(self.theta)
#         M43 = (-(sp.sqrt(self.k)))*sp.sin(self.theta)
#         M44 = sp.cos(self.theta)
#         M56 = l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1))

#         if I == 0:
#             M12 = l
#             M34 = l
#         else:
#             M34 = sp.sin(self.theta)*(1/sp.sqrt(self.k))
#             M12 = sp.sinh(self.theta)*(1/sp.sqrt(self.k))

#         mat = Matrix([[M11, M12, 0, 0, 0, 0],
#                         [M21, M22, 0, 0, 0, 0],
#                         [0, 0, M33, M34, 0, 0],
#                         [0, 0, M43, M44, 0, 0],
#                         [0, 0, 0, 0, 1, M56],
#                         [0, 0, 0, 0, 0, 1]])
        
#         return mat

#     def useMatrice(self, values, length = -1, current = -1):
#         if length <= 0:
#             length = self.length
#         if current < 0:
#             current = self.current

#         #   Necessary because code had problems working with numpy arrays
#         if isinstance(current, np.ndarray):
#             current = current[0]

#         self.k = np.abs((self.Q*self.G*current)/(self.M*self.C*self.beta*self.gamma))
#         self.theta = np.sqrt(self.k)*length

#         M11 = np.cosh(self.theta)
#         M21 = np.sqrt(self.k)*np.sinh(self.theta)
#         M22 = np.cosh(self.theta)
#         M33 = np.cos(self.theta)
#         M43 = (-(np.sqrt(self.k)))*np.sin(self.theta)
#         M44 = np.cos(self.theta)
#         M56 = length * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1))
#         if current == 0:
#             M12 = length
#             M34 = length
#         else:
#             M34 = np.sin(self.theta) * (1 / np.sqrt(self.k))
#             M12 = np.sinh(self.theta) * (1 / np.sqrt(self.k))

#         return super().useMatrice(values, (np.array([[M11, M12, 0, 0, 0, 0],
#                                                     [M21, M22, 0, 0, 0, 0],
#                                                     [0, 0, M33, M34, 0, 0],
#                                                     [0, 0, M43, M44, 0, 0],
#                                                     [0, 0, 0, 0, 1, M56],
#                                                     [0, 0, 0, 0, 0, 1]])))

'''old dipole class functions'''
#  def getSymbolicMatrice(self, length = None, angle = None):
#         if length is None: L = self.length
#         else: L = symbols(length, real = True)
#         if angle is None: a = self.angle
#         else: a = symbols(angle, real = True)

#         by = (self.M*self.C*self.beta*self.gamma / self.Q) * (a * sp.pi / 180 / self.length_total)
#         rho = self.M*self.C*self.beta*self.gamma / (self.Q * by)
#         theta = L / rho
#         C = sp.cos(theta)
#         S = sp.sin(theta)

#         M16 = rho * (1 - C) * (self.gamma / (self.gamma + 1))
#         M26 = S * (self.gamma / (self.gamma + 1))
#         M51 = self.f * (-S / (self.beta * self.C))
#         M52 = self.f * (-rho * (1 - C) / (self.beta * self.C))
#         M56 = self.f * (-rho * (L / rho - S) / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         mat = Matrix([[C, rho * S, 0, 0, 0, M16],
#                       [-S / rho, C, 0, 0, 0, M26],
#                       [0, 0, 1, L, 0, 0],
#                       [0, 0, 0, 1, 0, 0],
#                       [M51, M52, 0, 0, 1, M56],
#                       [0, 0, 0, 0, 0, 1]])
        
#         return mat

    
#     '''
#     performs a transformation to a 2d np array made of 1x6 variable matrices

#     values: np.array([list[int],...])
#     '''
#     def useMatrice(self, values, length = -1, angle = -1):
#         if length <= 0:
#             length = self.length
#         if angle < 0:
#             angle = self.angle

#         #   TEMPORARY PURPOSES
#         if isinstance(angle, np.ndarray):
#             angle = angle[0]

#         # Rectangular dipole
#         By = (self.M*self.C*self.beta*self.gamma / self.Q) * (angle * np.pi / 180 / self.length_total)

#         rho = self.M*self.C*self.beta*self.gamma / (self.Q * By)

#         theta = length / rho
#         C = np.cos(theta)
#         S = np.sin(theta)
#         L = length

#         M16 = rho * (1 - C) * (self.gamma / (self.gamma + 1))
#         M26 = S * (self.gamma / (self.gamma + 1))
#         M51 = self.f * (-S / (self.beta * self.C))
#         M52 = self.f * (-rho * (1 - C) / (self.beta * self.C))
#         M56 = self.f * (-rho * (L / rho - S) / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         M = np.array([[C, rho * S, 0, 0, 0, M16],
#                       [-S / rho, C, 0, 0, 0, M26],
#                       [0, 0, 1, length, 0, 0],
#                       [0, 0, 0, 1, 0, 0],
#                       [M51, M52, 0, 0, 1, M56],
#                       [0, 0, 0, 0, 0, 1]])

#         return super().useMatrice(values, M)

'''old dipole wedge functions'''
# def getSymbolicMatrice(self, length = None, angle = None):
#         if length is None: L = self.length
#         else: L = symbols(length, real = True)
#         if angle is None: a = self.angle
#         else: a = symbols(angle, real = True)
#         dipole_angle = self.dipole_angle
#         dipole_length = self.dipole_length

#         # Hard edge model for the wedge magnets
#         By = (self.M*self.C*self.beta*self.gamma / self.Q) * (dipole_angle * sp.pi / 180 / dipole_length)
#         R = self.M*self.C*self.beta*self.gamma / (self.Q * By)
#         k = sp.Abs((self.Q * By / self.length) / (self.M * self.C * self.beta * self.gamma)) # Verify
#         eta = (a * sp.pi / 180) * L / self.length # Verify
#         E = (L * k) / ((sp.cos(eta)) ** 2)
#         T = sp.tan(eta)
#         M56 = self.f * (L / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         mat = Matrix([[1, 0, 0, 0, 0, 0],
#                       [-T / R, 1, 0, 0, 0, 0],
#                       [0, 0, 1, 0, 0, 0],
#                       [0, 0, -(T + E / 3) / R, 1, 0, 0],
#                       [0, 0, 0, 0, 1, M56],
#                       [0, 0, 0, 0, 0, 1]])
        
#         return mat

#     '''
#     performs a transformation to a 2d np array made of 1x6 variable matrices

#     values: np.array([list[int],...])
#     '''
#     def useMatrice(self, values, length = -1, angle = -1):
#         if length <= 0:
#             length = self.length
#         if angle < 0:
#             angle = self.angle
#         dipole_angle = self.dipole_angle
#         dipole_length = self.dipole_length

#         #   TEMPORARY PURPOSES
#         if isinstance(angle, np.ndarray):
#             angle = angle[0]

#         # Hard edge model for the wedge magnets
#         By = (self.M*self.C*self.beta*self.gamma / self.Q) * (dipole_angle * np.pi / 180 / dipole_length)
#         R = self.M*self.C*self.beta*self.gamma / (self.Q * By)
#         k = np.abs((self.Q * By / self.length) / (self.M * self.C * self.beta * self.gamma)) # Verify
#         eta = (angle * np.pi / 180) * length / self.length # Verify
#         E = (length * k) / ((np.cos(eta)) ** 2)
#         T = np.tan(eta)
#         M56 = self.f * (length / (self.C * self.beta * self.gamma * (self.gamma + 1)))

#         M = np.array([[1, 0, 0, 0, 0, 0],
#                       [-T / R, 1, 0, 0, 0, 0],
#                       [0, 0, 1, 0, 0, 0],
#                       [0, 0, -(T + E / 3) / R, 1, 0, 0],
#                       [0, 0, 0, 0, 1, M56],
#                       [0, 0, 0, 0, 0, 1]])

#         return super().useMatrice(values, M)