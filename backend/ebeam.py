#   Authors: Christian Komo, Niels Bidault
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap


#in plotDriftTransform, add legend and gausian distribution for x and y points

#Replace list variables that are unchanging with tuples, more efficient for calculations

#Add legend for graphs 

# In the future, for twiss parameters and other methods, instead of taking entire 2D particle array,
# only perform calculations on just one variable list for faster performance while making optispeed function
# handle list splicing based on variable given to reach goal for.


class beam:
    def __init__(self):
        """
        Initialize beam object with plotting and computation defaults.
        """
        self.DDOF = 1 #  Unbiased Bessel correction for standard deviation calculation for twiss functions

        #  Plotting parameter configurations
        self.scatter_size = 30
        self.scatter_alpha = 0.7

         # Define custom colormap for plots with white for lowest values
        plasma = plt.cm.get_cmap('plasma', 256) #'magma'. 'inferno', 'plasma', 'viridis' for uniform (append _r for reverse).
        new_colors = plasma(np.linspace(0, 1, 256))
        new_colors[0] = [1, 1, 1, 0]
        plasma_with_white = LinearSegmentedColormap.from_list('plasma_with_white', new_colors)
        self.default_cmap = plasma_with_white
        self.lost_cmap = 'binary'  # Color map for lost particles
        self.BINS = 20  # Histogram bin count

    def ellipse_sym(self, xc, yc, twiss_axis, n=1, num_pts=40):
        '''
        Calculates and returns meshgrid coordinates (X, Y) and the evaluated implicit
        ellipse equation values (Z) for plotting an n-sigma ellipse.

        Parameters
        ----------
        xc : float
            The x-coordinate of the ellipse center (mean position).
        yc : float
            The y-coordinate of the ellipse center (mean angle).
        twiss_axis : pd.Series
            A pandas Series containing the Twiss parameters for a specific axis,
            including 'epsilon', 'alpha', 'beta', and 'gamma'.
        n : int, optional
            The number of standard deviations (sigma) defining the size of the ellipse.
            Defaults to 1 for a 1-sigma ellipse.
        num_pts : int, optional
            The number of points to use for generating the meshgrid for the ellipse.
            Defaults to 40.

        Returns
        -------
        tuple
            A tuple containing:
                - X (np.ndarray): 2D array of x-coordinates for the meshgrid.
                - Y (np.ndarray): 2D array of y-coordinates for the meshgrid.
                - Z (np.ndarray): 2D array representing the implicit ellipse equation:
                                  $\gamma (x-x_c)^2 + 2\alpha (x-x_c)(y-y_c) + \beta (y-y_c)^2 - \epsilon_{n} = 0$,
                                  where $\epsilon_{n} = n \times \epsilon$.
        '''
        emittance = n * twiss_axis[r"$\epsilon$ ($\pi$.mm.mrad)"]
        alpha = twiss_axis[r"$\alpha$"]
        beta = twiss_axis[r"$\beta$ (m)"]
        gamma = twiss_axis[r"$\gamma$ (rad/m)"]

        # Ellipse bounds
        x_max = xc + np.sqrt(emittance / (gamma - alpha ** 2 / beta))
        x_min = xc - np.sqrt(emittance / (gamma - alpha ** 2 / beta))
        y_max = yc + np.sqrt(emittance / (beta - alpha ** 2 / gamma))
        y_min = yc - np.sqrt(emittance / (beta - alpha ** 2 / gamma))

        x_vals = np.linspace(x_min, x_max, num_pts)
        y_vals = np.linspace(y_min, y_max, num_pts)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Ellipse implicit equation Z
        Z = gamma * (X - xc)** 2 + 2 * alpha * (X - xc) * (Y - yc) + beta * (Y - yc) ** 2 - emittance
        return X, Y, Z

    def cal_twiss(self, dist_6d, ddof=1):
        '''
        Calculates the Twiss parameters (emittance, alpha, beta, gamma, dispersion,
        dispersion prime, and phase advance) for each of the three phase space axes (x, y, z)
        from a 6D particle distribution.

        Parameters
        ----------
        dist_6d : np.ndarray
            A 2D numpy array representing the 6D particle distribution.
            Expected columns are typically [x, x', y, y', z, z'].
        ddof : int, optional
            Degrees of freedom for Bessel's correction in covariance calculation. Defaults to 1.

        Returns
        -------
        tuple
            A tuple containing:
                - dist_avg (np.ndarray): 1D array of the mean values for each of the 6 coordinates.
                - dist_cov (np.ndarray): 6x6 covariance matrix of the particle distribution.
                - twiss (pd.DataFrame): A DataFrame containing the calculated Twiss parameters
                                        for 'x', 'y', and 'z' axes, with appropriate column labels.
        '''
        dist_avg = np.mean(dist_6d, axis=0)
        dist_cov = np.cov(dist_6d, rowvar=False, ddof=ddof)

        label_twiss = [r"$\epsilon$ ($\pi$.mm.mrad)", r"$\alpha$", r"$\beta$ (m)", r"$\gamma$ (rad/m)", r"$D$ (mm)",
                       r"$D^{\prime}$ (mrad)", r"$\phi$ (deg)"]

        label_axes = ["x", "y", "z"]

        twiss_data = []

        sigma_delta = dist_cov[5, 5]  # variance of (10^-3 δW/W)^2, dimensionless

        for i in range(3):
            idx, idx_prime = 2 * i, 2 * i + 1

            var = dist_cov[idx, idx]
            var_prime = dist_cov[idx_prime, idx_prime]
            covar = dist_cov[idx, idx_prime]

            if i < 2:
                 # Transverse planes with dispersion
                D = dist_cov[idx, 5] / sigma_delta
                D_prime = dist_cov[idx_prime, 5] / sigma_delta

                # Dispersion-corrected variances
                var_disp_free = var - D ** 2 * sigma_delta
                var_prime_disp_free = var_prime - D_prime ** 2 * sigma_delta
                covar_disp_free = covar - D * D_prime * sigma_delta

                # Intrinsic emittance
                epsilon = np.sqrt(var_disp_free * var_prime_disp_free - covar_disp_free ** 2)

                alpha = -covar_disp_free / epsilon
                beta = var_disp_free / epsilon
                gamma = var_prime_disp_free / epsilon

                dispersion = D
                dispersion_prime = D_prime
            else:
                # Longitudinal plane (no dispersion)
                epsilon = np.sqrt(var * var_prime - covar ** 2)
                alpha = -covar / epsilon
                beta = var / epsilon
                gamma = var_prime / epsilon
                dispersion = dispersion_prime = 0.0

            # Standardized phase advance φ calculation
            phi_rad = 0.5 * np.arctan2(2 * alpha, gamma - beta)
            phi_deg = np.rad2deg(phi_rad)

            twiss_data.append([epsilon, alpha, beta, gamma, dispersion, dispersion_prime, phi_deg])

        twiss = pd.DataFrame(twiss_data, columns=label_twiss, index=label_axes[:3])

        return dist_avg, dist_cov, twiss

    def gen_6d_gaussian(self, mean, std_dev, num_particles=100):
        '''
        Generates a 6D Gaussian distributed beam of particles.

        Parameters
        ----------
        mean : float
            Mean/center of the distribution
        std_dev : np.ndarray
            A 1D numpy array of length 6, representing the standard deviations for each of the
            six phase space coordinates.
        num_particles : int, optional
            The number of particles to generate. Defaults to 100.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (num_particles, 6) containing the generated
            6D Gaussian distributed particle data.
        '''
        particles = np.random.normal(mean, std_dev, size=(num_particles, 6))
        return particles
    









    '''
    THIS FUNCTION BELOW IS A WIP
    '''

    def is_within_ellipse(self, x, y, xc, yc, twiss_axis, n):
        '''
        Checks whether a given point $(x, y)$ lies within or on the boundary of
        the n-sigma ellipse defined by the provided Twiss parameters.

        Parameters
        ----------
        x : float
            The x-coordinate of the point to check.
        y : float
            The y-coordinate of the point to check.
        xc : float
            The x-coordinate of the ellipse center (mean position).
        yc : float
            The y-coordinate of the ellipse center (mean angle).
        twiss_axis : pd.Series
            A pandas Series containing the Twiss parameters for a specific axis,
            including 'epsilon', 'alpha', 'beta', and 'gamma'.
        n : int
            The number of standard deviations (sigma) defining the size of the ellipse.

        Returns
        -------
        bool
            True if the point $(x, y)$ is within or on the ellipse, False otherwise.
        '''
        emittance = n * twiss_axis[r"$\epsilon$ ($\pi$.mm.mrad)"]
        alpha = twiss_axis[r"$\alpha$"]
        beta = twiss_axis[r"$\beta$ (m)"]
        gamma = twiss_axis[r"$\gamma$ (rad/m)"]

        # Calculate the ellipse equation
        Z = gamma * (x - xc) ** 2 + 2 * alpha * (x - xc) * (y - yc) + beta * (y - yc) ** 2 - emittance

        # Check if the point (x, y) is inside or on the ellipse
        return Z <= 0

    '''
    THIS FUNCTION BELOW IS A WIP
    '''
    def particles_in_ellipse(self, dist_6d, n=1):
        '''
        Counts how many particles fall within n-sigma ellipses for each phase space axis
        (x-x', y-y', z-z') and optionally plots the distributions with the ellipses.

        THIS FUNCTION IS A WORK IN PROGRESS (WIP) AND MAY REQUIRE FURTHER DEVELOPMENT.

        Parameters
        ----------
        dist_6d : np.ndarray
            6D particle distribution data.
        n : int, optional
            The number of standard deviations (sigma) for the ellipse size. Defaults to 1.

        Returns
        -------
        list
            A list containing the count of particles within the specified n-sigma ellipse
            for each axis (x, y, z).
        '''
        fig, axes = plt.subplots(2, 2)

        dist_avg, dist_cov, twiss = self.cal_twiss(dist_6d, ddof=1)
        count_within_ellipse = []  # List to store count of particles within ellipse for each axis

        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[i // 2, i % 2]
            twiss_axis = twiss.loc[axis]
            X, Y, Z = self.ellipse_sym(dist_avg[2 * i], dist_avg[2 * i + 1], twiss_axis, n=n, num_pts=60)
            
            # Plot particle points and ellipse contours
            ax.scatter(dist_6d[:, 2 * i], dist_6d[:, 2 * i + 1], s=15, alpha=0.7)
            ax.contour(X, Y, Z, levels=[0], colors='black', linestyles='--')

            # Evaluate particles within ellipse
            num_within_ellipse = 0 
            for j in range(len(dist_6d)):
                x, y = dist_6d[j, 2 * i], dist_6d[j, 2 * i + 1] #  Retrives x and y from each value list
                # Check if the particle is within the ellipse
                if self.is_within_ellipse(x, y, dist_avg[2 * i], dist_avg[2 * i + 1], twiss_axis, n = n):
                    num_within_ellipse += 1

          
            twiss_txt = '\n'.join(f'{label}: {np.round(value, 2)}' for label, value in twiss_axis.items())
            props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)
            ax.text(0.05, 0.95, twiss_txt, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            count_within_ellipse.append(num_within_ellipse)
            # Optional: Display the count on the plot
            ax.set_title(f'{axis} - Phase Space\nParticles within ellipse: {num_within_ellipse}')
            ax.set_xlabel(f'Position {axis}')
            ax.set_ylabel(f'Phase {axis} prime')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        return count_within_ellipse
















    def findVarValues(self, particles, variable):
        '''
        Extracts the values of a specified variable from the particle data.

        Parameters
        ----------
        particles : np.ndarray
            A 2D numpy array where each row represents a particle
            and columns represent different phase space coordinates.
            Expected format: [x, x', y, y', z, z'].
        variable : str
            The name of the variable to extract (e.g., "x", "x'", "y", "y'", "z", "z'").

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the values of the specified variable for all particles.
        '''
        particleDict = {"x": particles[:, 0], "x'": particles[:, 1],
                        "y": particles[:, 2], "y'": particles[:, 3],
                        "z": particles[:, 4], "z'": particles[:, 5]}
        return particleDict[variable]

    def std(self, particles, variable):
        '''
        Calculates the standard deviation of a specified variable for the given particles.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The name of the variable to calculate standard deviation for.

        Returns
        -------
        float
            The standard deviation of the variable.
        '''
        particleData = self.findVarValues(particles, variable)
        return np.std(particleData)

    def alpha(self, particles, variable):
        '''
        Calculates the alpha Twiss parameter for a given variable.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The alpha Twiss parameter.
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        return twiss.loc[variable].loc[r"$\alpha$"]
    
    def epsilon(self, particles, variable):
        '''
        Calculates the emittance (epsilon) Twiss parameter for a given variable.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The emittance (epsilon) Twiss parameter in ($\pi$.mm.mrad).
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        return twiss.loc[variable].loc[r"$\epsilon$ ($\pi$.mm.mrad)"]
    
    def beta(self, particles, variable):
        '''
        Calculates the beta Twiss parameter for a given variable.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The beta Twiss parameter in (m).
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        return twiss.loc[variable].loc[r"$\beta$ (m)"]
    
    def gamma(self, particles, variable):
        '''
        Calculates the gamma Twiss parameter for a given variable.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The gamma Twiss parameter in (rad/m).
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        return twiss.loc[variable].loc[r"$\gamma$ (rad/m)"]
    
    def phi(self, particles, variable):
        '''
        Calculates the phi Twiss parameter (phase advance) for a given variable.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The phi Twiss parameter in (deg).
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        return twiss.loc[variable].loc[r"$\phi$ (deg)"]
    
    def envelope(self, particles, variable):
        '''
        Calculates the beam envelope for a given variable.

        The envelope is calculated as $(10^3) * \sqrt{emittance * beta}$.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The beam envelope.
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        emittance = (10 ** -6) * twiss.loc[variable].loc[r"$\epsilon$ ($\pi$.mm.mrad)"]
        beta = twiss.loc[variable].loc[r"$\beta$ (m)"]
        envelope = (10 ** 3) * np.sqrt(emittance * beta)
        return envelope
    
    def disper(self, particles, variable):
        '''
        Calculates the dispersion (D) Twiss parameter for a given variable.

        Parameters
        ----------
        particles : np.ndarray
            Particle data.
        variable : str
            The phase space plane (e.g., 'x', 'y', 'z').

        Returns
        -------
        float
            The dispersion (D) Twiss parameter in (mm).
        '''
        ebeam = beam()
        dist_avg, dist_cov, twiss = ebeam.cal_twiss(particles, ddof=self.DDOF)
        return twiss.loc[variable].loc[r"$D$ (mm)"]


        

    



    def getXYZ(self, dist_6d):
        '''
        Calculates Twiss parameters and generates points for 1-sigma and 6-sigma ellipses
        for 2D phase spaces (x-x', y-y', z-z').

        Parameters
        ----------
        dist_6d : np.ndarray
            6D particle distribution data.

        Returns
        -------
        tuple
            A tuple containing:
                - std1 (list): List of [X, Y, Z] coordinates for 1-sigma ellipses.
                - std6 (list): List of [X, Y, Z] coordinates for 6-sigma ellipses.
                - dist_6d (np.ndarray): The input 6D particle distribution data.
                - twiss (pd.DataFrame): DataFrame containing calculated Twiss parameters.
        '''
        num_pts = 60  # Used for implicit plot of the ellipse
        ddof = 1  # Unbiased Bessel correction for standard deviation calculation
        dist_avg, dist_cov, twiss = self.cal_twiss(dist_6d, ddof=ddof)
        std6 = []
        std1 = []
        for i, axis in enumerate(twiss.index):
            # Access Twiss parameters for the current axis
            twiss_axis = twiss.loc[axis]
            X, Y, Z = self.ellipse_sym(dist_avg[2 * i], dist_avg[2 * i + 1], twiss_axis, n=1, num_pts=num_pts)
            std1.append([X,Y,Z])
            X, Y, Z = self.ellipse_sym(dist_avg[2 * i], dist_avg[2 * i + 1], twiss_axis, n=6, num_pts=num_pts)
            std6.append([X,Y,Z])
        
        return std1, std6, dist_6d, twiss

    def plotXYZ(self, dist_6d, std1, std6, twiss, ax1, ax2, ax3, ax4, 
                maxVals = [0,0,0,0,0,0], minVals = [0,0,0,0,0,0], defineLim=True, shape={}, scatter=False):
        '''
        Plots 2D phase spaces (x-x', y-y', z-z') and the x-y spatial distribution
        with Twiss parameters and optional aperture shapes.

        Parameters
        ----------
        dist_6d : np.ndarray
            6D particle distribution data.
        std1 : list
            List of [X, Y, Z] for 1-sigma ellipses for each plane.
        std6 : list
            List of [X, Y, Z] for 6-sigma ellipses for each plane.
        twiss : pd.DataFrame
            DataFrame containing calculated Twiss parameters.
        ax1 : matplotlib.axes.Axes
            Axes object for x-x' phase space.
        ax2 : matplotlib.axes.Axes
            Axes object for y-y' phase space.
        ax3 : matplotlib.axes.Axes
            Axes object for z-z' phase space.
        ax4 : matplotlib.axes.Axes
            Axes object for x-y spatial distribution.
        maxVals : list, optional
            List of maximum values for setting plot limits [x_max, x'_max, y_max, ...].
        minVals : list, optional
            List of minimum values for setting plot limits [x_min, x'_min, y_min, ...].
        defineLim : bool, optional
            If True, set plot limits based on maxVals and minVals.
        shape : dict, optional
            Dictionary defining an aperture shape (e.g., {"shape": "circle", "radius": R, "origin": [x0, y0]}).
        scatter : bool, optional
            If True, use scatter plot; otherwise, use hexbin plot.
        '''
        axlist = [ax1,ax2,ax3]
        # ax1.set_aspect(aspect = 1, adjustable='datalim')

        #  Define SymPy symbols for plotting
        x_sym, y_sym = sp.symbols('x y')
        x_labels = [r'Position $x$ (mm)', r'Position $y$ (mm)', r'Relative Bunch ToF $\Delta t$ $/$ $T_{rf}$ $(10^{-3})$', r'Position $x$ (mm)']
        y_labels = [r'Phase $x^{\prime}$ (mrad)', r'Phase $y^{\prime}$ (mrad)', r'Relative Energy $\Delta W / W_0$ $(10^{-3})$',
                    r'Position $y$ (mm)']
        label_twiss_z = ["$\epsilon$ ($\pi$.$10^{-6}$)", r"$\alpha$", r"$\beta$", r"$\gamma$", r"$D$ (m)",
                       r"$D^{\prime}$ (mrad)", r"$\phi$ (deg)"]
        
        #  Plot x-x', y-y', z-z' phase spaces
        for i, axis in enumerate(twiss.index):
            twiss_axis = twiss.loc[axis]

            #  Access Twiss parameters for the current axis
            ax = axlist[i]
            if defineLim:
                ax.set_xlim(minVals[2*i], maxVals[2*i])
                ax.set_ylim(minVals[2*i + 1], maxVals[2*i + 1])
                # ax.set_aspect(aspect = 1, adjustable='datalim')  # Testing purpose
                # print(ax1.get_xlim())
                # print(ax1.get_ylim())
                # ax.set(xlim=(-maxVals[2*i], maxVals[2*i]), ylim=(-maxVals[2*i + 1], maxVals[2*i + 1]))

            #  Plot particle axes data with sigma 1, sigma 6 
            self.heatmap(ax, dist_6d[:, 2 * i], dist_6d[:, 2 * i + 1], scatter=scatter)
            ax.contour(std1[i][0], std1[i][1], std1[i][2], levels=[0], colors='black', linestyles='--')
            ax.contour(std6[i][0], std6[i][1], std6[i][2], levels=[0], colors='black', linestyles='--')

            #  Display Twiss parameters on the plot
            if i == 2: #  For the longitudinal (z) plane, use a different set of Twiss parameters
                j = 0
                twiss_txt = ''
                for label, value in twiss_axis.items():
                    if j == 0:
                        space = ''
                    else:
                        space = '\n'
                    twiss_txt = twiss_txt + space + f'{label_twiss_z[j]}: {np.round(value, 3)}'
                    j = j + 1
                    if j == 4: #  Only show first 4 Twiss parameters for z-plane based on label_twiss_z
                        break
            else: #  For x and y planes, show all but the last two Twiss parameters
                twiss_items_short = list(twiss_axis.items())[:-2]
                twiss_txt = '\n'.join(f'{label}: {np.round(value, 3)}' for label, value in twiss_items_short)

            props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)
            ax.text(0.01, 0.97, twiss_txt, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)
            ax.set_title(f'{axis} - Phase Space')
            ax.set_xlabel(x_labels[i])
            ax.set_ylabel(y_labels[i])
            ax.grid(True)

        #  Plot for 'x, y - Space'
        withinArea = [[],[]]
        outsideArea = [[],[]]
        xyPart = [np.array(dist_6d[:, 0]), np.array(dist_6d[:, 2])]

        #  Handle different aperture shapes
        if shape.get("shape") == "circle":
            radius = shape.get("radius")
            origin = shape.get("origin")
            for ii in range(len(xyPart[0])):
                x, y = xyPart[0][ii], xyPart[1][ii]
                if (x - origin[0])**2 + (y - origin[1])**2 < radius**2:
                    withinArea[0].append(x)
                    withinArea[1].append(y)
                else:
                    outsideArea[0].append(x)
                    outsideArea[1].append(y)
            circle = patches.Circle(origin, radius, edgecolor="black", facecolor="None", zorder = 3)
            ax4.add_patch(circle)
            totalExtent = [xyPart[0].min(), xyPart[0].max(), xyPart[1].min(),xyPart[1].max()]
            self.heatmap(ax4, withinArea[0], withinArea[1], scatter=scatter, zorder=2, shapeExtent = totalExtent)
            self.heatmap(ax4, outsideArea[0], outsideArea[1], scatter=scatter, lost=True, zorder=1, shapeExtent = totalExtent)
            percentageInside = len(withinArea[0]) / len(xyPart[0]) * 100
        elif shape.get("shape") == "rectangle":
            length = shape.get("length")
            width = shape.get("width")
            origin = shape.get("origin")
            updatedOrigin = ((origin[0] - length/2),(origin[1] - width/2))
            for ii in range(len(xyPart[0])):
                x, y = xyPart[0][ii], xyPart[1][ii]
                if (origin[0] - (length/2)) < x < (origin[0] + (length/2)) and (origin[1] - (width/2)) < y < (origin[1] + (width/2)):
                    withinArea[0].append(x)
                    withinArea[1].append(y)
                else:
                    outsideArea[0].append(x)
                    outsideArea[1].append(y)
            rectangle = patches.Rectangle(updatedOrigin,length,width, edgecolor="black", facecolor="none", zorder = 3)
            ax4.add_patch(rectangle)
            totalExtent = [xyPart[0].min(), xyPart[0].max(), xyPart[1].min(),xyPart[1].max()]
            self.heatmap(ax4, withinArea[0], withinArea[1], scatter=scatter, zorder=2, shapeExtent=totalExtent)
            self.heatmap(ax4, outsideArea[0], outsideArea[1], scatter=scatter, lost=True, zorder=1, shapeExtent=totalExtent)
            percentageInside = len(withinArea[0]) / len(xyPart[0]) * 100
        else: #  No specific shape defined, plot all particles
            self.heatmap(ax4, xyPart[0], xyPart[1], scatter=scatter)
            percentageInside = 100  # No defined aperture so acceptance is 100 % by default

        if defineLim:
            # ax4.axis('equal')
            ax4.set_xlim(minVals[0], maxVals[0])
            ax4.set_ylim(minVals[2], maxVals[2])

        ax4.set_title(f'x, y - Space: {round(percentageInside,4)}% acceptance')
        ax4.set_xlabel(x_labels[i + 1])
        ax4.set_ylabel(y_labels[i + 1])
        ax4.grid(True)

    def heatmap(self, axes, x, y, scatter=False, lost=False, zorder=1, shapeExtent = None):
        '''
        Generates a heatmap (hexbin or scatter with density) of particle distribution.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The axes object to plot on.
        x : np.ndarray
            X-coordinates of particles.
        y : np.ndarray
            Y-coordinates of particles.
        scatter : bool, optional
            If True, creates a scatter plot with density; otherwise, creates a hexbin plot.
        lost : bool, optional
            If True, use the 'lost_cmap' colormap; otherwise, use 'default_cmap'.
            Use for plotting aperature acceptance particles
        zorder : int, optional
            Z-order for plotting.
        shapeExtent : tuple, optional
            (xmin, xmax, ymin, ymax) for setting the extent of the hexbin plot.
            Only used for hexbin plots.
        '''

        cmap = self.lost_cmap if lost else self.default_cmap

        # Scatter plot with color density
        if scatter:
            if not (len(x) == 0 or len(y) == 0):
                xy = np.vstack([x, y])
                density = gaussian_kde(xy)(xy)
                return axes.scatter(x, y, c=density, cmap=cmap, s=self.scatter_size, alpha=self.scatter_alpha)

        # Heatmap with hexagonal tiles
        else:
            return axes.hexbin(x, y, cmap=cmap, extent=shapeExtent, gridsize=self.BINS, zorder=zorder)
            #  cb = ax.figure.colorbar(hb, ax=ax, label='Point Count per Bin') # hb from axes.hexbin return 
