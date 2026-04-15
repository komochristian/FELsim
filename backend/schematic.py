#   Authors: Christian Komo, Niels Bidault
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.transforms import Bbox
from matplotlib.widgets import TextBox
import pandas as pd
import csv
import numpy as np
from beamline import *
from ebeam import beam
import datetime
from tqdm import tqdm
from matplotlib.widgets import Button


# Note: plotBeamTransform may show rounded interval values slightly inaccurate to actual lengths, but calculations are made with exact values, (rounded values only used for visualization)

class draw_beamline:
    def __init__(self):
        '''
        Beamline object creates graphs and plots to visualize beamline data and information
        '''
        self.figsize = (10, 9)
        self.matrixVariables = None
        self.sixdValues = None
        self.DEFAULTINTERVAL = 0.05
        self.DEFAULTINTERVALROUND = 2
        self.DEFAULTSPACINGPERCENTAGE = 0.02

    def driftTransformScatter(self, values, length, plot=True):
        '''
        Simulates particles passing through drift space

        Parameters
        ----------
        values: np.array(list[float][float])
            2D numPy list of particle elements
        length: float
            length of the drift space particle passes through
        plot: bool, optional
            tells function whether to plot particle data or not

        Returns
        -------
        x_transform: list[float]
            list containing each particles' x position
        y_transform: list[float]
            list containing each particles' y position
        '''
        x_pos = values[:, 0]
        phase_x = values[:, 1]
        y_pos = values[:, 2]
        phase_y = values[:, 3]
        x_transform = []
        y_transform = []

        for i in range(len(x_pos)):
            x_transform.append(x_pos[i] + length * phase_x[i])
        for i in range(len(y_pos)):
            y_transform.append(y_pos[i] + length * phase_y[i])

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(x_pos, y_pos, c='blue', s=15, alpha=0.7, label="Initial values")
            ax.scatter(x_transform, y_transform, c='green', s=15, alpha=0.7, label="Transformed values")
            ax.set_xlabel('Position x (mm)')
            ax.set_ylabel('Position y (mm)')
            plt.legend(loc='upper right')
            plt.suptitle("Drift Transformation over " + str(length) + " mm")
            plt.tight_layout()
            plt.show()

        return x_transform, y_transform

    def checkMinMax(self, matrixVariables, maxval, minval):
        '''
        Updates max and min values of a set of particles in a beamline. Used for finding and
        setting the plot boundaries displaying the position of particles.

        Parameters
        ----------
        matrixvairables: np.array(list[float][float])
            A 6 column 2d numpy array, each row containing 6 initial values of each particle's measurements
        maxval: list[float]
            list of each current maximum value for each variable throughout the beamline
        minval: list[float]
            list of each current minimum value for each variable throughout the beamline

        Returns
        -------
        maxval: list[float]
            updated list of maximum values
        minval: list[float]
            updated list of minimum values
        '''
        initialx = matrixVariables[:, 0]
        initialy = matrixVariables[:, 2]
        initialxphase = matrixVariables[:, 1]
        initialyphase = matrixVariables[:, 3]
        initialz = matrixVariables[:, 4]
        initialzphase = matrixVariables[:, 5]
        initialList = [initialx, initialxphase, initialy, initialyphase, initialz, initialzphase]
        for i in range(len(initialList)):
            maximum = max(initialList[i])
            if maximum > maxval[i]:
                maxval[i] = maximum
            minimum = min(initialList[i])
            if minimum < minval[i]:
                minval[i] = minimum
        return maxval, minval

    def _createLabels(self, xaxis, spacing):
        xLabels = []
        defaultSpace = spacing
        for i in range(len(xaxis)):
            if i % defaultSpace == 0:
                xLabels.append(xaxis[i])
            else:
                xLabels.append("")
        return xLabels

    def _csvWriteData(self, name, twiss, x_axis):
        with open(name, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if csvfile.tell() == 0:
                labels = ["z distance"]
                for lab in twiss:
                    for axis in twiss.index:
                        labels.append(str(axis) + ": " + str(lab))
                csvwriter.writerow(labels)
            for i in range(len(x_axis)):
                data = [x_axis[i]]
                for lab in twiss:
                    for axis in twiss.index:
                        list = twiss.at[axis, lab]
                        data.append(list[i])
                csvwriter.writerow(data)

    def _setEqualAxisScaling(self, maxVals, minVals):
        if maxVals[0] > maxVals[2]:
            maxVals[2] = maxVals[0]
        else:
            maxVals[0] = maxVals[2]
        if maxVals[1] > maxVals[3]:
            maxVals[3] = maxVals[1]
        else:
            maxVals[1] = maxVals[3]
        if minVals[0] < minVals[2]:
            minVals[2] = minVals[0]
        else:
            minVals[0] = minVals[2]
        if minVals[1] < minVals[3]:
            minVals[3] = minVals[1]
        else:
            minVals[1] = minVals[3]

    def _saveEPS(self, ax1, ax2, ax3, ax4, ax5, fig, scrollbar):
        bbox = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        x0, y0, x1, y1 = bbox.extents
        pad_left = 0.55
        pad_right = 0.7
        pad_bottom = 0.7
        pad_top = 0.1
        new_bbox = Bbox.from_extents(x0 - pad_left, y0 - pad_bottom, x1 + pad_right, y1 + pad_top)
        scrollbar.ax.set_visible(False)
        fig.savefig(f"dynamics_plot_z_{scrollbar.val}.eps", format='eps', bbox_inches=new_bbox)
        scrollbar.ax.set_visible(True)

        bbox1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbox_quadrants = Bbox.union([bbox1, bbox2, bbox3, bbox4])
        x0, y0, x1, y1 = bbox_quadrants.extents
        pad_left = 0.7
        pad_right = 0.1
        pad_bottom = 0.5
        pad_top = 0.3
        bbox_quadrants_asym = Bbox.from_extents(x0 - pad_left, y0 - pad_bottom, x1 + pad_right, y1 + pad_top)
        fig.savefig(f"phase_space_z_{scrollbar.val}.eps", format='eps', bbox_inches=bbox_quadrants_asym)

    def _getClosestZ(self, plot6dValues, val):
        '''
        Returns closest z distance from val and associated 6d matrix

        Parameters
        ----------
        plot6dValues
        '''
        z_values = np.array(list(plot6dValues.keys()))
        closest_z = z_values[np.argmin(np.abs(z_values - val))]
        matrix = plot6dValues[closest_z]
        return closest_z, matrix

    def simulateData(self, matrixVariables, beamSegments, defineLim, interval):
        """Simulate beam evolution through beamline (data only, no plotting)."""
        ebeam = beam()
        result = ebeam.getXYZ(matrixVariables)
        twiss = result[3]
        plot6dValues = {0: result}
        twiss_aggregated_df = pd.DataFrame(
            {axis: {label: [] for label in twiss.index} for axis in twiss.columns}
        )
        x_axis = [0]
        maxVals = [0, 0, 0, 0, 0, 0]
        minVals = [0, 0, 0, 0, 0, 0]

        for i, axis in enumerate(twiss.index):
            twiss_axis = twiss.loc[axis]
            for label, value in twiss_axis.items():
                twiss_aggregated_df.at[axis, label].append(value)

        if defineLim:
            maxVals, minVals = self.checkMinMax(matrixVariables, maxVals, minVals)
        if interval <= 0:
            interval = self.DEFAULTINTERVAL

        total_intervals = sum(int(segment.length // interval) + 1 for segment in beamSegments)

        with tqdm(total=total_intervals, desc="Simulating Beamline",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for i in range(len(beamSegments)):
                intTrack = beamSegments[i].length

                while intTrack >= interval:
                    matrixVariables = np.array(beamSegments[i].useMatrice(matrixVariables, length=interval))
                    x_axis.append(round(x_axis[-1] + interval, self.DEFAULTINTERVALROUND))

                    if defineLim:
                        maxVals, minVals = self.checkMinMax(matrixVariables, maxVals, minVals)

                    result = ebeam.getXYZ(matrixVariables)
                    twiss = result[3]
                    plot6dValues.update({x_axis[-1]: result})

                    for j, axis in enumerate(twiss.index):
                        twiss_axis = twiss.loc[axis]
                        for label, value in twiss_axis.items():
                            twiss_aggregated_df.at[axis, label].append(value)

                    pbar.update(1)
                    intTrack -= interval

                if intTrack > 0:
                    matrixVariables = np.array(beamSegments[i].useMatrice(matrixVariables, length=intTrack))
                    x_axis.append(round(x_axis[-1] + intTrack, self.DEFAULTINTERVALROUND))

                    if defineLim:
                        maxVals, minVals = self.checkMinMax(matrixVariables, maxVals, minVals)

                    result = ebeam.getXYZ(matrixVariables)
                    twiss = result[3]
                    plot6dValues.update({x_axis[-1]: result})

                    for j, axis in enumerate(twiss.index):
                        twiss_axis = twiss.loc[axis]
                        for label, value in twiss_axis.items():
                            twiss_aggregated_df.at[axis, label].append(value)

                    pbar.update(1)

        return twiss_aggregated_df, plot6dValues, x_axis, maxVals, minVals

    def plotBeamPositionTransform(self, matrixVariables, beamSegments, interval: float = -1, defineLim=True,
                                  saveData=False, saveFig=False, shape={}, plot=True, spacing=True,
                                  matchScaling=True, showIndice=False, scatter=False, apiCall=False,
                                  rendering=True, show_schematic='legacy', curved_trajectory=False):
        '''
        Simulates movement of particles through an accelerator beamline

        Parameters
        ----------
        matrixvairables: np.array(list[float][float])
            A 6r x 10c 2d numpy array containing initial values of each electron's measurements
        beamSegments: list[beamline]
            Numpy array/list containing beamline objects which represent the beam
        interval: float
            Arbitrary number specifying interval for graph to take measurements at
        defineLim: bool, optional
            If plot should change dynamically with the points or stay static.
        saveData: boolean, optional
            Boolean value specifying whether to save data into a csv file or not
        saveFig: bool, float, optional
            Boolean or float value specifying what z position to save eps figure at (default z pos of 0)
        shape: dict{}, optional
            dictionary storing info about the acceptance boundary
            ex. shape, width, radius, length, origin
        plot: bool, optional
            Optional boolean variable to plot simulation or not
        spacing: bool, optional
            Optional variable to optimize spacing of x labels when plotting for readibility
        matchScaling: bool, optional
            Whether to have same x' vs x and y' vs y axis scaling or not.
            defineLim must be True for same scaling setting to work
        showIndice: bool, optional
            Option to display each segment's indice visually
        scatter: bool, optional
            Option to display particle data as hexbins or a scatter color plot
        apicall: bool, optional
            If function is being called from API, changes return type to be compatible with nivo plotting
        rendering: bool, optional
            If false, skips all MatPlotLib plotting and rendering steps for faster data generation
        show_schematic: str or bool, optional
            'legacy': original block drawing (default, backwards compatible)
            'enhanced': new element-specific schematic rendering
            False or None: don't show beamline schematic
        curved_trajectory: bool, optional
            If True and show_schematic='enhanced', show curved reference trajectory for dipoles

        NOTE:
        shape is a dictionary defined as:
        shape = {"shape": "circle", "radius": 5, "origin": (0,5)}
        or
        shape = {"shape": "rectangle", "length": 200, "width": 500, "origin": (10,-4)}
        Only 2 shapes currently: rectangles and circles
        '''
        if apiCall:
            matplotlib.use('Agg')
        ebeam = beam()
        result = ebeam.getXYZ(matrixVariables)
        twiss = result[3]
        plot6dValues = {0: result}
        twiss_aggregated_df = pd.DataFrame(
            {axis: {label: [] for label in twiss.index} for axis in twiss.columns}
        )
        x_axis = [0]
        maxVals = [0, 0, 0, 0, 0, 0]
        minVals = [0, 0, 0, 0, 0, 0]

        for i, axis in enumerate(twiss.index):
            twiss_axis = twiss.loc[axis]
            for label, value in twiss_axis.items():
                twiss_aggregated_df.at[axis, label].append(value)

        if defineLim:
            maxVals, minVals = self.checkMinMax(matrixVariables, maxVals, minVals)
        if interval <= 0:
            interval = self.DEFAULTINTERVAL

        total_intervals = sum(int(segment.length // interval) + 1 for segment in beamSegments)

        with tqdm(total=total_intervals, desc="Simulating Beamline",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for i in range(len(beamSegments)):
                intTrack = beamSegments[i].length

                while intTrack >= interval:
                    matrixVariables = np.array(beamSegments[i].useMatrice(matrixVariables, length=interval))
                    x_axis.append(round(x_axis[-1] + interval, self.DEFAULTINTERVALROUND))

                    if defineLim:
                        maxVals, minVals = self.checkMinMax(matrixVariables, maxVals, minVals)

                    result = ebeam.getXYZ(matrixVariables)
                    twiss = result[3]
                    plot6dValues.update({x_axis[-1]: result})

                    for j, axis in enumerate(twiss.index):
                        twiss_axis = twiss.loc[axis]
                        for label, value in twiss_axis.items():
                            twiss_aggregated_df.at[axis, label].append(value)

                    pbar.update(1)
                    intTrack -= interval

                if intTrack > 0:
                    matrixVariables = np.array(beamSegments[i].useMatrice(matrixVariables, length=intTrack))
                    x_axis.append(round(x_axis[-1] + intTrack, self.DEFAULTINTERVALROUND))

                    if defineLim:
                        maxVals, minVals = self.checkMinMax(matrixVariables, maxVals, minVals)

                    result = ebeam.getXYZ(matrixVariables)
                    twiss = result[3]
                    plot6dValues.update({x_axis[-1]: result})

                    for j, axis in enumerate(twiss.index):
                        twiss_axis = twiss.loc[axis]
                        for label, value in twiss_axis.items():
                            twiss_aggregated_df.at[axis, label].append(value)

                    pbar.update(1)

        envelopeArr = []
        for i in range(0, 3):
            axis = twiss_aggregated_df.index[i]
            emittance = (10 ** -6) * np.array(twiss_aggregated_df.at[axis, twiss_aggregated_df.keys()[0]])
            beta = np.array(twiss_aggregated_df.at[axis, twiss_aggregated_df.keys()[2]])
            envelope = (10 ** 3) * np.sqrt(emittance * beta)
            envelopeArr.append(envelope)
        twiss_aggregated_df[r"Envelope $E$ (mm)"] = envelopeArr

        if saveData:
            name = "simulator-data-" + datetime.datetime.now().strftime(
                '%Y-%m-%d') + "_" + datetime.datetime.now().strftime('%H_%M_%S') + ".csv"
            self._csvWriteData(name, twiss_aggregated_df, x_axis)

        self.matrixVariables = matrixVariables
        self.sixdValues = plot6dValues

        if matchScaling and defineLim:
            self._setEqualAxisScaling(maxVals, minVals)

        if rendering:
            apiAxData, ax5 = self._currentcreateUI(plot6dValues, saveFig, maxVals, minVals, shape, defineLim, scatter,
                                                  twiss_aggregated_df,
                                                  x_axis, spacing, beamSegments, showIndice, plot, apiCall,
                                                  show_schematic, curved_trajectory)

        if apiCall:
            lineAxElements = {
                'twiss': twiss_aggregated_df,
                'x_axis': x_axis
            }
            return apiAxData, lineAxElements

        return twiss_aggregated_df

    def _createLinePlot(self, ax5, twiss_aggregated_df, x_axis, spacing, showIndice,
                       beamSegments, show_schematic='legacy', curved_trajectory=False):
        """
        Create line plot showing envelope and dispersion evolution along beamline.

        Parameters
        ----------
        ax5 : matplotlib.axes
            Axis for the line plot
        twiss_aggregated_df : pd.DataFrame
            Dataframe containing twiss parameters at each s position
        x_axis : list[float]
            List of s positions along beamline
        spacing : bool
            Whether to optimize x-tick spacing for readability
        showIndice : bool
            Whether to show element indices on schematic
        beamSegments : list
            List of beamline elements (lattice objects or dicts)
        show_schematic : str or bool, optional
            'legacy': original block drawing (default, backwards compatible)
            'enhanced': new element-specific schematic rendering
            False or None: don't show beamline schematic
        curved_trajectory : bool, optional
            If True and show_schematic='enhanced', show curved reference trajectory
            for dipoles (default: False)

        Returns
        -------
        lineList : list
            List of dispersion line objects
        ax6 : matplotlib.axes
            Twin axis for dispersion
        ax5 : matplotlib.axes
            Main axis
        """
        colors = ['dodgerblue', 'crimson', 'yellow', 'green']

        for i in range(0, 2):
            axis = twiss_aggregated_df.index[i]
            envelope = np.array(twiss_aggregated_df.at[axis, twiss_aggregated_df.keys()[7]])
            ax5.plot(x_axis, envelope,
                     color=colors[i], linestyle='-',
                     label=r'$E_' + axis + '$ (mm)')

        ax5.set_xticks(x_axis)
        ax5.set_xlim(0, x_axis[-1])

        if spacing:
            totalLen = x_axis[-1]
            lastTick = x_axis[0]
            xTickLab = [lastTick]
            for tick in x_axis[1:]:
                if (tick - lastTick) / totalLen > self.DEFAULTSPACINGPERCENTAGE:
                    xTickLab.append(tick)
                    lastTick = tick
                else:
                    xTickLab.append("")
            ax5.set_xticklabels(xTickLab, rotation=45, ha='right')
        else:
            ax5.set_xticklabels(x_axis, rotation=45, ha='right')

        ax5.tick_params(labelsize=9)
        ax5.set_xlabel(r"Distance from start of beam (m)")
        ax5.set_ylabel(r"Envelope $E$ (mm)")
        ax5.legend(loc='upper left')

        line = None
        lineList = []
        ax6 = ax5.twinx()
        for i in range(0, 2):
            axis = twiss_aggregated_df.index[i]
            dispersion = np.array(twiss_aggregated_df.at[axis, twiss_aggregated_df.keys()[4]])
            line, = ax6.plot(x_axis, dispersion,
                             color=colors[i], linestyle='--',
                             label=r'$D_' + axis + '$ (m)')
            lineList.append(line)
        ax6.set_ylabel(r'Dispersion $D$ (m)')
        ax6.legend(loc='upper right')

        if show_schematic == 'enhanced':
            ymin, ymax = ax5.get_ylim()
            ax5.set_ylim(ymin - (ymax * 0.05), ymax)
            ymin, ymax = ax5.get_ylim()

            schematic_height = 0.08 if not curved_trajectory else 0.12
            schematic_ax = ax5.inset_axes([0, -0.15, 1, schematic_height])
            self.plot_beamline_schematic(beamSegments, ax=schematic_ax,
                                         show_curved_trajectory=curved_trajectory,
                                         height_scale=0.4,
                                         show_labels=showIndice)
            schematic_ax.set_xlabel('')

        elif show_schematic == 'legacy':
            ymin, ymax = ax5.get_ylim()
            ax5.set_ylim(ymin - (ymax * 0.05), ymax)
            ymin, ymax = ax5.get_ylim()
            blockstart = 0
            moveUp = True
            for i, seg in enumerate(beamSegments):
                elem_info = self._extract_element_info(seg)
                length = elem_info['length']
                color = elem_info['color']

                rectangle = patches.Rectangle((blockstart, ymin), length, ymax * 0.05,
                                              linewidth=1, edgecolor=color, facecolor=color)
                ax5.add_patch(rectangle)
                if showIndice:
                    moveUp = not moveUp
                    recx = rectangle.get_x()
                    recy = rectangle.get_y()
                    if moveUp:
                        ax5.text(recx, recy / 2, str(i), size='small')
                    else:
                        ax5.text(recx, recy, str(i), size='small')
                blockstart += length

        return lineList, ax6, ax5

    def plot_beamline_schematic(self, beamSegments, ax=None, show_curved_trajectory=False,
                                height_scale=0.3, show_labels=False, element_spacing=0.01):
        """
        Plot beamline schematic showing element types and positions.
        Works with both FELsim lattice objects and BeamlineBuilder dicts.

        Parameters
        ----------
        beamSegments : list
            List of beamline elements (lattice objects or dicts)
        ax : matplotlib.axes, optional
            Axis to plot on (creates new if None)
        show_curved_trajectory : bool, optional
            If True, show actual curved floor coordinates (default: False)
        height_scale : float, optional
            Height scaling for element rectangles (default: 0.3)
        show_labels : bool, optional
            Show element names/indices (default: False)
        element_spacing : float, optional
            Vertical spacing (default: 0.01)

        Returns
        -------
        ax : matplotlib.axes
            The axis containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        if show_curved_trajectory:
            # Plot in actual 2D floor coordinates
            self._plot_curved_trajectory_mode(ax, beamSegments, height_scale, show_labels)
        else:
            # Standard s-coordinate plot
            self._plot_straight_mode(ax, beamSegments, height_scale, show_labels)

        return ax

    def _plot_straight_mode(self, ax, beamSegments, height_scale, show_labels):
        """Plot in standard s-coordinate mode (straight reference)."""
        s_pos = 0.0

        for i, elem in enumerate(beamSegments):
            elem_info = self._extract_element_info(elem)
            length = elem_info['length']
            elem_type = elem_info['type']

            self._draw_element_rect(ax, s_pos, length, elem_type, elem_info, height_scale)

            if show_labels:
                label = elem_info.get('name', f'{elem_type}_{i}')
                if label:
                    ax.text(s_pos + length / 2, height_scale * 0.6, label,
                            rotation=90, ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(s_pos + length / 2, height_scale * 0.6, str(i),
                            rotation=0, ha='center', va='bottom', fontsize=8)

            s_pos += length

        ax.axhline(0, color='k', linewidth=1, linestyle='-')
        ax.set_xlim(0, s_pos)
        ax.set_ylim(-height_scale * 1.2, height_scale * 1.2)
        ax.set_xlabel('s (m)', fontsize=10)
        ax.set_ylabel('', fontsize=10)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    def _plot_curved_trajectory_mode(self, ax, beamSegments, height_scale, show_labels):
        """Plot showing actual curved trajectory in floor coordinates."""
        # Track position and angle in floor coordinates
        x_pos = 0.0
        y_pos = 0.0
        angle = 0.0  # Current direction angle

        traj_x = [x_pos]
        traj_y = [y_pos]

        # Also track s-position for element placement along trajectory
        s_pos = 0.0

        for i, elem in enumerate(beamSegments):
            elem_info = self._extract_element_info(elem)
            length = elem_info['length']
            elem_type = elem_info['type']

            # Draw element at current floor position
            self._draw_element_rect_floor(ax, x_pos, y_pos, angle, length, elem_type,
                                          elem_info, height_scale)

            # Update trajectory
            if elem_type in ['DPH', 'DIPOLE']:
                bend_angle_rad = np.radians(elem_info['angle'])
                if abs(bend_angle_rad) > 1e-10:
                    # Curved section
                    traj_x, traj_y, x_pos, y_pos, angle = self._add_arc_to_trajectory_2d(
                        traj_x, traj_y, x_pos, y_pos, angle, length, bend_angle_rad
                    )
                else:
                    # Straight dipole (zero bend)
                    x_pos += length * np.cos(angle)
                    y_pos += length * np.sin(angle)
                    traj_x.append(x_pos)
                    traj_y.append(y_pos)
            else:
                # Straight section
                x_pos += length * np.cos(angle)
                y_pos += length * np.sin(angle)
                traj_x.append(x_pos)
                traj_y.append(y_pos)

            if show_labels:
                label = elem_info.get('name', f'{elem_type}_{i}')
                # Place label perpendicular to trajectory
                label_x = (traj_x[-2] + traj_x[-1]) / 2 if len(traj_x) > 1 else x_pos
                label_y = (traj_y[-2] + traj_y[-1]) / 2 if len(traj_y) > 1 else y_pos
                offset_x = -height_scale * 0.6 * np.sin(angle)
                offset_y = height_scale * 0.6 * np.cos(angle)

                if label:
                    ax.text(label_x + offset_x, label_y + offset_y, label,
                            rotation=np.degrees(angle), ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(label_x + offset_x, label_y + offset_y, str(i),
                            rotation=0, ha='center', va='bottom', fontsize=8)

            s_pos += length

        # Plot horizontal reference axis (initial straight direction)
        x_extent = max(abs(max(traj_x)), abs(min(traj_x))) * 1.15
        ax.plot([0, x_extent], [0, 0], 'k--', linewidth=1.0, alpha=0.4,
                label='Horizontal axis', zorder=0)

        # Plot the trajectory
        ax.plot(traj_x, traj_y, 'k-', linewidth=1.5, label='Reference trajectory')
        ax.legend(loc='best', fontsize=8)

        # Set equal aspect ratio for proper geometry
        ax.set_aspect('equal', adjustable='datalim')

        # Auto-scale with margins
        x_range = max(traj_x) - min(traj_x)
        y_range = max(traj_y) - min(traj_y)
        margin = max(x_range, y_range, 1.0) * 0.15

        ax.set_xlim(min(traj_x) - margin, max(traj_x) + margin)
        ax.set_ylim(min(traj_y) - margin, max(traj_y) + margin)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)

    def _draw_element_rect_floor(self, ax, x_start, y_start, angle, length,
                                 elem_type, elem_info, height_scale):
        """Draw element rectangle in floor coordinates along the trajectory."""
        color = elem_info['color']

        if elem_type == 'QPF':
            height = height_scale
            offset = 0
        elif elem_type == 'QPD':
            height = -height_scale
            offset = -height_scale
        elif elem_type == 'DRIFT':
            # Don't draw drifts in floor coordinate mode (clutters the plot)
            return
        else:
            # Dipoles and others
            height = height_scale * 0.5
            offset = -height / 2

        # Rectangle corners in local coordinates
        # We'll draw along the trajectory direction
        corners_local = np.array([
            [0, offset],
            [length, offset],
            [length, offset + abs(height)],
            [0, offset + abs(height)]
        ])

        # Rotate and translate to floor coordinates
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        corners_floor = corners_local @ rotation.T
        corners_floor[:, 0] += x_start
        corners_floor[:, 1] += y_start

        poly = plt.Polygon(corners_floor, facecolor=color, edgecolor='black',
                           linewidth=0.5, alpha=0.7)
        ax.add_patch(poly)

    def _extract_element_info(self, elem):
        """Extract standardized info from either lattice object or dict."""
        info = {}

        if isinstance(elem, dict):
            info['type'] = elem.get('type', 'UNKNOWN')
            info['length'] = elem.get('length', 0.0)
            info['angle'] = elem.get('angle', 0.0)
            info['current'] = elem.get('current', 0.0)
            info['color'] = self._get_color_for_type(info['type'])
            info['name'] = elem.get('name', '')
            return info

        class_name = elem.__class__.__name__

        type_map = {
            'driftLattice': 'DRIFT',
            'qpfLattice': 'QPF',
            'qpdLattice': 'QPD',
            'dipole': 'DPH',
            'dipole_wedge': 'DPW'
        }

        info['type'] = type_map.get(class_name, class_name)
        info['length'] = getattr(elem, 'length', 0.0)
        info['angle'] = getattr(elem, 'angle', 0.0)
        info['current'] = getattr(elem, 'current', 0.0)
        info['color'] = getattr(elem, 'color', self._get_color_for_type(info['type']))
        info['name'] = getattr(elem, 'name', '')

        return info

    def _get_color_for_type(self, elem_type):
        """Standard color scheme for element types."""
        colors = {
            'DRIFT': 'white',
            'QPF': 'cornflowerblue',
            'QPD': 'lightcoral',
            'DPH': 'forestgreen',
            'DPW': 'lightgreen',
            'DIPOLE': 'forestgreen',
            'SBEND': 'forestgreen'
        }
        return colors.get(elem_type, 'gray')

    def _draw_element_rect(self, ax, s_start, length, elem_type, elem_info, height_scale):
        """Draw standard rectangle for element."""
        color = elem_info['color']

        if elem_type == 'QPF':
            height = height_scale
            y_pos = 0
        elif elem_type == 'QPD':
            height = -height_scale
            y_pos = -height_scale
        elif elem_type == 'DRIFT':
            ax.plot([s_start, s_start + length], [0, 0],
                    'k-', linewidth=0.5, alpha=0.3)
            return
        else:
            height = height_scale * 0.5
            y_pos = -height / 2

        rect = patches.Rectangle((s_start, y_pos), length, abs(height),
                                 facecolor=color, edgecolor='black',
                                 linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)

    def _draw_dipole_curved(self, ax, s_start, length, bend_angle,
                            cumulative_angle, height_scale, color):
        """Draw arc representation for dipole with bending."""
        if abs(bend_angle) < 1e-10:
            return

        rect = patches.Rectangle((s_start, -height_scale * 0.25), length, height_scale * 0.5,
                                 facecolor=color, edgecolor='black',
                                 linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)

        angle_deg = np.degrees(bend_angle)
        ax.text(s_start + length / 2, height_scale * 0.4,
                f'{angle_deg:.1f}°', ha='center', fontsize=7, color='white',
                weight='bold')

    def _add_arc_to_trajectory(self, s_traj, y_traj, s_start, length, bend_angle):
        """
        Add circular arc to trajectory for sector dipole.

        For sector dipoles, the reference trajectory follows a circular arc.
        This function computes the trajectory in the s-y plane where:
        - s is the longitudinal coordinate along the beamline
        - y is the perpendicular displacement due to bending
        """
        if abs(bend_angle) < 1e-10:
            s_traj.append(s_start + length)
            y_traj.append(y_traj[-1])
            return s_traj, y_traj

        rho = length / bend_angle

        n_points = max(10, int(abs(bend_angle) * 50))

        y_start = y_traj[-1]  # FIX: Start from CURRENT y position, not initial

        for i in range(1, n_points + 1):
            frac = i / n_points
            theta = bend_angle * frac
            s_current = s_start + length * frac

            y_displacement = rho * (1 - np.cos(theta))

            s_traj.append(s_current)
            y_traj.append(y_start + y_displacement)  # FIX: Add to starting y, not y_traj[0]

        return s_traj, y_traj

    def _add_arc_to_trajectory_2d(self, traj_x, traj_y, x_start, y_start, angle_start, length, bend_angle):
        """
        Add circular arc to 2D trajectory for sector dipole.

        Parameters
        ----------
        traj_x, traj_y : list
            Trajectory coordinate lists to append to
        x_start, y_start : float
            Starting position
        angle_start : float
            Starting angle (radians)
        length : float
            Arc length of dipole
        bend_angle : float
            Bending angle (radians)

        Returns
        -------
        traj_x, traj_y : list
            Updated trajectory lists
        x_end, y_end : float
            Final position after arc
        angle_end : float
            Final angle after arc
        """
        if abs(bend_angle) < 1e-10:
            x_end = x_start + length * np.cos(angle_start)
            y_end = y_start + length * np.sin(angle_start)
            traj_x.append(x_end)
            traj_y.append(y_end)
            return traj_x, traj_y, x_end, y_end, angle_start

        rho = length / bend_angle  # Bending radius

        # Center of the circular arc
        # Perpendicular to initial direction
        x_center = x_start - rho * np.sin(angle_start)
        y_center = y_start + rho * np.cos(angle_start)

        # Parametric arc
        n_points = max(10, int(abs(bend_angle) * 50))

        for i in range(1, n_points + 1):
            frac = i / n_points
            theta = bend_angle * frac

            # Angle in global coordinates
            current_angle = angle_start + theta

            # Position on arc
            x_current = x_center + rho * np.sin(current_angle)
            y_current = y_center - rho * np.cos(current_angle)

            traj_x.append(x_current)
            traj_y.append(y_current)

        # Final position and angle
        x_end = traj_x[-1]
        y_end = traj_y[-1]
        angle_end = angle_start + bend_angle

        return traj_x, traj_y, x_end, y_end, angle_end

    def _add_bend_to_trajectory(self, s_traj, y_perp, angle_in, s_start, length, bend_angle):
        """
        Add bend to trajectory, tracking perpendicular displacement from straight reference.

        Parameters
        ----------
        s_traj : list
            s-coordinates
        y_perp : list
            Perpendicular displacements from initial straight reference
        angle_in : float
            Incoming trajectory angle relative to initial direction (radians)
        s_start : float
            Starting s position of bend
        length : float
            Arc length of bend
        bend_angle : float
            Bending angle (radians)

        Returns
        -------
        s_traj, y_perp : list
            Updated trajectory
        angle_out : float
            Outgoing angle
        """
        if abs(bend_angle) < 1e-10:
            # Straight section at current angle
            s_traj.append(s_start + length)
            # Displacement increases linearly with s if traveling at an angle
            y_perp.append(y_perp[-1] + length * np.sin(angle_in))
            return s_traj, y_perp, angle_in

        rho = length / bend_angle  # Bending radius
        y_start = y_perp[-1]  # Starting perpendicular displacement

        n_points = max(10, int(abs(bend_angle) * 50))

        for i in range(1, n_points + 1):
            frac = i / n_points
            s_current = s_start + length * frac
            theta = bend_angle * frac  # Angle traversed so far in this dipole

            # Current trajectory angle relative to initial reference
            current_angle = angle_in + theta

            # Calculate position along arc in local dipole coordinates
            # Arc goes from 0 to length, centered at angle = angle_in + bend_angle/2
            arc_length = length * frac

            # Perpendicular displacement has two components:
            # 1. Displacement accumulated before this dipole (y_start)
            # 2. Additional displacement due to traveling through dipole at an angle

            # For a sector bend, the chord length is 2*rho*sin(theta/2)
            # The sagitta (perpendicular distance from chord to arc midpoint) is rho*(1-cos(theta))
            # But we need displacement relative to the INITIAL straight line, not the chord

            # Displacement in y (perpendicular to initial direction):
            # If we imagine unrolling the bend, the particle moves:
            # - Along the arc by arc_length
            # - The y-component of this arc segment depends on the local angle

            # More simply: integrate sin(angle) over the arc
            # For small segments: Δy ≈ Δs * sin(local_angle)
            # For sector dipole: y(θ) = ρ * sin(θ) when starting from angle_in=0
            # General case: need to account for angle_in

            # Using exact formula for circular arc:
            # Starting at angle angle_in, after rotating by theta, we've moved:
            # Δy = ρ * [sin(angle_in + theta) - sin(angle_in)]

            delta_y = rho * (np.sin(angle_in + theta) - np.sin(angle_in))

            s_traj.append(s_current)
            y_perp.append(y_start + delta_y)

        angle_out = angle_in + bend_angle

        return s_traj, y_perp, angle_out

    def _currentcreateUI(self, plot6dValues, saveFig, maxVals, minVals, shape, defineLim, scatter, twiss_aggregated_df,
                        x_axis, spacing, beamSegments, showIndice, plot, apiCall, show_schematic='legacy',
                        curved_trajectory=False):
        ebeam = beam()

        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.8, 0.8, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])

        if isinstance(saveFig, bool):
            savePhaseSpace = saveFig
            saveZ = 0
        elif isinstance(saveFig, (int, float)):
            savePhaseSpace = True
            saveZ = saveFig
        else:
            raise ValueError("saveFig must be either False, True, or a float (z value)")

        closest_initial_z, matrix = self._getClosestZ(plot6dValues, saveZ)

        ebeam.plotXYZ(matrix[2], matrix[0], matrix[1], matrix[3], ax1, ax2, ax3, ax4, maxVals, minVals, defineLim,
                      shape, scatter=scatter)

        ax5 = plt.subplot(gs[2, :])
        lineList, ax6, m = self._createLinePlot(ax5, twiss_aggregated_df, x_axis, spacing, showIndice,
                                               beamSegments, show_schematic, curved_trajectory)

        if apiCall:
            axesDict = {}
            for index, mat in plot6dValues.items():
                fig = plt.figure(figsize=(10, 7))
                gs = gridspec.GridSpec(2, 2)
                ax1 = plt.subplot(gs[0, 0])
                ax2 = plt.subplot(gs[0, 1])
                ax3 = plt.subplot(gs[1, 0])
                ax4 = plt.subplot(gs[1, 1])
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                ebeam.plotXYZ(mat[2], mat[0], mat[1], mat[3], ax1, ax2, ax3, ax4, maxVals, minVals, defineLim, shape,
                              scatter=scatter)
                plt.tight_layout()

                axesDict.update({index: ax1})
            deadFig, ax5 = plt.subplots(figsize=(10, 1))
            lineList, ax6, m = self._createLinePlot(ax5, twiss_aggregated_df, x_axis, spacing, showIndice,
                                                   beamSegments, show_schematic, curved_trajectory)
            return axesDict, ax5

        plt.suptitle("Beamline Simulation")
        plt.tight_layout()

        dimensions = ax5.get_position().bounds
        scrollax = plt.axes((dimensions[0], 0.01, dimensions[2], 0.01), facecolor='lightgoldenrodyellow')
        scrollbar = Slider(scrollax, f'z: {closest_initial_z}', 0, x_axis[-1], valinit=closest_initial_z,
                           valstep=np.array(x_axis))
        scrollbar.valtext.set_visible(False)

        def update_scroll(val):
            matrix = plot6dValues.get(scrollbar.val)
            if matrix is None:
                val, matrix = self._getClosestZ(plot6dValues, val)
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            scrollbar.label.set_text("z: " + str(val))
            ebeam.plotXYZ(matrix[2], matrix[0], matrix[1], matrix[3], ax1, ax2, ax3, ax4, maxVals, minVals,
                          defineLim, shape, scatter=scatter)
            fig.canvas.draw_idle()

        scrollbar.on_changed(update_scroll)

        if savePhaseSpace:
            self._saveEPS(ax1, ax2, ax3, ax4, ax5, fig, scrollbar)

        lineTwissData = []
        twissDataNames = []
        for key in twiss_aggregated_df.keys():
            lineTwissData.append([twiss_aggregated_df.at['x', key], twiss_aggregated_df.at['y', key]])
            twissDataNames.append(key)

        class CircularList:
            index = 0

            def drawNewLines(self, ind):
                data = lineTwissData[ind % len(lineTwissData)]
                label = twissDataNames[ind % len(twissDataNames)]
                for i, axis in enumerate(['x', 'y']):
                    line = lineList[i]
                    line.set_ydata(data[i])

                    line.set_label(label.split(' ')[0] + '$_' + axis + '$')

                ax6.relim()
                ax6.autoscale_view()
                ax6.set_ylabel(label)
                ax6.legend(loc='upper right')
                plt.draw()

            def nextL(self, event):
                self.index += 1
                self.drawNewLines(self.index)

            def prevL(self, event):
                self.index -= 1
                self.drawNewLines(self.index)

        axprev = fig.add_axes([dimensions[0] + dimensions[2] + 0.02, dimensions[1] - 0.04, 0.03, 0.03])
        axnext = fig.add_axes([dimensions[0] + dimensions[2] + 0.05, dimensions[1] - 0.04, 0.03, 0.03])
        bnext = Button(axnext, 'Next', hovercolor="lightblue")
        bprev = Button(axprev, 'Prev', hovercolor="lightblue")
        circList = CircularList()
        bnext.on_clicked(circList.nextL)
        bprev.on_clicked(circList.prevL)

        def goToZ(zCoord):
            try:
                zCoord = float(zCoord)
                scrollbar.set_val(zCoord)
            except ValueError:
                pass

        topRightDim = ax2.get_position().bounds
        textBoxHeight = 0.03
        textAx = fig.add_axes(
            [topRightDim[0] + topRightDim[2] + 0.02, topRightDim[1] + topRightDim[3] - textBoxHeight, 0.05,
             textBoxHeight])
        textBox = TextBox(textAx, label="Input Z", hovercolor="lightblue", initial=str(saveZ))
        textBox.on_submit(goToZ)
        textBox.label.set_verticalalignment('top')
        textBox.label.set_horizontalalignment('center')
        textBox.label.set_position((0.5, -0.2))

        def _saveEPS(event):
            self._saveEPS(ax1, ax2, ax3, ax4, ax5, fig, scrollbar)

        textBoxDim = textAx.get_position().bounds
        axSave = fig.add_axes([textBoxDim[0], textBoxDim[1] - 0.07, 0.05, 0.03])
        saveButton = Button(axSave, 'Save .eps', hovercolor="lightblue")
        saveButton.on_clicked(_saveEPS)

        if plot:
            plt.show()

        return [[], [], [], []], ax5