import sys
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

# PyQt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QFormLayout, QTextEdit,
    QSlider, QDoubleSpinBox, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches # For beamline visualization

# Import qt_material for modern styling
from qt_material import apply_stylesheet

# --- IMPORTANT: Replace these with your actual imports ---
# from beamline import * # Assuming this imports lattice, driftLattice, qpfLattice, dipole, etc.
# from ebeam import beam # Assuming this imports your beam and BeamAnalyzer classes

# --- Placeholder Dummy Classes (for running the example without your full code) ---
# Replace these with your actual classes once your beamline.py and ebeam.py are ready
class DummyLattice:
    def __init__(self, length, fringeType='decay', color="gray"):
        self.length = length
        self.fringeType = fringeType
        self.color = color # Added color for visualization

    def useMatrice(self, particles, length=None):
        # Dummy transformation: particles just drift a bit
        # This needs to be replaced with your actual useMatrice logic from your elements
        current_len = length if length is not None else self.length
        # Simulate a small random perturbation or drift
        particles_copy = particles.copy()
        particles_copy[:, 0] += current_len * particles_copy[:, 1] # x += L * x'
        particles_copy[:, 2] += current_len * particles_copy[:, 3] # y += L * y'
        return particles_copy

class DummyDrift(DummyLattice):
    def __init__(self, length):
        super().__init__(length, color="skyblue")

class DummyQPF(DummyLattice):
    def __init__(self, length, gradient):
        super().__init__(length, color="red")
        self.gradient = gradient # Placeholder for actual physics

class DummyQPD(DummyLattice):
    def __init__(self, length, gradient):
        super().__init__(length, color="blue")
        self.gradient = gradient # Placeholder for actual physics

class DummyDipole(DummyLattice):
    def __init__(self, length, angle):
        super().__init__(length, color="green")
        self.angle = angle # Placeholder for actual physics

class DummyDipoleWedge(DummyLattice):
    def __init__(self, length, angle):
        super().__init__(length, color="lightgreen")
        self.angle = angle # Placeholder for actual physics


class DummyBeam:
    def __init__(self):
        self.E = 45 # MeV, placeholder for beam energy

    def cal_twiss(self, particles, ddof=1):
        # Simplified dummy calculation for demonstration
        dist_avg = np.mean(particles, axis=0)
        dist_cov = np.cov(particles, rowvar=False, ddof=ddof)

        # Dummy Twiss parameters (replace with your actual calculation)
        # Ensure keys match those expected in twiss_aggregated_df
        twiss_data = {
            r'$\epsilon$ ($\pi$.mm.mrad)': {'x': 0.5, 'y': 0.6, 'z': 100},
            r'$\alpha$': {'x': 0.1, 'y': -0.2, 'z': 0.0},
            r'$\beta$ (m)': {'x': 1.0, 'y': 1.2, 'z': 10.0},
            r'$D$ (mm)': {'x': 0.05, 'y': -0.02, 'z': 0.0}, # Dummy Dispersion
            r"$D'$ (mrad)": {'x': 0.01, 'y': 0.005, 'z': 0.0}, # Dummy Dispersion Prime
        }
        twiss_df = pd.DataFrame.from_dict(twiss_data, orient='columns') # orient='columns' for column-major
        twiss_df.index.name = "axis" # Ensure index name is set for consistency
        return dist_avg, dist_cov, twiss_df

    def gen_6d_gaussian(self, mean, std_dev, num_particles):
        # Generate dummy 6D Gaussian particles for demonstration
        return np.random.normal(mean, std_dev, size=(num_particles, 6))

    def plotXYZ(self, dist_6d, std1, std6, twiss, ax1, ax2, ax3, ax4, maxVals, minVals, defineLim, shape, scatter=False):
        # Placeholder for your actual plotXYZ function
        # This will draw dummy data on the Matplotlib axes
        x, xp, y, yp, z, zp = dist_6d[:, 0], dist_6d[:, 1], dist_6d[:, 2], \
                              dist_6d[:, 3], dist_6d[:, 4], dist_6d[:, 5]

        # Use DummyBeamAnalyzer's heatmap method
        DummyBeamAnalyzer().heatmap(ax1, x, xp, scatter=scatter)
        ax1.set_title("x-x' Phase Space")
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("x' (mrad)")
        ax1.grid(True)
        # Set limits if definedLim is True (using dummy max/min here)
        if defineLim and len(maxVals) >= 2:
            ax1.set_xlim(minVals[0], maxVals[0])
            ax1.set_ylim(minVals[1], maxVals[1])

        DummyBeamAnalyzer().heatmap(ax2, y, yp, scatter=scatter)
        ax2.set_title("y-y' Phase Space")
        ax2.set_xlabel("y (mm)")
        ax2.set_ylabel("y' (mrad)")
        ax2.grid(True)
        if defineLim and len(maxVals) >= 4:
            ax2.set_xlim(minVals[2], maxVals[2])
            ax2.set_ylim(minVals[3], maxVals[3])

        DummyBeamAnalyzer().heatmap(ax3, x, y, scatter=scatter)
        ax3.set_title("x-y Cross Section")
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("y (mm)")
        ax3.grid(True)
        if defineLim and len(maxVals) >= 4: # Using x and y bounds here
            ax3.set_xlim(minVals[0], maxVals[0])
            ax3.set_ylim(minVals[2], maxVals[2])

        DummyBeamAnalyzer().heatmap(ax4, z, zp, scatter=scatter)
        ax4.set_title("z-z' Phase Space")
        ax4.set_xlabel("z (mm)")
        ax4.set_ylabel("z' (mrad)")
        ax4.grid(True)
        if defineLim and len(maxVals) >= 6:
            ax4.set_xlim(minVals[4], maxVals[4])
            ax4.set_ylim(minVals[5], maxVals[5])


class DummyBeamAnalyzer:
    def __init__(self):
        pass

    def getXYZ(self, dist_6d):
        # Simplified dummy for getXYZ
        twiss = DummyBeam().cal_twiss(dist_6d)
        return None, None, dist_6d, twiss # std1, std6, final_particles, twiss_df

    def heatmap(self, axes, x, y, scatter=False, lost=False, zorder=1, shapeExtent=None):
        if scatter:
            axes.scatter(x, y, s=5, alpha=0.7)
        else:
            axes.hexbin(x, y, gridsize=50, cmap='viridis')

# --- End of Placeholder Classes ---


class BeamlineSimulatorUI(QMainWindow):
    DEFAULTINTERVAL = 0.05
    DEFAULTINTERVALROUND = 2
    DEFAULTSPACINGPERCENTAGE = 0.02

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beamline Simulation UI")
        self.setGeometry(100, 100, 1200, 900) # Increased size for more content

        # --- Instantiate your actual beam and analyzer objects here ---
        self.ebeam = DummyBeam() # Replace with beam()
        self.beam_analyzer = DummyBeamAnalyzer() # Replace with BeamAnalyzer()

        # --- Data Storage ---
        self.matrixVariables = None
        self.plot6dValues = {} # Stores 6D particle data at each z-interval
        self.twiss_aggregated_df = None # Stores Twiss parameters over z
        self.x_axis = [] # Z-positions where data was recorded
        self.maxVals = [0]*6 # Max values for x, x', y, y', z, z' for plot limits
        self.minVals = [0]*6 # Min values for x, x', y, y', z, z' for plot limits
        self.beamSegments = [] # List of beamline elements

        # --- UI State Variables ---
        self.current_twiss_index = 0 # For 'Next' / 'Prev' buttons
        self.current_z_index = 0 # Current index for slider value within x_axis
        self.save_z_initial = 0 # Default Z for saving

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Parameters and Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(350)

        # Beam Parameters Group
        beam_params_group = QGroupBox("Beam Parameters")
        beam_params_layout = QFormLayout(beam_params_group)
        self.num_particles_input = QLineEdit("10000")
        beam_params_layout.addRow("Num Particles:", self.num_particles_input)
        self.particle_type_combo = QComboBox()
        self.particle_type_combo.addItems(["electron", "proton", "12,5 (C12 5+)"])
        beam_params_layout.addRow("Particle Type:", self.particle_type_combo)
        self.kinetic_energy_input = QLineEdit("45") # MeV
        beam_params_layout.addRow("Kinetic Energy (MeV):", self.kinetic_energy_input)
        left_layout.addWidget(beam_params_group)

        # Initial Std Dev Group
        std_dev_group = QGroupBox("Initial Std Dev (6D)")
        std_dev_layout = QFormLayout(std_dev_group)
        self.std_x = QLineEdit("10")
        self.std_xp = QLineEdit("0.1")
        self.std_y = QLineEdit("10")
        self.std_yp = QLineEdit("0.1")
        self.std_z = QLineEdit("1")
        self.std_zp = QLineEdit("0.01")
        std_dev_layout.addRow("Std Dev x (mm):", self.std_x)
        std_dev_layout.addRow("Std Dev x' (mrad):", self.std_xp)
        std_dev_layout.addRow("Std Dev y (mm):", self.std_y)
        std_dev_layout.addRow("Std Dev y' (mrad):", self.std_yp)
        std_dev_layout.addRow("Std Dev z (10^-3):", self.std_z)
        std_dev_layout.addRow("Std Dev z' (10^-3):", self.std_zp)
        left_layout.addWidget(std_dev_group)

        # Beamline Definition (Simplified for example)
        beamline_group = QGroupBox("Beamline Elements (Length, Type)")
        beamline_layout = QVBoxLayout(beamline_group)
        # Example: Input for a simple beamline sequence
        # Users might input a string like "D(1) QPF(0.5,0.1) D(2)"
        self.beamline_def_input = QTextEdit("D(1)\nQPF(0.5,0.1)\nD(2)")
        self.beamline_def_input.setPlaceholderText("Enter beamline elements (e.g., D(1)\nQ(0.5,0.1))")
        self.beamline_def_input.setMinimumHeight(100)
        beamline_layout.addWidget(self.beamline_def_input)
        left_layout.addWidget(beamline_group)


        # Simulation Options Group
        sim_options_group = QGroupBox("Simulation Options")
        sim_options_layout = QFormLayout(sim_options_group)
        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItems(["Hexbin (Density)", "Scatter (Individual Particles)"])
        sim_options_layout.addRow("Plot Style:", self.plot_style_combo)
        # Add interval, defineLim, matchScaling, showIndice as QComboBox or QLineEdit
        self.interval_input = QLineEdit(str(self.DEFAULTINTERVAL))
        sim_options_layout.addRow("Interval (m):", self.interval_input)
        self.define_lim_checkbox = QComboBox()
        self.define_lim_checkbox.addItems(["True", "False"])
        sim_options_layout.addRow("Dynamic Plot Limits:", self.define_lim_checkbox)
        self.match_scaling_checkbox = QComboBox()
        self.match_scaling_checkbox.addItems(["True", "False"])
        sim_options_layout.addRow("Match Scaling (x/y):", self.match_scaling_checkbox)
        self.show_indice_checkbox = QComboBox()
        self.show_indice_checkbox.addItems(["False", "True"])
        sim_options_layout.addRow("Show Segment Index:", self.show_indice_checkbox)

        left_layout.addWidget(sim_options_group)

        # Run Button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation_and_update_ui)
        left_layout.addWidget(self.run_button)

        # Spacer to push everything to the top
        left_layout.addStretch(1)

        main_layout.addWidget(left_panel)

        # --- Right Panel: Plots and Results ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Phase Space Plots (Top Figure)
        self.phase_space_figure = Figure(figsize=(8, 6))
        self.phase_space_canvas = FigureCanvas(self.phase_space_figure)
        right_layout.addWidget(self.phase_space_canvas)

        # Dynamics/Envelope Plots (Bottom Figure)
        self.dynamics_figure = Figure(figsize=(8, 4)) # Smaller height
        self.dynamics_canvas = FigureCanvas(self.dynamics_figure)
        right_layout.addWidget(self.dynamics_canvas)

        # Controls below plots
        controls_layout = QVBoxLayout()
        # Z-position slider
        slider_layout = QHBoxLayout()
        self.z_slider_label = QLabel("Z: 0.00 m")
        slider_layout.addWidget(self.z_slider_label)
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0) # Will set maximum later based on x_axis length
        self.z_slider.setValue(0)
        self.z_slider.setTickInterval(1)
        self.z_slider.setSingleStep(1)
        self.z_slider.valueChanged.connect(self.update_plots_from_slider)
        slider_layout.addWidget(self.z_slider)
        controls_layout.addLayout(slider_layout)

        # Go to Z, Save, and Twiss Navigation Buttons
        bottom_controls_layout = QHBoxLayout()

        # Go to Z
        go_to_z_layout = QHBoxLayout()
        go_to_z_layout.addWidget(QLabel("Go to Z (m):"))
        self.go_to_z_input = QLineEdit("0.00")
        self.go_to_z_input.setFixedWidth(80)
        self.go_to_z_input.returnPressed.connect(self.go_to_z_position) # Connect Enter key
        self.go_to_z_btn = QPushButton("Go")
        self.go_to_z_btn.clicked.connect(self.go_to_z_position)
        go_to_z_layout.addWidget(self.go_to_z_input)
        go_to_z_layout.addWidget(self.go_to_z_btn)
        bottom_controls_layout.addLayout(go_to_z_layout)

        # Save EPS Button
        self.save_eps_button = QPushButton("Save .eps Snapshots")
        self.save_eps_button.clicked.connect(self.save_eps_snapshots)
        bottom_controls_layout.addWidget(self.save_eps_button)

        # Twiss Navigation Buttons
        twiss_nav_layout = QHBoxLayout()
        self.prev_twiss_button = QPushButton("Prev Twiss")
        self.prev_twiss_button.clicked.connect(self.navigate_twiss_data)
        self.next_twiss_button = QPushButton("Next Twiss")
        self.next_twiss_button.clicked.connect(lambda: self.navigate_twiss_data(1)) # Pass direction 1 for next
        twiss_nav_layout.addWidget(self.prev_twiss_button)
        twiss_nav_layout.addWidget(self.next_twiss_button)
        bottom_controls_layout.addLayout(twiss_nav_layout)

        controls_layout.addLayout(bottom_controls_layout)

        # Twiss Parameters / Text Output Area
        self.twiss_output = QTextEdit()
        self.twiss_output.setReadOnly(True)
        self.twiss_output.setFixedHeight(80) # Adjust height as needed
        controls_layout.addWidget(self.twiss_output)

        right_layout.addLayout(controls_layout)
        main_layout.addWidget(right_panel)

        self.update_plot_initial() # Draw empty plots at startup

    def update_plot_initial(self):
        # Clears and initializes both figures
        self.phase_space_figure.clear()
        ax1 = self.phase_space_figure.add_subplot(221)
        ax2 = self.phase_space_figure.add_subplot(222)
        ax3 = self.phase_space_figure.add_subplot(223)
        ax4 = self.phase_space_figure.add_subplot(224)
        ax1.set_title("x-x'")
        ax2.set_title("y-y'")
        ax3.set_title("x-y")
        ax4.set_title("z-z'")
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel("Position")
            ax.set_ylabel("Phase")
            ax.grid(True)
        self.phase_space_figure.suptitle("Phase Space Plots (No Data)")
        self.phase_space_figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.phase_space_canvas.draw()

        self.dynamics_figure.clear()
        ax5 = self.dynamics_figure.add_subplot(111)
        ax5.set_xlabel("Distance from start of beam (m)")
        ax5.set_ylabel("Envelope E (mm) / Dispersion D (mm)")
        ax5.set_title("Beam Dynamics (No Data)")
        ax5.grid(True)
        self.dynamics_figure.tight_layout()
        self.dynamics_canvas.draw()

        self.twiss_output.setText("Run a simulation to see results.")

    def parse_beamline_definition(self, text_def):
        """
        Parses the text input for beamline elements and returns a list of objects.
        Example format: "D(1)\nQPF(0.5,0.1)\nD(2)"
        """
        segments = []
        lines = text_def.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                # Basic parsing for D(L), QPF(L,G), QPD(L,G), Dipole(L,A), DipoleWedge(L,A)
                if line.startswith("D("):
                    length = float(line[2:-1])
                    segments.append(DummyDrift(length)) # Replace with your Drift()
                elif line.startswith("QPF("):
                    parts = line[4:-1].split(',')
                    length = float(parts[0])
                    gradient = float(parts[1])
                    segments.append(DummyQPF(length, gradient)) # Replace with your QPF()
                elif line.startswith("QPD("):
                    parts = line[4:-1].split(',')
                    length = float(parts[0])
                    gradient = float(parts[1])
                    segments.append(DummyQPD(length, gradient)) # Replace with your QPD()
                elif line.startswith("Dipole("):
                    parts = line[7:-1].split(',')
                    length = float(parts[0])
                    angle = float(parts[1])
                    segments.append(DummyDipole(length, angle)) # Replace with your Dipole()
                elif line.startswith("DipoleWedge("):
                    parts = line[12:-1].split(',')
                    length = float(parts[0])
                    angle = float(parts[1])
                    segments.append(DummyDipoleWedge(length, angle)) # Replace with your DipoleWedge()
                else:
                    print(f"Warning: Unknown element type: {line}")
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                self.twiss_output.setText(f"Error parsing beamline: {e}\nCheck format like D(1), QPF(0.5,0.1)")
                return None
        return segments


    def _checkMinMax(self, matrixVariables, maxval, minval):
        initialx = matrixVariables[:, 0]
        initialxphase = matrixVariables[:, 1]
        initialy = matrixVariables[:, 2]
        initialyphase = matrixVariables[:, 3]
        initialz = matrixVariables[:, 4]
        initialzphase = matrixVariables[:, 5]
        initialList = [initialx, initialxphase, initialy, initialyphase, initialz, initialzphase]
        for i in range(len(initialList)):
            maximum = np.max(initialList[i])
            if maximum > maxval[i]:
                maxval[i] = maximum
            minimum = np.min(initialList[i])
            if minimum < minval[i]:
                minval[i] = minimum
        return maxval, minval

    def _setEqualAxisScaling(self, maxVals, minVals):
        # Assuming x, x', y, y' are in indices 0,1,2,3 respectively
        # x and y positions
        max_xy = max(maxVals[0], maxVals[2])
        min_xy = min(minVals[0], minVals[2])
        maxVals[0] = max_xy
        maxVals[2] = max_xy
        minVals[0] = min_xy
        minVals[2] = min_xy

        # x' and y' phases
        max_xpyp = max(maxVals[1], maxVals[3])
        min_xpyp = min(minVals[1], minVals[3])
        maxVals[1] = max_xpyp
        maxVals[3] = max_xpyp
        minVals[1] = min_xpyp
        minVals[3] = min_xpyp

    def _getClosestZ(self, plot6dValues, val):
        z_values = np.array(list(plot6dValues.keys()))
        closest_z_idx = np.argmin(np.abs(z_values - val))
        closest_z = z_values[closest_z_idx]
        matrix = plot6dValues[closest_z]
        return closest_z, matrix, closest_z_idx


    def _run_simulation_backend(self):
        """
        Executes the beamline simulation and aggregates data.
        This is essentially the core logic from your plotBeamPositionTransform.
        """
        try:
            # 1. Get input values
            num_particles = int(self.num_particles_input.text())
            particle_type = self.particle_type_combo.currentText()
            kinetic_energy = float(self.kinetic_energy_input.text())

            mean_vals = np.zeros(6)
            std_dev_vals = np.array([
                float(self.std_x.text()), float(self.std_xp.text()),
                float(self.std_y.text()), float(self.std_yp.text()),
                float(self.std_z.text()), float(self.std_zp.text())
            ])

            interval = float(self.interval_input.text())
            define_lim = self.define_lim_checkbox.currentText() == "True"
            match_scaling = self.match_scaling_checkbox.currentText() == "True"
            show_indice = self.show_indice_checkbox.currentText() == "True"
            scatter_plot = self.plot_style_combo.currentText() == "Scatter (Individual Particles)"

            self.beamSegments = self.parse_beamline_definition(self.beamline_def_input.toPlainText())
            if self.beamSegments is None: # Parsing failed
                return False, "Beamline parsing error."

            # Update beam properties (assuming your beam object has such a method)
            # self.ebeam.changeBeamType(particle_type, kinetic_energy)
            # For dummy:
            self.ebeam.E = kinetic_energy

            self.matrixVariables = self.ebeam.gen_6d_gaussian(mean_vals, std_dev_vals, num_particles)

            # Initialize data containers
            self.plot6dValues = {}
            self.x_axis = [0]
            self.maxVals = [-np.inf] * 6 # Initialize with -inf for max
            self.minVals = [np.inf] * 6 # Initialize with +inf for min

            # Calculate initial Twiss parameters and store
            result = self.beam_analyzer.getXYZ(self.matrixVariables)
            initial_twiss_df = result[3] # Assuming getXYZ returns (std1, std6, final_particles, twiss_df)

            # Initialize twiss_aggregated_df with correct structure and first values
            self.twiss_aggregated_df = pd.DataFrame(
                {col: {idx: [] for idx in initial_twiss_df.index} for col in initial_twiss_df.columns}
            )
            for col in initial_twiss_df.columns:
                for idx in initial_twiss_df.index:
                    self.twiss_aggregated_df.loc[idx, col].append(initial_twiss_df.loc[idx, col])

            self.plot6dValues[0] = result # Store initial state

            if define_lim:
                self.maxVals, self.minVals = self._checkMinMax(self.matrixVariables, self.maxVals, self.minVals)

            total_intervals = sum(int(segment.length // interval) + 1 for segment in self.beamSegments)

            # Simulation loop with tqdm (for console progress, not critical for GUI)
            self.twiss_output.setText("Simulating...")
            QApplication.processEvents() # Update GUI
            with tqdm(total=total_intervals, desc="Simulating Beamline",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                current_z = 0.0
                for i, segment in enumerate(self.beamSegments):
                    intTrack = segment.length
                    while intTrack >= interval:
                        self.matrixVariables = np.array(segment.useMatrice(self.matrixVariables, length=interval))
                        current_z += interval
                        rounded_z = round(current_z, self.DEFAULTINTERVALROUND)
                        self.x_axis.append(rounded_z)

                        if define_lim:
                            self.maxVals, self.minVals = self._checkMinMax(self.matrixVariables, self.maxVals, self.minVals)

                        result = self.beam_analyzer.getXYZ(self.matrixVariables)
                        current_twiss = result[3]
                        self.plot6dValues[rounded_z] = result

                        for col in current_twiss.columns:
                            for idx in current_twiss.index:
                                self.twiss_aggregated_df.loc[idx, col].append(current_twiss.loc[idx, col])

                        pbar.update(1)
                        intTrack -= interval

                    if intTrack > 0: # Remaining length
                        self.matrixVariables = np.array(segment.useMatrice(self.matrixVariables, length=intTrack))
                        current_z += intTrack
                        rounded_z = round(current_z, self.DEFAULTINTERVALROUND)
                        self.x_axis.append(rounded_z)

                        if define_lim:
                            self.maxVals, self.minVals = self._checkMinMax(self.matrixVariables, self.maxVals, self.minVals)

                        result = self.beam_analyzer.getXYZ(self.matrixVariables)
                        current_twiss = result[3]
                        self.plot6dValues[rounded_z] = result

                        for col in current_twiss.columns:
                            for idx in current_twiss.index:
                                self.twiss_aggregated_df.loc[idx, col].append(current_twiss.loc[idx, col])

                        pbar.update(1)

            if match_scaling and define_lim:
                self._setEqualAxisScaling(self.maxVals, self.minVals)

            return True, "Simulation complete."

        except ValueError as ve:
            return False, f"Input error: {ve}. Please check numeric inputs."
        except Exception as e:
            return False, f"An unexpected error occurred during simulation: {e}"

    def run_simulation_and_update_ui(self):
        success, message = self._run_simulation_backend()
        self.twiss_output.setText(message)
        if success:
            self.update_all_plots() # Update plots with fresh simulation data
            self.configure_slider()
            self.go_to_z_input.setText(f"{self.x_axis[0]:.2f}") # Reset to start Z

    def configure_slider(self):
        if self.x_axis:
            self.z_slider.setMinimum(0)
            self.z_slider.setMaximum(len(self.x_axis) - 1)
            self.z_slider.setValue(0) # Reset to start of beamline
            self.current_z_index = 0
            self.z_slider_label.setText(f"Z: {self.x_axis[0]:.2f} m")
        else:
            self.z_slider.setMinimum(0)
            self.z_slider.setMaximum(0)
            self.z_slider.setValue(0)
            self.z_slider_label.setText("Z: 0.00 m")

    def update_plots_from_slider(self, index):
        if not self.x_axis or index >= len(self.x_axis):
            return

        self.current_z_index = index
        z_value = self.x_axis[index]
        self.z_slider_label.setText(f"Z: {z_value:.2f} m")

        # Get the 6D data for the current Z
        closest_z, matrix_data, _ = self._getClosestZ(self.plot6dValues, z_value)
        particles_6d = matrix_data[2] # Assuming it's the 3rd element from getXYZ return

        # Update Phase Space Plots
        self.phase_space_figure.clear()
        ax1 = self.phase_space_figure.add_subplot(221)
        ax2 = self.phase_space_figure.add_subplot(222)
        ax3 = self.phase_space_figure.add_subplot(223)
        ax4 = self.phase_space_figure.add_subplot(224)

        define_lim = self.define_lim_checkbox.currentText() == "True"
        scatter_plot = self.plot_style_combo.currentText() == "Scatter (Individual Particles)"
        # Assuming your plotXYZ function takes ax1-4 directly
        self.ebeam.plotXYZ(particles_6d, None, None, None, ax1, ax2, ax3, ax4,
                           self.maxVals, self.minVals, define_lim, {}, scatter=scatter_plot)

        self.phase_space_figure.suptitle(f"Phase Space Plots (Z = {z_value:.2f} m)")
        self.phase_space_figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.phase_space_canvas.draw_idle()

        # Update Twiss Parameters display
        twiss_df = self.twiss_aggregated_df.copy()
        if twiss_df is not None and not twiss_df.empty:
            # Need to get the twiss data at the specific Z-index
            twiss_x_at_z = {key: twiss_df.loc['x', key][index] for key in twiss_df.columns}

            twiss_text = "<h3>Calculated Twiss Parameters (x-plane)</h3>"
            # Ensure keys used here match your twiss_df column names precisely
            if r'$\epsilon$ ($\pi$.mm.mrad)' in twiss_x_at_z:
                eptext = r'$\epsilon$ ($\pi$.mm.mrad)'
                twiss_text += f"$\epsilon$: {twiss_x_at_z[eptext]:.3f} ($\pi$.mm.mrad)<br>"
            if r'$\alpha$' in twiss_x_at_z:
                alphatext = r'$\alpha$'
                twiss_text += f"$\alpha$: {twiss_x_at_z[alphatext]:.3f}<br>"
            if r'$\beta$ (m)' in twiss_x_at_z:
                betatext = r'$\beta$ (m)'
                twiss_text += f"$\beta$: {twiss_x_at_z[betatext]:.3f} (m)"
            self.twiss_output.setHtml(twiss_text)
        else:
            self.twiss_output.setText("Twiss parameters not available.")

        # Update dynamics plot to show the current Z position marker
        self._plot_dynamics(update_marker=True)


    def update_all_plots(self):
        # This function is called after a full simulation to draw all plots for the first time
        if not self.x_axis or not self.twiss_aggregated_df:
            self.update_plot_initial() # Show empty plots if no data
            return

        # Phase Space Plot (initial Z)
        self.update_plots_from_slider(0) # Update phase space and twiss output for Z=0

        # Dynamics/Envelope Plot
        self._plot_dynamics()


    def _plot_dynamics(self, update_marker=False):
        # Plot and configure line graph data
        self.dynamics_figure.clear()
        ax5 = self.dynamics_figure.add_subplot(111)
        colors = ['dodgerblue', 'crimson'] # For x and y envelopes
        disp_colors = ['green', 'orange'] # For x and y dispersion

        # Get current twiss data name based on index
        twiss_column_names = list(self.twiss_aggregated_df.columns)
        if not twiss_column_names:
            ax5.set_title("Beam Dynamics (No Twiss Data)")
            self.dynamics_figure.tight_layout()
            self.dynamics_canvas.draw_idle()
            return

        current_twiss_name_idx = self.current_twiss_index % len(twiss_column_names)
        current_twiss_name = twiss_column_names[current_twiss_name_idx]

        # Plot envelope and current twiss parameter
        envelope_lines = []
        twiss_lines = []

        for i, axis in enumerate(['x', 'y']):
            # Envelope (E) - uses Emittance and Beta
            if r'$\epsilon$ ($\pi$.mm.mrad)' in self.twiss_aggregated_df.columns and r'$\beta$ (m)' in self.twiss_aggregated_df.columns:
                emittance = (10 ** -6) * np.array(self.twiss_aggregated_df.loc[axis, r'$\epsilon$ ($\pi$.mm.mrad)'])
                beta = np.array(self.twiss_aggregated_df.loc[axis, r'$\beta$ (m)'])
                envelope = (10 ** 3) * np.sqrt(emittance * beta)
                line, = ax5.plot(self.x_axis, envelope, color=colors[i], linestyle='-',
                                label=f'$E_{axis}$ (mm)')
                envelope_lines.append(line)

            # Current Selected Twiss Parameter (e.g., Dispersion, Alpha, Beta)
            if current_twiss_name in self.twiss_aggregated_df.columns:
                twiss_values = np.array(self.twiss_aggregated_df.loc[axis, current_twiss_name])
                ax6 = ax5.twinx() # Create twin axis for Twiss parameters
                line, = ax6.plot(self.x_axis, twiss_values, color=disp_colors[i], linestyle='--',
                                label=f'${current_twiss_name.split(" ")[0]}_{axis}$ ({current_twiss_name.split(" ")[1] if len(current_twiss_name.split(" ")) > 1 else ""})')
                twiss_lines.append(line)
                ax6.set_ylabel(f'{current_twiss_name}')
                ax6.legend(loc='upper right') # Legend for twin axis

        # Set common labels for ax5
        ax5.set_xlabel(r"Distance from start of beam (m)")
        ax5.set_ylabel(r"Envelope $E$ (mm)")
        ax5.legend(loc='upper left') # Legend for main axis

        ax5.set_xticks(self.x_axis)
        ax5.set_xlim(0, self.x_axis[-1] if self.x_axis else 0)

        # Auto space x tick labels for readability
        if self.show_indice_checkbox.currentText() == "True": # Re-using this option for spacing
            totalLen = self.x_axis[-1] if self.x_axis else 1
            lastTick = self.x_axis[0] if self.x_axis else 0
            xTickLab = [lastTick]
            for tick in self.x_axis[1:]:
                if (tick - lastTick) / totalLen > self.DEFAULTSPACINGPERCENTAGE:
                    xTickLab.append(tick)
                    lastTick = tick
                else:
                    xTickLab.append("")
            # Format non-empty labels to self.DEFAULTINTERVALROUND decimal places
            xTicks_disp = [f"{x:.{self.DEFAULTINTERVALROUND}f}" if isinstance(x, (float,int)) else "" for x in xTickLab]
            ax5.set_xticklabels(xTicks_disp, rotation=45, ha='right')
        else:
            # Format all labels to self.DEFAULTINTERVALROUND decimal places
            xTicks_disp = [f"{x:.{self.DEFAULTINTERVALROUND}f}" for x in self.x_axis]
            ax5.set_xticklabels(xTicks_disp, rotation=45, ha='right')

        ax5.tick_params(labelsize=9)
        ax5.set_title("Beam Dynamics Simulation")


        # Create visual representation of beamline segments
        ymin, ymax = ax5.get_ylim()
        # Ensure some space at the bottom for segment visualization
        ax5.set_ylim(ymin - (ymax * 0.05), ymax)
        ymin, ymax = ax5.get_ylim() # Re-get after setting new ymin
        blockstart = 0
        moveUp = True # For alternating text position
        for i, seg in enumerate(self.beamSegments):
            # Calculate segment width
            seg_width = seg.length
            # Create a rectangle patch
            rectangle = patches.Rectangle((blockstart, ymin), seg_width, ymax * 0.05,
                                          linewidth=1, edgecolor=seg.color, facecolor=seg.color, alpha=0.7)
            ax5.add_patch(rectangle)
            if self.show_indice_checkbox.currentText() == "True":
                rec_center_x = blockstart + seg_width / 2
                rec_text_y = ymin + (ymax * 0.05) / 2 # Center vertically within the box
                # Alternate text position slightly up/down for readability if overlapping
                if moveUp:
                    ax5.text(rec_center_x, rec_text_y + ymax * 0.01, str(i), size='small',
                             ha='center', va='center', color='black')
                else:
                    ax5.text(rec_center_x, rec_text_y - ymax * 0.01, str(i), size='small',
                             ha='center', va='center', color='black')
                moveUp = not moveUp
            blockstart += seg_width

        # Add a vertical line marker at the current Z position
        if update_marker and self.x_axis and self.current_z_index < len(self.x_axis):
            current_z_val = self.x_axis[self.current_z_index]
            ax5.axvline(x=current_z_val, color='grey', linestyle=':', linewidth=2, label=f'Current Z: {current_z_val:.2f}m')
            ax5.legend(loc='lower left')


        self.dynamics_figure.tight_layout()
        self.dynamics_canvas.draw_idle()


    def navigate_twiss_data(self, direction=0): # 0 for prev, 1 for next
        if not self.twiss_aggregated_df or self.twiss_aggregated_df.empty:
            return

        twiss_column_names = list(self.twiss_aggregated_df.columns)
        if not twiss_column_names:
            return

        if direction == 1: # Next
            self.current_twiss_index = (self.current_twiss_index + 1) % len(twiss_column_names)
        else: # Previous
            self.current_twiss_index = (self.current_twiss_index - 1 + len(twiss_column_names)) % len(twiss_column_names)

        self._plot_dynamics() # Redraw dynamics plot with new twiss data

    def go_to_z_position(self):
        try:
            target_z = float(self.go_to_z_input.text())
            if not self.x_axis:
                self.twiss_output.setText("No simulation data to go to Z.")
                return

            # Find the index of the closest Z value in self.x_axis
            z_values = np.array(self.x_axis)
            closest_z_idx = np.argmin(np.abs(z_values - target_z))

            self.z_slider.setValue(closest_z_idx) # This will trigger update_plots_from_slider
            self.twiss_output.setText(f"Moved to Z = {self.x_axis[closest_z_idx]:.2f} m")

        except ValueError:
            self.twiss_output.setText("Invalid Z value. Please enter a number.")
        except Exception as e:
            self.twiss_output.setText(f"Error navigating to Z: {e}")

    def _save_eps_single_plot(self, fig, filename, current_val):
        # This now assumes fig is a Matplotlib figure and the filename is provided.
        fig.savefig(filename, format='eps', bbox_inches='tight')
        print(f"Saved {filename}") # For console feedback

    def save_eps_snapshots(self):
        # Save current phase space
        self._save_eps_single_plot(self.phase_space_figure,
                                   f"phase_space_z_{self.x_axis[self.current_z_index]:.2f}.eps",
                                   self.x_axis[self.current_z_index])

        # Save current dynamics plot
        self._save_eps_single_plot(self.dynamics_figure,
                                   f"dynamics_plot_z_{self.x_axis[self.current_z_index]:.2f}.eps",
                                   self.x_axis[self.current_z_index])

        self.twiss_output.setText(f"EPS snapshots saved for Z = {self.x_axis[self.current_z_index]:.2f} m")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # --- Apply qt_material stylesheet here ---
    apply_stylesheet(app, theme='dark_blue.xml') # Choose your preferred theme

    window = BeamlineSimulatorUI()
    window.show()
    sys.exit(app.exec_())