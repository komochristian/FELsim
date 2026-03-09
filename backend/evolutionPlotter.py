# evolutionPlotter.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from typing import Optional, Dict

from beamEvolution import BeamEvolution
from ebeam import beam as ebeam_class

from loggingConfig import get_logger_with_fallback


class EvolutionPlotter:
    """
    Matplotlib-based plotter for BeamEvolution data with interactive navigation.

    Handles non-uniform s-spacing transparently. Supports both local (adaptive)
    and global (fixed) axis scaling modes for phase space visualization.
    """

    ELEMENT_COLORS = {
        'DRIFT': 'white',
        'driftLattice': 'white',
        'QPF': 'cornflowerblue',
        'QUAD_F': 'cornflowerblue',
        'qpfLattice': 'cornflowerblue',
        'QPD': 'lightcoral',
        'QUAD_D': 'lightcoral',
        'qpdLattice': 'lightcoral',
        'DPH': 'forestgreen',
        'DIPOLE': 'forestgreen',
        'dipole': 'forestgreen',
        'DIPOLE_CONSOLIDATED': 'forestgreen',
        'DPW': 'lightgreen',
        'DIPOLE_WEDGE': 'lightgreen',
        'dipole_wedge': 'lightgreen'
    }

    def __init__(self, figsize=(12, 10), axis_mode='local', debug=None):
        """
        Parameters
        ----------
        axis_mode : {'local', 'global'}
            'local' adapts axes to current s-position (default),
            'global' maintains consistent axes across beamline
        """
        self.figsize = figsize
        self.axis_mode = axis_mode
        self._global_extents = None
        self._ebeam = ebeam_class()

        self.envelope_colors = {'x': 'dodgerblue', 'y': 'crimson'}
        self.scatter_size = 30
        self.scatter_alpha = 0.7

        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

    def _compute_global_extents(self, evolution: BeamEvolution):
        """Compute axis limits encompassing entire beamline evolution."""
        maxVals = np.zeros(6)
        minVals = np.zeros(6)

        for s, particles in evolution.particles.items():
            for i in range(6):
                maxVals[i] = max(maxVals[i], np.max(particles[:, i]))
                minVals[i] = min(minVals[i], np.min(particles[:, i]))

        margin = 0.1
        ranges = maxVals - minVals
        maxVals += margin * ranges
        minVals -= margin * ranges

        return maxVals, minVals

    def plot(self,
             evolution: BeamEvolution,
             show_phase_space: bool = True,
             show_envelope: bool = True,
             show_schematic: bool = True,
             interactive: bool = True,
             shape: Dict = None,
             scatter: bool = False,
             save_path: Optional[str] = None,
             envelope_ylim: Optional[tuple] = None) -> plt.Figure:
        """
        Plot beam evolution through beamline.

        Parameters
        ----------
        evolution : BeamEvolution
            Simulation results containing particle distributions and Twiss parameters
        show_phase_space : bool
            Display x-x', y-y', z-z', x-y phase space plots
        show_envelope : bool
            Display envelope evolution along beamline
        show_schematic : bool
            Overlay beamline element schematic on envelope plot
        interactive : bool
            Enable slider for phase space navigation along s
        shape : dict, optional
            Aperture geometry for acceptance calculation (e.g., {'shape': 'circle', 'radius': 5})
        scatter : bool
            Use scatter plot with density coloring instead of hexbin
        save_path : str, optional
            Export figure to file
        envelope_ylim : tuple, optional
            Y-axis limits (ymin, ymax) for envelope plot in mm

        Returns
        -------
        fig : matplotlib.Figure
        """
        if not evolution.s_positions:
            raise ValueError("BeamEvolution contains no particle data")

        fig = plt.figure(figsize=self.figsize)

        if self.axis_mode == 'global':
            self._global_extents = self._compute_global_extents(evolution)

        if show_phase_space and show_envelope:
            gs = gridspec.GridSpec(
                3, 2,
                height_ratios=[0.8, 0.8, 1],
                hspace=0.4,
                wspace=0.3
            )
            ax_phase = [
                plt.subplot(gs[0, 0]),
                plt.subplot(gs[0, 1]),
                plt.subplot(gs[1, 0]),
                plt.subplot(gs[1, 1])
            ]
            ax_envelope = plt.subplot(gs[2, :])
        elif show_envelope:
            gs = gridspec.GridSpec(1, 1)
            ax_phase = None
            ax_envelope = plt.subplot(gs[0, 0])
        else:
            gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)
            ax_phase = [plt.subplot(gs[i, j]) for i in range(2) for j in range(2)]
            ax_envelope = None

        twiss_df = evolution.get_twiss_evolution()

        if show_envelope and ax_envelope is not None:
            self._plot_envelope(ax_envelope, twiss_df, evolution.elements,
                                show_schematic, envelope_ylim)

        initial_s = evolution.s_positions[0]
        if show_phase_space and ax_phase is not None and initial_s in evolution.particles:
            self._plot_phase_space(
                ax_phase,
                evolution.particles[initial_s],
                shape=shape or {},
                scatter=scatter
            )

        slider = None
        if interactive and show_phase_space and ax_phase is not None:
            slider = self._setup_slider(fig, evolution, ax_phase, shape, scatter)

        title = f"{evolution.simulator_name} Simulation"
        title += f" ({len(evolution.s_positions)} samples, {evolution.num_particles} particles)"
        plt.suptitle(title)

        # Use subplots_adjust to avoid slider/layout conflicts
        slider_margin = 0.06 if interactive else 0.02
        fig.subplots_adjust(
            left=0.10,
            right=0.90,
            top=0.93,
            bottom=slider_margin + 0.05,
            hspace=0.45,
            wspace=0.30
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        return fig

    def _plot_envelope(self, ax, twiss_df, elements, show_schematic,
                        envelope_ylim=None):
        """Plot envelope evolution with optional beamline schematic overlay."""
        s = twiss_df['s'].values
        env_x = twiss_df['envelope_x'].values
        env_y = twiss_df['envelope_y'].values

        ax.plot(s, env_x, color=self.envelope_colors['x'],
                linestyle='-', label=r'$E_x$ (mm)', linewidth=1.5)
        ax.plot(s, env_y, color=self.envelope_colors['y'],
                linestyle='-', label=r'$E_y$ (mm)', linewidth=1.5)

        ax.scatter(s, env_x, color=self.envelope_colors['x'], s=15, zorder=3)
        ax.scatter(s, env_y, color=self.envelope_colors['y'], s=15, zorder=3)

        ax.set_xlabel(r'Distance from beamline origin, $s$ (m)')
        ax.set_ylabel('Envelope (mm)')
        ax.legend(loc='upper left')
        ax.set_xlim(0, s[-1] if len(s) > 0 else 1)
        ax.grid(True, alpha=0.3)

        # Apply user-specified y-limits if provided
        if envelope_ylim is not None:
            ax.set_ylim(envelope_ylim)

        if show_schematic and elements:
            ymin, ymax = ax.get_ylim()
            schematic_height = (ymax - ymin) * 0.05
            ax.set_ylim(ymin - schematic_height * 1.2, ymax)
            ymin, _ = ax.get_ylim()

            for elem in elements:
                color = self.ELEMENT_COLORS.get(elem.element_type, 'gray')
                rect = patches.Rectangle(
                    (elem.s_start, ymin),
                    elem.length,
                    schematic_height,
                    linewidth=0.5,
                    edgecolor='black' if color == 'white' else color,
                    facecolor=color
                )
                ax.add_patch(rect)

        # Secondary axis for dispersion
        ax2 = ax.twinx()
        if 'dispersion_x' in twiss_df.columns:
            ax2.plot(s, twiss_df['dispersion_x'],
                     color=self.envelope_colors['x'], linestyle='--',
                     label=r'$D_x$ (mm)', alpha=0.7)
            ax2.plot(s, twiss_df['dispersion_y'],
                     color=self.envelope_colors['y'], linestyle='--',
                     label=r'$D_y$ (mm)', alpha=0.7)
            ax2.set_ylabel('Dispersion (mm)')
            ax2.legend(loc='upper right')

    def _plot_phase_space(self, axes, particles, shape, scatter):
        """Update phase space plots with particle distribution at current s."""
        result = self._ebeam.getXYZ(particles)
        std1, std6, dist_6d, twiss = result

        if self.axis_mode == 'global' and self._global_extents is not None:
            maxVals, minVals = self._global_extents
        else:
            # Adaptive scaling based on current particle distribution
            maxVals = [np.max(np.abs(particles[:, i])) * 1.2 for i in range(6)]
            minVals = [-m for m in maxVals]

        self._ebeam.plotXYZ(
            dist_6d, std1, std6, twiss,
            axes[0], axes[1], axes[2], axes[3],
            maxVals, minVals,
            defineLim=True,
            shape=shape,
            scatter=scatter
        )

    def _setup_slider(self, fig, evolution, ax_phase, shape, scatter):
        """Configure interactive slider for s-position navigation."""
        s_values = evolution.s_positions

        slider_ax = plt.axes([0.15, 0.01, 0.7, 0.02])
        slider = Slider(
            slider_ax,
            '',
            s_values[0],
            s_values[-1],
            valinit=s_values[0],
            valstep=s_values
        )

        slider.valtext.set_text(f's = {s_values[0]:.4f} m')

        def update(val):
            # Snap to nearest sampled position
            s = min(s_values, key=lambda x: abs(x - val))

            if s in evolution.particles:
                for ax in ax_phase:
                    ax.clear()
                self._plot_phase_space(ax_phase, evolution.particles[s], shape, scatter)
                slider.valtext.set_text(f's = {s:.4f} m')
                fig.canvas.draw_idle()

        slider.on_changed(update)

        fig.text(0.08, 0.015, 'Position:', fontsize=10, va='center')

        # Prevent garbage collection
        fig._slider = slider

        return slider

    def save_evolution_data(self, evolution: BeamEvolution, filepath: str):
        """Export Twiss evolution to CSV."""
        df = evolution.get_twiss_evolution()
        df.to_csv(filepath, index=False)
        print(f"Saved Twiss evolution to {filepath}")