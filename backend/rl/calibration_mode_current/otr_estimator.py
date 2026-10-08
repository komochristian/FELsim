import numpy as np
from scipy.optimize import minimize
from calibration_reference import (
    width_negative_log_likelihood, 
    read_noise_negative_log_likelihood
)

class Estimator:
    """
    Maximum Likelihood Estimator for the Calibration SMDP Environment.
    Maintains the 'belief state' of the 14 hidden machine parameters.
    """
    def __init__(self):
        # The 14 Unknowns[cite: 1]:
        # Indices 0-5: 6x Quad offset (b0)
        # Indices 6-11: 6x Quad RMS noise (sigma_k)
        # Index 12: 1x Camera read noise (sigma_read)
        # Index 13: 1x Lens-to-sensor distance (z2)
        self.num_params = 14
        self.params = np.zeros(self.num_params)
        
        # Initialize strictly positive physical priors
        self.params[6:12] = 0.01   # Starting guess for sigma_k (m^-2)
        self.params[12] = 3.0      # Starting guess for sigma_read (electrons)
        self.params[13] = 0.825    # Starting guess for nominal z2 distance (m)[cite: 1]
        
        # Initialize the 14x14 symmetric covariance matrix[cite: 1]
        # Represents the estimator's uncertainty; starts as a diagonal prior
        self.covariance = np.eye(self.num_params) * 1e-4

    def update(self, measurements):
        """
        Ingests a list of Measurements, evaluates bounded likelihoods, 
        and updates the parameter estimates and covariance via optimization.
        """
        # Filter for fits that passed quality gates (SNR, margins, chi2)[cite: 1]
        valid_m = [m for m in measurements if m.valid]
        if not valid_m:
            return  # Skip update if no valid measurements were acquired

        # Define bounds to prevent physically impossible parameter spaces[cite: 1]
        # b0: unconstrained; sigma_k > 0; sigma_read > 0; z2 loosely bounded around 0.825
        bounds = (
            [(None, None)] * 6 +     # b0 
            [(1e-6, None)] * 6 +     # sigma_k 
            [(1e-2, None)] +         # sigma_read
            [(0.5, 1.2)]             # z2
        )

        def objective(candidate_params):
            """
            Combined negative log-likelihood objective function.
            """
            # 1. Read Noise Likelihood: derived from empty dark pixels[cite: 1]
            sigma_read = candidate_params[12]
            nll_read = read_noise_negative_log_likelihood(valid_m, sigma_read)
            
            # 2. Beam Width Likelihood: evaluates predicted vs measured spatial widths[cite: 1]
            # Closure isolates candidate parameters for the forward model
            def predict_closure(m):
                return self._forward_physics_model(candidate_params, m)
                
            nll_width = width_negative_log_likelihood(valid_m, predict_closure)
            
            # Total NLL; must balance residuals against predicted variance[cite: 1]
            return nll_width + nll_read

        # Execute bounded optimization
        res = minimize(
            objective, 
            self.params, 
            bounds=bounds, 
            method="L-BFGS-B",
            options={"disp": False}
        )

        # Update belief state upon successful convergence
        if res.success:
            self.params = res.x
            # Update covariance using the inverse Hessian matrix approximation
            if hasattr(res, "hess_inv"):
                self.covariance = res.hess_inv.todense()

    def _forward_physics_model(self, candidate_params, measurement):
        """
        Physics coupling to FELSim and OTR optics.
        Takes candidate parameters and the current machine settings (currents),
        and tracks a simulation pulse to predict the beam distribution at the screen.
        
        Returns:
            mean_xy: np.ndarray shape (2,) - predicted [sigma_x, sigma_y]
            cov_xy: np.ndarray shape (2,2) - predicted transverse covariance (C_total)
        """
        # Note: Implement FELSim finite-difference sensitivities or tracking here[cite: 1]
        # Example dummy return to satisfy the prototype architecture:
        predicted_mean_xy = np.array([0.001, 0.001])
        predicted_cov_xy = np.eye(2) * 1e-6
        return predicted_mean_xy, predicted_cov_xy

    def get_current_belief(self):
        """
        Formats the estimator's internal state into the flat observation vector
        required by the Gymnasium RL agent.
        """
        # Extract the upper triangle of the 14x14 symmetric covariance matrix
        # This reduces 196 elements to exactly 105 independent entries[cite: 1]
        upper_triangle_indices = np.triu_indices(self.num_params)
        flat_covariance = self.covariance[upper_triangle_indices]
        
        # Returns shape (14,) and shape (105,)
        return self.params, flat_covariance