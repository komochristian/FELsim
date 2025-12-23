# test_cosy_1.py

#import sys
#import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosyParticleSimulator import COSYParticleSimulator
from pprint import pprint

excel_path = '../../beam_excel/Beamline_elements.xlsx'

config = {
    "simulation": {"KE": 45, "order": 2, "dimensions": 2},
    "variable_mapping": {1: {"current": "I1"}, 9: {"current": "I"}}
}

simulator = COSYParticleSimulator(excel_path, use_enge_coeffs=False, use_mge_for_dipoles=False, config_dict=config)
checkpoint_elements = [1, 10, 20, 30, 40, 50]
simulator.enable_particle_tracking(checkpoint_elements=checkpoint_elements)
# Parse the beamline from Excel
beamline = simulator.parse_beamline()  # Returns list of dicts
qpf_indices = simulator.find_elements('QPF')

xVar = {
    1: {"current": "I1"},
    9: {"current": "I2"}
}

simulator.update_config(config_dict={
    "simulation": {"KE": 37.5, "order": 3},
    "variable_mapping": {1: {"current": "I1"}, 3: {"current": "I2"}, 9: {"current": "I3"}}
})

simulator.set_optimization_initial_point({
    "I1": {"bounds": (0, 10), "start": 2},
    "I2": {"bounds": (0, 10), "start": 3}
})

simulator.set_optimization_objectives({
    86: [{"measure": ["y", "alpha"], "goal": 0, "weight": 1},
         {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5}],
    "optimizer_settings": {"eps": 1e-10, "Nmax": 0, "Nalgorithm": 3}
})

# simulator.apply_variable_mapping(xVar)

# Print the parsed beamline for verification
# simulator.print_beamline()
# Generate the COSY INFINITY input file and save it to 'results' directory
input_file_path = simulator.generate_input(output_dir='results')
# Read and print the generated .fox file content for testing
# with open(input_file_path, 'r') as f:
#    fox_content = f.read()
# print("\nGenerated COSY INFINITY Input File Content:")
# print("-" * 80)
# print(fox_content)
#print("-" * 80)

# Optionally run the simulation (uncomment to test full execution)
simulator.run_simulation(output_dir='results')
results = simulator.analyze_results()
transfer_matrix = results.read_linear_transfer_map()
print("First order transfer map:\n", transfer_matrix)
json_data = results.read_json_results()

particles = simulator.generate_6d_gaussian(
    std_dev=[0.8, 0.1, 0.8, 0.1, 2.0, 0.5],
    num_particles=10000
)
print(particles)
cosy_particles = simulator.transform_to_cosy_coordinates(particles)
simulator.write_particle_file(cosy_particles, filename='fort.200')
cosy_particles_readback = simulator.read_particle_file('fort.200', output_dir='results')
particles_back = simulator.transform_from_cosy_coordinates(cosy_particles_readback)
print(particles_back)

simulator.validate_coordinate_transformation()

# Read specific checkpoints using integer indices
particles_at_1 = simulator.read_particle_file(1, output_dir='results')  # fort.10001
particles_at_10 = simulator.read_particle_file(10, output_dir='results')  # fort.10010
particles_at_20 = simulator.read_particle_file(20, output_dir='results')  # fort.10020

# Or read all at once
checkpoints = simulator.read_checkpoints(checkpoint_elements, output_dir='results')

setting = simulator.get_full_config()
# pprint(setting, indent=2, width=100, sort_dicts=False)

# Convert the complex values
beta_x_complex = results.convert_complex_pair(json_data['twiss']['beta_x'])
beta_y_complex = results.convert_complex_pair(json_data['twiss']['beta_y'])
# print("System length [m]:", json_data['spos'])
# print("beta_x [m]:", beta_x_complex)
#print("beta_y [m]:", beta_y_complex)
