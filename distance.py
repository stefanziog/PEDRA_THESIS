import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def calculate_trajectory_distance(file_path, interpolation_factor=5):
    # Load the 3D trajectory data
    df = pd.read_csv(file_path)
    
    # Compute the total traveled distance using Euclidean distance formula
    distances = np.sqrt(
        np.diff(df["x"])**2 +
        np.diff(df["y"])**2 +
        np.diff(df["z"])**2
    )
    total_distance = np.sum(distances)
    
    # Compute the straight-line distance from start to end
    start_point = df.iloc[0][["x", "y", "z"]].values
    end_point = df.iloc[-1][["x", "y", "z"]].values
    straight_line_distance = np.linalg.norm(end_point - start_point)
    
    # Perform interpolation
    num_interpolated_points = len(df) * interpolation_factor
    t_original = np.linspace(0, 1, len(df))
    t_interpolated = np.linspace(0, 1, num_interpolated_points)
    
    # Interpolation functions
    interp_x = interp1d(t_original, df["x"], kind="cubic")
    interp_y = interp1d(t_original, df["y"], kind="cubic")
    interp_z = interp1d(t_original, df["z"], kind="cubic")
    
    # Generate interpolated points
    x_interp = interp_x(t_interpolated)
    y_interp = interp_y(t_interpolated)
    z_interp = interp_z(t_interpolated)
    
    # Recalculate total distance with interpolated trajectory
    distances_interpolated = np.sqrt(np.diff(x_interp)**2 + np.diff(y_interp)**2 + np.diff(z_interp)**2)
    total_distance_interpolated = np.sum(distances_interpolated)
    
    # Print results
    print(f"Results for: {file_path}")
    print(f"Total traveled distance: {total_distance:.2f} meters")
    print(f"Straight-line distance: {straight_line_distance:.2f} meters")
    print(f"Interpolated total distance: {total_distance_interpolated:.2f} meters\n")
    
    return total_distance, straight_line_distance, total_distance_interpolated

# Define file paths for each algorithm
file_paths = {
    "DeepReinforce": r"C:\Users\Autostudents\Pictures\DeepReinforce_Results\3d_trajectory.csv",
    "DeepQlearning": r"C:\Users\Autostudents\Pictures\DeepQlearning_Results\3d_trajectory.csv",
    "DeepPPO": r"C:\Users\Autostudents\Pictures\DeepPPO_Results\3d_trajectory.csv"
}

# Run calculations for each file
for name, path in file_paths.items():
    print(f"Processing {name}...")
    calculate_trajectory_distance(path)
