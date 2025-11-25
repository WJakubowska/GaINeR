import sys
import os
import argparse
import numpy as np
import open3d as o3d
import taichi as ti

# Ensure engine can be imported
sys.path.append(os.path.abspath('taichi_elements'))
from engine.mpm_solver import MPMSolver

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Run MPM simulation on a point cloud.')
parser.add_argument('-i', '--in-file', type=str, required=True, help='Input .ply file with initial points')
parser.add_argument('-o', '--out-dir', type=str, required=True, help='Output folder for .ply sequence')
parser.add_argument('--scale', type=float, default=0.5, help='Scaling factor for the point cloud')
parser.add_argument('--offset', type=float, nargs='+', default=[0.25, 0.4], help='Offset to center the object')
args = parser.parse_args()


class Rescale:
    """Handles normalization of point cloud data to fit simulation bounds."""

    def __init__(self, scale=1.0, offset=[0.0, 0.0]):
        self.min = None
        self.max = None
        self.scale = scale
        self.offset = np.array(offset)

    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)

    def transform(self, x):
        delta = self.max - self.min
        # Avoid division by zero if the cloud is flat on one axis
        delta[delta == 0] = 1.0
        return self.scale * (x - self.min) / delta + self.offset

    def inverse(self, x):
        delta = self.max - self.min
        delta[delta == 0] = 1.0
        return (x - self.offset) / self.scale * delta + self.min


# --- Simulation Configuration ---
ti.init()
os.makedirs(args.out_dir, exist_ok=True)

mpm = MPMSolver(res=(512, 512), E_scale=4)
mpm.add_surface_collider(point=(0.5, 0.25), normal=(0, 1), surface=mpm.surface_slip)

# --- Data Loading and Normalization ---
print(f"Loading input point cloud from: {args.in_file}")
pcd = o3d.io.read_point_cloud(args.in_file)
points_3d = np.asarray(pcd.points)

# Use X and Y axes for the 2D simulation
points_2d = points_3d[:, [0, 1]]

scaler = Rescale(scale=args.scale, offset=args.offset)
scaler.fit(points_2d)
pts_norm = scaler.transform(points_2d)

mpm.add_particles(
    particles=pts_norm,
    material=MPMSolver.material_elastic
)
print(f"Added {len(pts_norm)} particles. Starting simulation...")


def save_frame(positions_2d, frame_idx):
    """Inverse transforms simulation points and saves them as a .ply file."""
    unnorm_points = scaler.inverse(positions_2d)

    # Prepare 3D data for output (Z is constant)
    output_points_3d = np.zeros((len(unnorm_points), 3), dtype=np.float32)
    output_points_3d[:, 0] = unnorm_points[:, 0]
    output_points_3d[:, 1] = unnorm_points[:, 1]
    output_points_3d[:, 2] = 0.5

    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = o3d.utility.Vector3dVector(output_points_3d)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, f"{frame_idx:05d}.ply"), output_pcd)


# --- Simulation Loop ---
# Save initial state (frame 0)
initial_particles = mpm.particle_info()
save_frame(initial_particles['position'], 0)

for frame in range(1, 100):
    mpm.step(1e-2)

    particles = mpm.particle_info()
    save_frame(particles['position'], frame)

    print('.', end='', flush=True)

print(f"\nSimulation finished. Output files are in '{args.out_dir}'.")
