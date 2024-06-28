import os
import torch
import numpy as np
import trimesh
from skimage import measure
from tqdm import tqdm
from dataset import SDFDataset
from model import SDFMLP
import argparse

def generate_grid_points(grid_size=128, range_min=-1, range_max=1):
    x = np.linspace(range_min, range_max, grid_size)
    y = np.linspace(range_min, range_max, grid_size)
    z = np.linspace(range_min, range_max, grid_size)
    grid_points = np.array(np.meshgrid(x, y, z, indexing='ij'))
    grid_points = grid_points.reshape(3, -1).T
    return grid_points

def test_sdf_model(data_dir, model_path, output_dir, grid_size=128, batch_size=64, use_fourier=False):
    # Load dataset for consistency, although we are generating a grid for testing
    dataset = SDFDataset(data_dir)
    
    # Initialize model and load trained parameters
    model = SDFMLP(use_fourier=use_fourier)
    model.load_state_dict(torch.load(model_path))

    # Check if GPU is available and move model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Generate grid points
    grid_points = generate_grid_points(grid_size)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    # Predict SDF values for grid points
    sdf_values = []
    with torch.no_grad():
        for i in tqdm(range(0, grid_points_tensor.shape[0], batch_size), desc="Predicting SDF values"):
            batch_points = grid_points_tensor[i:i+batch_size]
            batch_sdf_values = model(batch_points).squeeze().cpu().numpy()
            sdf_values.append(batch_sdf_values)
    
    sdf_values = np.concatenate(sdf_values, axis=0).reshape((grid_size, grid_size, grid_size))

    # Apply marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(sdf_values, level=0)

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Save the mesh to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mesh_path = os.path.join(output_dir, 'extracted_mesh.ply')
    mesh.export(mesh_path)

    print(f'Mesh saved to {mesh_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SDF Model and extract mesh.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save the mesh')
    parser.add_argument('--grid_size', type=int, default=128, help='Grid size for the range cube')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction')
    parser.add_argument('--use_fourier', action='store_true', help='Use Fourier feature mapping')

    args = parser.parse_args()

    test_sdf_model(args.data_dir, args.model_path, args.output_dir, args.grid_size, args.batch_size, args.use_fourier)
