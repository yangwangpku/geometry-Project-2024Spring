# Copyright (c) 2018 Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure
import trimesh

class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self.vol_bnds = vol_bnds
    self.voxel_size = float(voxel_size)
    self.trunc_margin = 5 * self.voxel_size  # truncation on SDF

    # Compute voxel grid dimensions
    self.vox_dims = (np.ceil((self.vol_bnds[:, 1] - self.vol_bnds[:, 0]) / self.voxel_size) + 1).astype(int)

    # Initialize voxel volume (TSDF, color and weight)

    # set the initial tsdf_vol to trunc_margin for better results near boundaries
    self.tsdf_vol = (np.zeros(self.vox_dims) + self.trunc_margin).reshape(-1) # truncated signed distance function
    self.color_vol = np.ones((self.tsdf_vol.shape[0],3)) # color volume
    self.weight_vol = (np.zeros(self.vox_dims) + 1e-6).reshape(-1)

    # Get voxel grid coordinates
    x = self.vol_bnds[0, 0] - 1e-6 + np.arange(0,self.vox_dims[0]) * self.voxel_size
    y = self.vol_bnds[1, 0] - 1e-6 + np.arange(0,self.vox_dims[1]) * self.voxel_size
    z = self.vol_bnds[2, 0] - 1e-6 + np.arange(0,self.vox_dims[2]) * self.voxel_size

    # The order of np.meshgrid is confusing, but this is the correct order
    yy,xx,zz = np.meshgrid(y,x,z)
    self.vox_coords = np.stack([xx,yy,zz],axis=-1).reshape(-1, 3)

  def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.,color_im=None):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """

    # Task2: Convert voxel grid coordinates to pixel coordinates

    vox_coords_homogeneous = np.hstack((self.vox_coords, np.ones((self.vox_coords.shape[0], 1))))
    cam_coords = np.dot(np.linalg.inv(cam_pose), vox_coords_homogeneous.T).T[:, :3]
    pix_coords = np.dot(cam_intr, cam_coords.T).T
    pix_coords = pix_coords[:, :2] / pix_coords[:, 2:]

    # Eliminate pixels outside depth images
    valid_pix = np.logical_and(pix_coords[:, 0] >= 0, pix_coords[:, 0] < depth_im.shape[1])
    valid_pix = np.logical_and(valid_pix, pix_coords[:, 1] >= 0)
    valid_pix = np.logical_and(valid_pix, pix_coords[:, 1] < depth_im.shape[0])

    # Sample depth values
    pix_coords = pix_coords[valid_pix].astype(int)
    sampled_depths = depth_im[pix_coords[:, 1], pix_coords[:, 0]]
    sampled_colors = color_im[pix_coords[:, 1], pix_coords[:, 0]]

    # Task3: Compute TSDF for current frame

    # Compute the distance of the sampled points to the camera plane
    dist = sampled_depths - cam_coords[valid_pix][:, 2]

    # Compute the SDF values
    sdf_vals = dist / self.trunc_margin

    # Compute the TSDF values
    tsdf_vals = np.minimum(1.0, np.maximum(-1.0, sdf_vals))

    # Task4: Integrate TSDF into voxel volume
    obs_weights = obs_weight * (abs(sdf_vals) <= 1.0)

    old_tsdf_vol = self.tsdf_vol.copy()

    self.tsdf_vol[valid_pix] = (self.weight_vol[valid_pix] * self.tsdf_vol[valid_pix] + obs_weights * tsdf_vals) / (self.weight_vol[valid_pix] + obs_weights) # update TSDF
    self.color_vol[valid_pix] = (self.weight_vol[valid_pix][:, None] * self.color_vol[valid_pix] + obs_weights[:, None] * sampled_colors) / (self.weight_vol[valid_pix][:, None] + obs_weights[:, None])  # update color
    self.weight_vol[valid_pix] = self.weight_vol[valid_pix] + obs_weights
    
    # tsdf_diff = np.sum(np.abs(old_tsdf_vol[valid_pix] - tsdf_vals) * obs_weights) / np.sum(obs_weights)
    # print("tsdf_diff:",tsdf_diff)

    # valid_coords = self.vox_coords[valid_pix]
    # used_coords = valid_coords[obs_weights > 0.5]
    # print("used_coords_x_min:",used_coords[:,0].min())
    # print("used_coords_x_max:",used_coords[:,0].max())
    # print("used_coords_y_min:",used_coords[:,1].min())
    # print("used_coords_y_max:",used_coords[:,1].max())
    # print("used_coords_z_min:",used_coords[:,2].min())
    # print("used_coords_z_max:",used_coords[:,2].max())
    

  def export_mesh(self, filename):
    """Export the TSDF volume to a mesh file in PLY format.

    Args:
        filename (str): The path to the output file.
    """

    # Extract the surface mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(self.tsdf_vol.reshape(self.vox_dims), level=0.0, spacing=(self.voxel_size, self.voxel_size, self.voxel_size))
    
    # Adjust the vertex positions based on the volume bounds
    verts += self.vol_bnds[:, 0]

    # Find the nearest voxel index for each vertex
    verts_idx = np.floor((verts - self.vol_bnds[:, 0]) / self.voxel_size + 0.5).astype(int)

    # Get the color for each vertex
    colors = self.color_vol[verts_idx[:, 0] * self.vox_dims[1] * self.vox_dims[2] + verts_idx[:, 1] * self.vox_dims[2] + verts_idx[:, 2] ] * 255.0

    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)

    # Export the mesh to a PLY file
    mesh.export(filename, file_type='ply')

  
    

def cam_to_world(depth_im, cam_intr, cam_pose,export_name="pointcloud.ply"):
  """Get 3D point cloud from depth image and camera pose
  """
  # Get the shape of the depth image
  height, width = depth_im.shape

  # Create a grid of pixel coordinates
  xx, yy = np.meshgrid(np.arange(width), np.arange(height))

  # Flatten the grid and depth image
  xx = xx.flatten()
  yy = yy.flatten()
  depth = depth_im.flatten()

  # Stack the pixel coordinates to create a matrix of the form
  pix_coords = np.vstack((xx * depth, yy * depth, depth)).T
  # Convert pixel coordinates to normalized camera coordinates
  camera_pts = np.dot(np.linalg.inv(cam_intr), pix_coords.T).T

  # Add a column of ones for homogeneous coordinates
  camera_pts_homogeneous = np.hstack((camera_pts, np.ones((camera_pts.shape[0], 1))))

  # Transform points from camera frame to world frame
  world_pts_homogeneous = np.dot(cam_pose, camera_pts_homogeneous.T).T
  # Convert back from homogeneous coordinates
  world_pts = world_pts_homogeneous[:, :3]


  # Create a point cloud
  pointcloud = trimesh.PointCloud(vertices=world_pts)
  # Export the point cloud
  pointcloud.export(export_name)
  return world_pts.T
