"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
import os
import imageio

import fusion


if __name__ == "__main__":

  print("Estimating voxel volume bounds...")
  n_imgs = 1000
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')

  # load volume bounds if exists
  if os.path.exists("vol_bounds.txt"):
    vol_bnds = np.loadtxt("vol_bounds.txt", delimiter=' ')
  else:
    vol_bnds = np.zeros((3,2))
    for i in range(n_imgs):
      # Read depth image and camera pose
      depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
      depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
      depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
      cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix

      # Task1: Convert depth image to world coordinates
      view_frust_pts = fusion.cam_to_world(depth_im, cam_intr, cam_pose)
      
      # Extend voxel volume bounds
      vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
      vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))


      # save voxel volume bounds
    np.savetxt("vol_bounds.txt", vol_bnds, delimiter=' ')

  # Initialize TSDF voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read depth image and camera pose
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

    # Read color image
    color_im = cv2.imread("data/frame-%06d.color.jpg"%(i))
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB) / 255.0

    # Integrate observation into voxel volume
    tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.,color_im=color_im)

    if i % 10 == 0:
      tsdf_vol.export_mesh("result/%06d.ply"%(i))

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)

