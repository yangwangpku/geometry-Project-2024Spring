import openmesh as om
import numpy as np

def bilateral_mesh_denoising(mesh, iterations=10, sigma_c=1.0, sigma_s=1.0):
    for iter in range(iterations):
        # Compute vertex normals if not already present
        if not mesh.has_vertex_normals():
            mesh.request_vertex_normals()
            mesh.update_vertex_normals()

        # Compute the new positions for each vertex
        new_positions = {}
        for vh in mesh.vertices():
            if not mesh.is_boundary(vh):
                p = mesh.point(vh)
                n = mesh.normal(vh)
                neighbors = [(mesh.point(neighbor_vh), mesh.normal(neighbor_vh)) for neighbor_vh in mesh.vv(vh)]
                
                weights = [np.exp(-np.linalg.norm(p - q) ** 2 / (2 * sigma_c ** 2)) * 
                           np.exp(-np.dot(p - q, n - m) / (2 * sigma_s ** 2))
                           for q, m in neighbors]
                
                weighted_sum = sum(w * q for w, (q, m) in zip(weights, neighbors))
                total_weight = sum(weights)
                
                new_positions[vh.idx()] = weighted_sum / total_weight if total_weight > 0 else p

        # Update the vertex positions
        for vh_idx, new_position in new_positions.items():
            mesh.set_point(mesh.vertex_handle(vh_idx), new_position)

        om.write_mesh(f"denoised_{iter}.obj", mesh)
        

# Load the mesh
mesh = om.read_trimesh("smoothing.obj")

# Perform Bilateral Mesh Denoising
bilateral_mesh_denoising(mesh, iterations=5, sigma_c=0.5, sigma_s=0.5)

# Save the denoised mesh
om.write_mesh("denoised.obj", mesh)
