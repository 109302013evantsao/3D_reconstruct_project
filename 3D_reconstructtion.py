# This is a project a about 3D reconstruction of a scene from a set of 2D images.

# First, get the camera parameters from the set of images.
# to get the camera intrinsic matrix, we need to get the focal length(fx, fy) and the optical point(cx, cy).

# the original paper about point-e has one example ofgenerating 3D point cloud from a single image.
# but i found a example from multi images to generate 3D point cloud.

# with follwing code, https://github.com/halixness/stable-point-e/blob/main/notebooks/4_local_photo_point_e.ipynb


# for the images to point cloud, we need to use the Point-E model to generate the point cloud from the images.
from PIL import Image
from PIL import ImageChops
import torch
from tqdm.auto import tqdm
from torchvision import transforms
from sc_point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from sc_point_e.diffusion.sampler import PointCloudSampler
from sc_point_e.models.download import load_checkpoint
from sc_point_e.models.configs import MODEL_CONFIGS, model_from_config
from sc_point_e.util.plotting import plot_point_cloud
from sc_point_e.util.pc_to_mesh import marching_cubes_mesh

import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
# for image format transformation
from os import listdir
from os.path import splitext
import os

# remove the background
from rembg import remove

# for point cloud to mesh
import skimage.measure as measure

# for point cloud to mesh
import open3d as o3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from https://stackoverflow.com/questions/10759117/converting-jpg-images-to-png
# transform the image format from jpg to png
image_folder_path_jpg_origin = "D:/github/3D_reconstruct_project/input_images/JPG/" # change the path to your image folder
image_folder_path_png_target = "D:/github/3D_reconstruct_project/input_images/PNG/" # change the path to a new folder for transformed images
image_folder_path_rbg = "D:/github/3D_reconstruct_project/input_images/PNG_RBG/"

# -----------------------------------------------------------------------------

# transform the image format from jpg to png
for files in listdir(image_folder_path_jpg_origin):
    filename, extension = splitext(files)
    try:
        if extension not in ['.py', '.png']:
            im = Image.open(image_folder_path_jpg_origin + filename + extension)
            # im=im.rotate(270, expand=True)
            im.save(image_folder_path_png_target + filename + '.png')
    except OSError:
        print('Cannot convert %s' % files)

# -----------------------------------------------------------------------------

# load images and remove background
# change the path to your image folder 
images_folder_path = image_folder_path_png_target
image_fizes = listdir(images_folder_path)
image_list=[]
multi_views = []

for f in image_fizes:

    filename, extension = splitext(f)
    img = Image.open(images_folder_path + f)

    image_size = 768

    # resize the image while maintaining aspect ratio
    aspect_ratio = img.width / img.height
    new_width = image_size
    new_height = int(new_width / aspect_ratio)
    img = img.resize((new_width, new_height))

    img.save(image_folder_path_png_target + filename + '.png')

    # image_list.append(img)

    # BG removal with simple ONNX APIs (pre-trained U2Net model)
    img_rbg = remove(img)
    
    # find the bounding box of the object
    bbox = ImageChops.invert(img_rbg).getbbox()

    # calculate the center of the object
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2

    # create a new image with desired size and paste the object into it
    new_image_size = image_size*2
    new_img = Image.new('RGBA', (new_image_size, new_image_size), (0, 0, 0, 0))

    # calculate the position to paste the object
    paste_x = new_image_size // 2 - center_x
    paste_y = new_image_size // 2 - center_y
    new_img.paste(img_rbg, (paste_x, paste_y))

    new_img.save(image_folder_path_rbg + filename + '.png')

    multi_views.append(new_img)


# num_views = len(image_fizes)
# cols = 2
# # +1 avoid the zero division and not enough rows
# rows = int(round(num_views / cols,0))

# f, axarr = plt.subplots(rows, cols)
# for i in range(num_views):
#     axarr[i // cols, i % cols].imshow(np.asarray(multi_views[i]))

# plt.show()

# -----------------------------------------------------------------------------

times = 3

# create a folder for layout data
# set your layout data folder
layout_data_folder = f"D:/github/3D_reconstruct_project/layout_folder/test{times}"

if not os.path.exists(layout_data_folder):
    os.makedirs(layout_data_folder)

# -----------------------------------------------------------------------------

PILtoTensor = transforms.ToTensor()

# create a point cloud from images with Point-E
print("Creating base model")
base_name = "base300M"  # Use base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print("Creating upsample model")
upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

print("Downloading base checkpoint")
base_model.load_state_dict(load_checkpoint(base_name, device))

print("Downloading upsampler checkpoint")
upsampler_model.load_state_dict(load_checkpoint("upsample", device))

# Combine the image-to-point cloud and upsampler model
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=["R", "G", "B"],
    guidance_scale=[3.0, 3.0],
)

# # Load an image to condition on
# # img_path = "D:/github/3D_reconstruct_project/input_images/20240506_182244.png" # Fill in your image path
# # img = Image.open(img_path)

# Produce a sample from the model (this takes around 3 minutes on base300M)
samples = None

for x in tqdm(
    sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images = multi_views))
):
    samples = x
pc = sampler.output_to_point_clouds(samples)[0]


# beacuse the point cloud format from point-e is not the same as the open3d point cloud format
# we need to convert the point cloud format to open3d point cloud format
# from https://docs.voxel51.com/tutorials/pointe.html

def generate_pcd_from_image(pc):
    pointe_pcd = pc

    channels = pointe_pcd.channels
    r, g, b = channels["R"], channels["G"], channels["B"]
    colors = np.vstack((r, g, b)).T
    points = pointe_pcd.coords

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

item_name = 'Kunimi_Tama'
item_pcd_o3d = generate_pcd_from_image(pc)
item_pcd_file = item_name + f'_{times}_pcd.ply'
o3d.io.write_point_cloud(layout_data_folder + "/" + item_pcd_file, item_pcd_o3d)

# plot the point cloud
o3d.visualization.draw_geometries([item_pcd_o3d])

# --------------------------------------------------------------------

# plot the mesh

# alpha = 0.05
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#     item_pcd_o3d, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

alpha = 0.04
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    item_pcd_o3d, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# alpha = 0.03
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#     item_pcd_o3d, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

o3d.io.write_triangle_mesh(layout_data_folder + "/" + item_name + f"_{times}_mesh.ply", mesh)


mesh.compute_vertex_normals()
item_pcd_o3d = mesh.sample_points_poisson_disk(3000)
o3d.visualization.draw_geometries([item_pcd_o3d], point_show_normal=True)

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               item_pcd_o3d, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([item_pcd_o3d, rec_mesh])

o3d.io.write_triangle_mesh(layout_data_folder + "/" + item_name + f"_{times}_rec_mesh.ply", rec_mesh)































# --------------------------------------------------------------------
# # abandon the code below
# print('creating SDF model...')
# name = 'sdf'
# model = model_from_config(MODEL_CONFIGS[name], device)
# model.eval()

# print('loading SDF model...')
# model.load_state_dict(load_checkpoint(name, device))

# # Produce a mesh (with vertex colors)
# mesh = marching_cubes_mesh(
#     pc=pc,
#     model=model,
#     batch_size=4096,
#     grid_size=32, # increase to 128 for resolution used in evals
#     progress=True,
# )
# # # plot the mesh (wrong)
# # fig = px.scatter_3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2])
# # fig.show()


# with open('3D_reconstuction_mesh.ply', 'wb+') as f:
#     mesh.write_ply(f)