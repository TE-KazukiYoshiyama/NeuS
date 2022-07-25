import argparse
import numpy as np
import cv2 as cv
import os
from glob import glob
from scipy.io import loadmat
import trimesh

# Most logics come from https://gist.github.com/Totoro97/43664cfc28110a469d88a158af040014

def clean_points_by_mask(points, scan, base_data_path):
    cameras = np.load(f'{base_data_path}/scan{scan}/cameras.npz')
    mask_lis = sorted(glob(f'{base_data_path}/scan{scan}/mask/*.png'))
    n_images = 49 if scan < 83 else 64
    inside_mask = np.ones(len(points)) > 0.5
    for i in range(n_images):
        print(f"  Processing image_id={i}")
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (101, 101))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        inside_mask &= curr_mask.astype(bool)

    return inside_mask


def main(args):
    scans = args.scans

    base_exp_path = args.base_exp_path
    base_data_path = args.base_data_path
    file_name = args.file_name

    for scan in scans:
        print(f"Processing scan{scan}")
        mesh_dir_path = f"{base_exp_path}/scan{scan}/womask_large_roi/meshes"
        old_mesh = trimesh.load(f"{mesh_dir_path}/{file_name}.ply")
        old_vertices = old_mesh.vertices[:]
        old_faces = old_mesh.faces[:]
        mask = clean_points_by_mask(old_vertices, scan, base_data_path)
        indexes = np.ones(len(old_vertices)) * -1
        indexes = indexes.astype(np.long)
        indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

        faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
        new_faces = old_faces[np.where(faces_mask)]
        new_faces[:, 0] = indexes[new_faces[:, 0]]
        new_faces[:, 1] = indexes[new_faces[:, 1]]
        new_faces[:, 2] = indexes[new_faces[:, 2]]
        new_vertices = old_vertices[np.where(mask)]

        new_mesh = trimesh.Trimesh(new_vertices, new_faces)
        
        meshes = new_mesh.split(only_watertight=False)
        new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

        new_mesh.export(f"{mesh_dir_path}/{file_name}_clean.ply")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--scans', type=int, nargs='+', required=True, 
                        default=[ 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122 ])
    parser.add_argument('--base_data_path', type=str, default="./public_data")
    parser.add_argument('--base_exp_path', type=str, default="./exp")
    parser.add_argument('--file_name', type=str, default="00300000")
    args = parser.parse_args()

    main(args)