"""Render a multiview NeRF-style dataset from a single .glb file using Blender.

Adapted from the Pixie data curation renderer
(research/Pixie/pixie/data_curation/render_blender.py) and from
blender/render_with_blender.py in this repo. Unlike those two scripts, this one
only supports .glb input and always uses the BlenderNeRF "camera on sphere"
operator so the output directory matches the structure already used by
blender/rendered_data/rubber1 (transforms.json, transforms_train.json,
transforms_test.json, train/*.png).
"""

import argparse
import math
import os
import shutil
import sys

import bpy
from mathutils import Vector


def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.get_devices()

    for compute_device_type in ("CUDA", "OPENCL", "NONE"):
        try:
            cprefs.compute_device_type = compute_device_type
            print(f"Compute device selected: {compute_device_type}")
            break
        except TypeError:
            pass

    accelerated_types = ("CUDA", "OPENCL")
    accelerated = any(device.type in accelerated_types for device in cprefs.devices)
    print(f"Accelerated render = {accelerated}")

    for device in cprefs.devices:
        device.use = not accelerated or device.type in accelerated_types
        print(f"Device enabled ({device.type}) = {device.use}")

    return accelerated


enable_cuda_devices()

parser = argparse.ArgumentParser(description="Render a multiview dataset from a .glb file.")
parser.add_argument("--obj_path", type=str, required=True, help="Path to the .glb file to render.")
parser.add_argument(
    "--output_dir", type=str, required=True,
    help="Directory the dataset is written to (e.g. blender/rendered_data/<obj_name>).",
)
parser.add_argument("--num_images", type=int, default=200, help="Number of training views to render.")
parser.add_argument("--resolution", type=int, default=512, help="Render resolution (square).")
parser.add_argument("--camera_dist_min", type=float, default=1.0, help="Minimum camera distance from origin.")
parser.add_argument("--camera_dist_max", type=float, default=1.4, help="Maximum camera distance from origin.")
parser.add_argument("--scene_scale", type=float, default=1.0, help="Extra scale factor applied after normalization.")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"],
)
parser.add_argument("--samples", type=int, default=32, help="Cycles render samples.")
parser.add_argument("--seed", type=int, default=0, help="Seed for the camera-on-sphere sampling.")
parser.add_argument(
    "--blender_nerf_addon_path", type=str, required=True, help="Path to the BlenderNeRF addon zip file.",
)
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

if not args.obj_path.endswith(".glb"):
    raise ValueError(f"render_glb.py only supports .glb files, got: {args.obj_path}")

print(f"Installing BlenderNeRF addon from: {args.blender_nerf_addon_path}")
bpy.ops.preferences.addon_install(filepath=args.blender_nerf_addon_path, overwrite=True)
bpy.ops.preferences.addon_enable(module="BlenderNeRF")

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = args.samples
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True

scene.world.use_nodes = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

# pure black background (not transparent)
render.film_transparent = False
world = bpy.data.worlds["World"]
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
bg_node.inputs[1].default_value = 1.0


def add_lighting() -> None:
    """Three-point + top studio lighting, matching blender/render_with_blender.py."""
    if "Light" in bpy.data.objects:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="AREA", location=(2, -2, 2))
    key_light = bpy.context.object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 5
    key_light.rotation_euler = (0.6, 0.2, 0.8)

    bpy.ops.object.light_add(type="AREA", location=(-2, -1, 1))
    fill_light = bpy.context.object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 200
    fill_light.data.size = 7
    fill_light.rotation_euler = (0.5, -0.2, -0.8)

    bpy.ops.object.light_add(type="AREA", location=(0, 3, 2))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 300
    rim_light.data.size = 4
    rim_light.rotation_euler = (0.8, 0, 0)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 4))
    top_light = bpy.context.object
    top_light.name = "Top_Light"
    top_light.data.energy = 150
    top_light.data.size = 10
    top_light.rotation_euler = (0, 0, 0)


def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(obj_path: str) -> None:
    bpy.ops.import_scene.gltf(filepath=obj_path, merge_vertices=True)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_bbox():
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes():
        found = True
        for coord in obj.bound_box:
            coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> None:
    """Fit the object into a unit cube centered at the origin."""
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * args.scene_scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def render_with_blendernerf(object_uid: str) -> None:
    """Use the BlenderNeRF "camera on sphere" operator to render the normalized scene."""
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scene = bpy.context.scene
    scene.train_data = True
    scene.test_data = False
    scene.aabb = 32
    scene.render_frames = True
    scene.nerf = True
    scene.save_path = output_dir

    scene.cos_dataset_name = object_uid
    scene.sphere_location = (0.0, 0.0, 0.0)
    scene.sphere_rotation = (0.0, 0.0, 0.0)
    scene.sphere_scale = (1.0, 1.0, 1.0)

    if hasattr(scene, "sphere_radius_min") and hasattr(scene, "sphere_radius_max"):
        scene.sphere_radius_min = args.camera_dist_min / 2
        scene.sphere_radius_max = args.camera_dist_max / 2
        scene.sphere_radius = (args.camera_dist_min + args.camera_dist_max) / 4
    else:
        scene.sphere_radius = args.camera_dist_min / 2

    scene.focal = 20.0
    scene.cos_nb_frames = args.num_images
    scene.seed = args.seed
    scene.upper_views = True
    scene.outwards = False

    try:
        bpy.ops.object.camera_on_sphere()
        print(f"Successfully rendered {args.num_images} images using BlenderNeRF add-on")
    except Exception as e:
        if "BlenderNeRF Camera" in str(e) and "not found" in str(e):
            print(f"Warning: harmless BlenderNeRF camera error (continuing): {e}")
        else:
            print(f"Error during BlenderNeRF rendering: {e}")
            raise

    zip_path = os.path.join(output_dir, f"{object_uid}.zip")
    if os.path.exists(zip_path):
        shutil.unpack_archive(zip_path, output_dir)
        os.remove(zip_path)

    import json
    with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": 0.0, "frames": []}, f)


def process_object(obj_path: str) -> None:
    reset_scene()
    load_object(obj_path)
    object_uid = os.path.basename(obj_path).split(".")[0]

    normalize_scene()
    print(f"Scene normalized for {object_uid}")

    add_lighting()
    render_with_blendernerf(object_uid)

    glb_path = os.path.join(args.output_dir, f"{object_uid}_normalized_scene.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        export_apply=False,
        export_texcoords=True,
        export_normals=True,
        export_attributes=True,
        export_materials="EXPORT",
        export_animations=False,
        check_existing=False,
    )
    print(f"Normalized scene exported to {glb_path}")


if __name__ == "__main__":
    import time

    start = time.time()
    process_object(args.obj_path)
    print(f"Finished {args.obj_path} in {time.time() - start:.1f} seconds")
