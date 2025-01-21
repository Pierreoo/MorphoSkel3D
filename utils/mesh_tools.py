import numpy as np
import trimesh


def export_ply(ref_mesh: trimesh.Trimesh, model_name: str, target_dir="./"):
    result = trimesh.exchange.ply.export_ply(ref_mesh, encoding="ascii")
    output_file = open((target_dir + model_name + ".ply"), "wb+")
    output_file.write(result)
    output_file.close()


def export_glb(ref_mesh: trimesh.Trimesh, model_name: str):
    '''support vertex_colors as Nx4 array in [0, 1]'''
    with open(model_name, 'wb') as f:
        f.write(trimesh.exchange.gltf.export_glb(ref_mesh))
        f.close()


def export_obj(nv: np.ndarray, nf: np.ndarray, name: str, export_lines=False):
    if name[:-4] != ".obj":
        name += ".obj"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")
    for e in nv:
        file.write("v {} {} {}\n".format(*e))
    file.write("\n")
    for face in nf:
        header = "l " if export_lines else "f "
        file.write(header + " ".join([str(fi + 1) for fi in face]) + "\n")
    file.write("\n")


def export_off(nv: np.ndarray, nf: np.ndarray, name: str):
    name += ".off"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")

    file.write("OFF \n")
    file.write("{} {} 0 \n".format(len(nv), len(nf)))
    for e in nv:
        file.write("{} {} {}\n".format(*e))
    for face in nf:
        file.write("{} ".format(len(face)) +
                   " ".join([str(fi) for fi in face]) + "\n")
    file.write("\n")
