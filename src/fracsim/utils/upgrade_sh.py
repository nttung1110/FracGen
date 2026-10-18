import numpy as np
from plyfile import PlyData, PlyElement
import argparse


def upgrade_to_sh3(input_ply, output_ply):
    ply = PlyData.read(input_ply)
    vertex = ply["vertex"]
    data = vertex.data

    old_names = data.dtype.names

    # If already has f_rest, skip
    if any(name.startswith("f_rest") for name in old_names):
        print("This PLY already has f_rest fields.")
        return

    n = len(data)

    # Existing fields
    new_dtype = list(data.dtype.descr)

    # Add f_rest_0 ... f_rest_44
    for i in range(45):
        new_dtype.append((f"f_rest_{i}", "f4"))

    new_data = np.empty(n, dtype=new_dtype)

    # Copy old fields
    for name in old_names:
        new_data[name] = data[name]

    # Initialize SH higher-order coefficients to zero
    for i in range(45):
        new_data[f"f_rest_{i}"] = 0.0

    new_vertex = PlyElement.describe(new_data, "vertex")

    # Preserve other elements if any
    elements = [new_vertex]
    for elem in ply.elements:
        if elem.name != "vertex":
            elements.append(elem)

    PlyData(elements, text=ply.text).write(output_ply)

    print(f"Saved upgraded SH degree 3 PLY to: {output_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    upgrade_to_sh3(args.input, args.output)