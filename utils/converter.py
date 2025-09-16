import os
import sys
import csv
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def read_csv_file(filename):
    matrix = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row = [float(cell) for cell in row]  # Convert row elements to numbers
            matrix.append(row)
    return matrix


def plot_heatmap(matrix, block, xlim=None, ylim=None):
    matrix = list(zip(*matrix))
    # Create a new figure window
    plt.figure()
    # Convert matrix to a NumPy array
    data = np.array(matrix)

    # Define ranges to limit the display
    if xlim is not None and ylim is not None:
        x_start, x_end = xlim
        y_start, y_end = ylim
        data = data[y_start:y_end + 1, x_start:x_end + 1]

    # Get the size of the data
    num_rows, num_cols = data.shape

    # Create a heatmap using extent and origin
    plt.imshow(data, cmap='hot', interpolation='nearest', extent=[0, num_cols, 0, num_rows], origin='lower')

    # Add a color bar
    plt.colorbar()

    # Set axis labels corresponding to the actual xlim and ylim values
    if xlim is not None:
        plt.xticks(np.arange(0, num_cols), np.arange(xlim[0], xlim[1] + 1))
    if ylim is not None:
        plt.yticks(np.arange(0, num_rows), np.arange(ylim[0], ylim[1] + 1))

    # Display the plot without blocking
    plt.show(block=block)


def compute_intersection_area(x, y, i, j):
    # Coordinates of the top-left corner of the square (i, j)
    x1, y1 = i - 0.5, j - 0.5
    # Coordinates of the bottom-right corner of the square (i+1, j+1)
    x2, y2 = i + 0.5, j + 0.5

    # Calculate the coordinate of the top-left corner of the intersection
    x_left = max(x - 0.5, x1)
    y_top = min(y + 0.5, y2)

    # Calculate the coordinate of the bottom-right corner of the intersection
    x_right = min(x + 0.5, x2)
    y_bottom = max(y - 0.5, y1)

    # Calculate the intersection area
    intersection_area = max(0, x_right - x_left) * max(0, y_top - y_bottom)

    return intersection_area


def expand_matrix(matrix, n1, m1):
    n, m = matrix.shape
    expanded_matrix = np.zeros((n1, m1))
    expanded_matrix[:n, :m] = matrix
    expanded_matrix[:n, m:] = np.repeat(matrix[:, -1][:, np.newaxis], m1 - m, axis=1)
    expanded_matrix[n:, :] = np.repeat(expanded_matrix[n - 1, :][np.newaxis, :], n1 - n, axis=0)
    return expanded_matrix


def shift(x, y):
    # Create a new plot
    x -= 0.5
    y -= 0.5

    # Calculate and output the intersection area with each of the 9 squares
    matrix = np.zeros((3, 3))
    for i in range(-1, 2):
        for j in range(-1, 2):
            intersection_area = compute_intersection_area(x, y, i, j)
            matrix[i + 1][j + 1] = intersection_area
    # Show the plot
    return matrix


def find_core(sp):
    min_x = min(min(sp["I"]["x"]), min(sp["V"]["x"]), min(sp["R"]["x1"]), min(sp["R"]["x2"]))
    max_x = max(max(sp["I"]["x"]), max(sp["V"]["x"]), max(sp["R"]["x1"]), max(sp["R"]["x2"]))
    min_y = min(min(sp["I"]["y"]), min(sp["V"]["y"]), min(sp["R"]["y1"]), min(sp["R"]["y2"]))
    max_y = max(max(sp["I"]["y"]), max(sp["V"]["y"]), max(sp["R"]["y1"]), max(sp["R"]["y2"]))
    return (min_x, max_x, min_y, max_y)


def read_sp(filename):
    # calculate the coordinates of all instances
    sp = {
        "I": {
            "x": [],
            "y": [],
            "v": [],
            "m": []
        },
        "V": {
            "x": [],
            "y": [],
            "v": [],
            "m": []
        },
        "R": {
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
            "m1": [],
            "m2": [],
            "v": []
        }
    }
    with open(filename, 'r') as file:
        for line in file:
            if line[0] == "I":
                parts = line.strip().split(' ')
                coordinates = parts[1].split('_')
                sp["I"]["x"].append(int(coordinates[2]))
                sp["I"]["y"].append(int(coordinates[3]))
                sp["I"]["v"].append(float(parts[3]))
                sp["I"]["m"].append(coordinates[1])
            elif line[0] == "R":
                parts = line.strip().split(' ')
                coordinates1 = parts[1].split('_')
                coordinates2 = parts[2].split('_')
                sp["R"]["x1"].append(int(coordinates1[2]))
                sp["R"]["y1"].append(int(coordinates1[3]))
                sp["R"]["x2"].append(int(coordinates2[2]))
                sp["R"]["y2"].append(int(coordinates2[3]))
                sp["R"]["m1"].append(coordinates1[1])
                sp["R"]["m2"].append(coordinates2[1])
                sp["R"]["v"].append(float(parts[3]))
            elif line[0] == "V":
                parts = line.strip().split(' ')
                coordinates = parts[1].split('_')
                sp["V"]["x"].append(int(coordinates[2]))
                sp["V"]["y"].append(int(coordinates[3]))
                sp["V"]["v"].append(float(parts[3]))
                sp["V"]["m"].append(coordinates[1])
    return sp


def write_matrix_to_csv(matrix, file_path):
    # Open the file for writing
    with open(file_path, mode='w', newline='') as csvfile:
        # Create a writer object
        writer = csv.writer(csvfile)

        # Write each row of the matrix to the file
        for row in matrix:
            writer.writerow(row)


def netlist2currmap(sp, force_dim=None, step_x=2000, step_y=2000, mv=(0, 0)):
    # Create an empty matrix
    x_coord = sp["I"]["x"]
    y_coord = sp["I"]["y"]
    values = sp["I"]["v"]
    # Calculate the core size
    _, max_x, _, max_y = find_core(sp)
    x_width = int(max_x / step_x) + 1
    y_width = int(max_y / step_y) + 1
    matrix = np.zeros((x_width, y_width))
    for i in range(len(values)):
        posx = (x_coord[i] // step_x)
        posy = (y_coord[i] // step_y)

        # Calculate coefficients for value distribution
        dx = x_coord[i] / step_x - posx
        dy = y_coord[i] / step_y - posy
        sh = shift(dx, dy)

        # Determine if the point lies on the boundary
        is_bndry = 0
        if posx in [0, x_width - 1]:
            is_bndry = 1
        if posy in [0, y_width - 1]:
            is_bndry = 1

        # Distribute the value among the cells
        if is_bndry:
            matrix[posx, posy] += values[i]
        else:
            matrix[posx - 1:posx - 1 + 3, posy - 1:posy - 1 + 3] += sh * values[i]
    matrix = gaussian_filter(matrix, sigma=0.6)


    if force_dim != None:
        matrix = expand_matrix(matrix, force_dim[0], force_dim[1])
    return matrix[mv[0]:,mv[1]:]


def netlist2voltge(sp, force_dim=None, step_x=2000, step_y=2000, mv=(0, 0)):
    # Create an empty matrix
    x_coord = sp["V"]["x"]
    y_coord = sp["V"]["y"]
    values = sp["V"]["v"]
    # Calculate the core size
    _, max_x, _, max_y = find_core(sp)
    x_width = int(max_x / step_x) + 1
    y_width = int(max_y / step_y) + 1
    matrix = np.zeros((x_width, y_width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            x = 1000 + 2000 * i
            y = 1000 + 2000 * j
            dist = []
            sum_1 = 0
            for k in range(len(values)):
                x_p = x_coord[k]
                y_p = y_coord[k]
                dst = ((x_p - x) ** 2 + (y_p - y) ** 2) ** (0.5)
                dist.append(dst)
                sum_1 += 1 / dst
            matrix[i, j] = (1 / sum_1) / 2000

    if force_dim != None:
        matrix = expand_matrix(matrix, force_dim[0], force_dim[1])
    return matrix[mv[0]:,mv[1]:]


def netlist2density(sp, force_dim=None, step_x=2000, step_y=2000, mv=(0, 0)):
    # Create an empty matrix
    x1 = sp["R"]["x1"]
    y1 = sp["R"]["y1"]
    x2 = sp["R"]["x2"]
    y2 = sp["R"]["y2"]
    values = sp["R"]["v"]
    # Calculate the core size
    _, max_x, _, max_y = find_core(sp)
    x_width = int(max_x / step_x) + 1
    y_width = int(max_y / step_y) + 1
    matrix = np.zeros((x_width, y_width))
    for i in range(len(values)):
        posx = int(x1[i] // step_x)
        posy = int(y1[i] // step_y)
        matrix[posx, posy] += values[i]
        posx = int(x2[i] // step_x)
        posy = int(y2[i] // step_y)
        matrix[posx, posy] += values[i]
    matrix = gaussian_filter(matrix, 4)
    if force_dim != None:
        matrix = expand_matrix(matrix, force_dim[0], force_dim[1])
    return matrix[mv[0]:,mv[1]:]


def convert_spice(sp_in, curr_out, volt_out, dens_out, force_dim=None):
    sp = read_sp(sp_in)
    curr_map = netlist2currmap(sp, force_dim=force_dim)
    voltage_map = netlist2voltge(sp, force_dim=force_dim)
    density_map = netlist2density(sp, force_dim=force_dim)
    if curr_out is not None:
        write_matrix_to_csv(curr_map, curr_out)
    if volt_out is not None:
        write_matrix_to_csv(voltage_map, volt_out)
    if dens_out is not None:
        write_matrix_to_csv(density_map, dens_out)
    return curr_map, voltage_map, density_map


def convert_irdrop_to_csv_file(input_file, output_file, force_dim=None):
    lines = open(input_file).readlines()[1:]
    maxx = 0
    maxy = 0
    points = []
    values = []
    for l in lines:
        arr = l.strip().split(',')
        irdrop = float(arr[1])
        data = arr[0].split('_')
        if data[0] == '0':
            continue
        if data[1] != 'm1':
            continue
        x = int(data[-2])
        y = int(data[-1])
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
        points.append((x / 2000, y / 2000))
        values.append(irdrop)

    x_width = int(maxx / 2000) + 1
    y_width = int(maxy / 2000) + 1
    matrix = np.zeros((x_width, y_width))

    points = np.array(points)
    values = np.array(values)
    grid_x, grid_y = np.mgrid[0:matrix.shape[0], 0:matrix.shape[1]]
    pred = griddata(points, values, (grid_x, grid_y), method='cubic')
    if force_dim != None:
        pred = expand_matrix(pred, force_dim[0], force_dim[1])

    # In case of nan replace
    pred = np.nan_to_num(pred, copy=True, nan=0.0)

    if 0:
        if (pred < 0).any():
            print(len(pred[pred < 0].flatten()))
            print('Negative...')
            exit()

    write_matrix_to_csv(pred, output_file)


if __name__ == "__main__":
    # Checking for required command line arguments
    if len(sys.argv) != 3:
        print("Usage: python converter.py path/to/source/folder path/to/destination/folder")
        print("INP - Folder with files netlist.sp & ir_drop.csv")
        print("INP - Folder for output results current_map.csv & eff_dist_map.csv & pdn_density.csv & ir_drop_map.csv")
        sys.exit(1)

    # Extracting Command Line Arguments
    if(not os.path.exists(sys.argv[1])):
        raise FileExistsError(f"Path to {sys.argv[1]} does not exist")

    if(not os.path.exists(sys.argv[2])):
        raise FileExistsError(f"Path to {sys.argv[2]} does not exist")
    
    spice_file = os.path.join(sys.argv[1], "netlist.sp")
    ir_drop_file = os.path.join(sys.argv[2], "netlist.csv")
    current_map_file = os.path.join(sys.argv[2], "current_map.csv")
    voltage_map_file = os.path.join(sys.argv[2], "eff_dist_map.csv")
    density_map_file = os.path.join(sys.argv[2], "pdn_density.csv")
    ir_drop_map_file = os.path.join(sys.argv[2], "ir_drop_map.csv")

    # Calling functions to convert files
    convert_spice(spice_file, current_map_file, voltage_map_file, density_map_file)
    convert_irdrop_to_csv_file(ir_drop_file, ir_drop_map_file)

    verbose = 0
    if verbose:
        plot_heatmap(read_csv_file(current_map_file), 0)
        plot_heatmap(read_csv_file(voltage_map_file), 0)
        plot_heatmap(read_csv_file(density_map_file), 0)
        plot_heatmap(read_csv_file(ir_drop_map_file), 1)
