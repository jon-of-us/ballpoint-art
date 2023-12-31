import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def convert(image: np.ndarray, start_node_image: np.ndarray = None):
    """every pixal which is 0 in startpoint_image is selected as start node"""
    normalized = image / 255

    # weight = 1 / (min_step + normalized * (max_step - min_step))
    weight = 1 - normalized

    width, height = image.shape
    # zero is reserved for the start node
    node_indices = np.arange(width * height).reshape(width, height) + 1

    vectors = np.array([[0, 1], [1, 1], [1, 2], [2, 1]])
    rotated_vectors = vectors[:, ::-1].copy()
    rotated_vectors[:, 0] *= -1
    vectors = np.concatenate((vectors, rotated_vectors))

    nodes0 = []
    nodes1 = []
    weights = []

    if start_node_image is None or not start_node_image.any():
        start_node_image = np.ones_like(image)
        start_node_image[0, 0] = 0
    """nodes with plain white color are selected as start nodes"""
    start_nodes = node_indices[start_node_image == 0]
    print("number of start nodes for dijkstra integral: ", len(start_nodes))
    nodes0.append(np.zeros_like(start_nodes))
    nodes1.append(start_nodes)
    weights.append(np.zeros_like(start_nodes))

    x_range = np.array([0, width])
    y_range = np.array([0, height])

    def range_overlap(a: np.array, b: np.array):
        """ranges as np.arrays [start, end] return overlapping slice"""
        return slice(max(a[0], b[0]), min(a[1], b[1]))

    for vector in vectors:
        x_rng0 = range_overlap(x_range + vector[0], x_range)
        y_rng0 = range_overlap(y_range + vector[1], y_range)
        nodes0.append(node_indices[x_rng0, y_rng0].flatten())
        x_rng1 = range_overlap(x_range - vector[0], x_range)
        y_rng1 = range_overlap(y_range - vector[1], y_range)
        nodes1.append(node_indices[x_rng1, y_rng1].flatten())
        weight_scale = np.linalg.norm(vector)
        weights.append(weight[x_rng0, y_rng0].flatten() * weight_scale)

    matrix = csr_matrix(
        (
            np.concatenate(weights),
            (
                np.concatenate(nodes0),
                np.concatenate(nodes1),
            ),
        ),
        shape=(width * height + 1, width * height + 1),
    )

    dist_matrix = dijkstra(matrix, directed=False, indices=0)
    dist_matrix = dist_matrix[1:]
    dist_matrix = dist_matrix.reshape(image.shape)

    return dist_matrix


if __name__ == "__main__":
    convert(np.ones((10, 12)))
