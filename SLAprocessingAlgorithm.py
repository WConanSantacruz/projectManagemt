def find_and_place_rectangle(matrices, rect_width, rect_height, fill_value):
    """
    Adds a rectangle to the first available space in the list of matrices.
    If no space is available, a new matrix is created.

    Parameters:
    matrices (list of list of list of int): List of matrices to place rectangles.
    rect_width (int): Width of the rectangle.
    rect_height (int): Height of the rectangle.
    fill_value (int): Value to fill the rectangle with.

    Returns:
    list of list of list of int: Updated list of matrices.
    """
    for matrix in matrices:
        rows, cols = len(matrix), len(matrix[0])
        for i in range(rows - rect_height + 1):
            for j in range(cols - rect_width + 1):
                # Check if rectangle fits in the current position
                if all(matrix[i + k][j:j + rect_width] == [0] * rect_width for k in range(rect_height)):
                    # Place the rectangle
                    for k in range(rect_height):
                        matrix[i + k][j:j + rect_width] = [fill_value] * rect_width
                    return matrices  # Return after placing the rectangle

    # If no space found, create a new matrix and place the rectangle
    new_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for k in range(rect_height):
        new_matrix[k][:rect_width] = [fill_value] * rect_width
    matrices.append(new_matrix)
    return matrices

# Example usage
if __name__ == "__main__":
    # Initial matrix list with one 5x5 matrix
    matrices = [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    ]

    # Add rectangles
    matrices = find_and_place_rectangle(matrices, 2, 3, 1)  # Place 2x3 rectangle with value 1
    matrices = find_and_place_rectangle(matrices, 3, 2, 2)  # Place 3x2 rectangle with value 2
    matrices = find_and_place_rectangle(matrices, 5, 5, 3)  # Place 5x5 rectangle with value 3 (new matrix)

    # Print matrices
    for idx, matrix in enumerate(matrices):
        print(f"Matrix {idx + 1}:")
        for row in matrix:
            print(row)
        print()
