import numpy as np

def bresenham_line(point1, point2):
    """
    Generate the coordinates of points on a line between two given points 
    using Bresenham's line algorithm.

    Parameters
    ----------
    point1 : tuple of int
        The starting point of the line as (x0, y0).
    point2 : tuple of int
        The ending point of the line as (x1, y1).

    Returns
    -------
    np.ndarray
        An array of points representing the line, where each point is a tuple (x, y).
        The endpoint (x1, y1) is excluded from the returned array.
    
    Notes
    -----
    Bresenham's line algorithm is an efficient way to generate points on a straight line 
    between two given coordinates in a grid-based environment, such as for raster graphics.
    """

    # Unpack the coordinates of the starting and ending points
    x0, y0 = point1[0], point1[1]
    x1, y1 = point2[0], point2[1]

    # List to store the points along the line
    points = []

    # Compute the difference between the starting and ending points
    dx = abs(x1 - x0)  # Absolute difference in the x direction
    dy = abs(y1 - y0)  # Absolute difference in the y direction

    # Initialize the starting point
    x, y = x0, y0

    # Determine the step direction in x and y (positive or negative)
    sx = -1 if x0 > x1 else 1  # Step for x direction
    sy = -1 if y0 > y1 else 1  # Step for y direction

    # Determine whether the line is more horizontal or vertical
    if dx > dy:
        # More horizontal: use dx as the major axis
        err = dx / 2.0  # Initialize the error term

        # Iterate until we reach the end point along the x-axis
        while x != x1:
            points.append((x, y))  # Add the current point to the list
            err -= dy  # Update the error term
            if err < 0:  # If error exceeds threshold
                y += sy  # Move in y direction
                err += dx  # Adjust the error term
            x += sx  # Move in x direction
    else:
        # More vertical: use dy as the major axis
        err = dy / 2.0  # Initialize the error term

        # Iterate until we reach the end point along the y-axis
        while y != y1:
            points.append((x, y))  # Add the current point to the list
            err -= dx  # Update the error term
            if err < 0:  # If error exceeds threshold
                x += sx  # Move in x direction
                err += dy  # Adjust the error term
            y += sy  # Move in y direction

    # Add the final endpoint (x1, y1) to the list
    points.append((x, y))

    # Convert the list of points to a NumPy array, excluding the last point
    return np.array(points[:-1])