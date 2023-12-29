import cv2

# Check if a point is within a given rectangle
def rect_contains(rect, point):
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

# Create a Delaunay triangulation for the given data points
def make_delaunay(size, data_points):
    # Create a rectangular region
    frame_width = size[1]
    frame_height = size[0]

    rect = (0, 0, frame_width, frame_height)

    # Initialize a Delaunay subdivision using Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Convert the data points list to a more accessible format
    data_points = data_points.tolist()
    points = [(int(x[0]), int(x[1])) for x in data_points]
    point_dict = {x[0]: x[1] for x in list(zip(points, range(len(data_points))))}

    # Insert data points into the subdivision
    for p in points:
        subdiv.insert(p)

    # Create a list of Delaunay triangles
    delaunay_triangles = []
    triangle_list = subdiv.getTriangleList()
    rect_region = (0, 0, frame_width, frame_height)

    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(rect_region, pt1) and rect_contains(rect_region, pt2) and rect_contains(rect_region, pt3):
            delaunay_triangles.append((point_dict[pt1], point_dict[pt2], point_dict[pt3]))

    # Return the list of Delaunay triangles
    return delaunay_triangles