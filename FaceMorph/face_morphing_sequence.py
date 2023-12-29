from subprocess import Popen, PIPE
import cv2
import numpy as np
from PIL import Image


# Interpolate the image points
def affine_transform(image, M, output_size):
    img_height, img_width = image.shape[:2]
    output_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    for y_out in range(output_size[1]):
        for x_out in range(output_size[0]):
            input_coords = np.dot(M, [y_out, x_out, 1])
            x_in, y_in = input_coords[:2]

            # Clamp the coordinates to the valid image range
            x_in = min(max(x_in, 0), img_width - 1)
            y_in = min(max(y_in, 0), img_height - 1)

            x1, y1 = int(x_in), int(y_in)
            x2, y2 = min(img_width - 1, x1 + 1), min(img_height - 1, y1 + 1)

            dx = x_in - x1
            dy = y_in - y1

            for channel in range(3):
                pixel_value = (1 - dx) * (1 - dy) * image[y1, x1, channel] + \
                              dx * (1 - dy) * image[y1, x2, channel] + \
                              (1 - dx) * dy * image[y2, x1, channel] + \
                              dx * dy * image[y2, x2, channel]
                output_image[x_out, y_out, channel] = int(pixel_value)

    return output_image


# Apply an affine transformation to the source image based on source and destination triangles
def triangle_transformation(source_image, source_triangles, destination_triangles, output_size):
    # Calculate the affine transformation matrix for triangles
    transformation_matrix = cv2.getAffineTransform(np.float32(source_triangles), np.float32(destination_triangles))

    # Apply the found affine transformation to the source image
    transformed_image = cv2.warpAffine(source_image, transformation_matrix, (output_size[0], output_size[1]), None,
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return transformed_image


# Warp and blend triangular regions from img1 and img2 to create the output image
def morph_triangle(image1, image2, output_image, triangles1, triangles2, triangles, alpha):
    # Calculate bounding rectangles for the triangles
    rect1 = cv2.boundingRect(np.float32([triangles1]))
    rect2 = cv2.boundingRect(np.float32([triangles2]))
    rect = cv2.boundingRect(np.float32([triangles]))

    # Offset triangle points by the top-left corner of their respective rectangles
    triangles_rect = [((point[0] - rect[0]), (point[1] - rect[1])) for point in triangles]
    triangles1_rect = [((point[0] - rect1[0]), (point[1] - rect1[1])) for point in triangles1]
    triangles2_rect = [((point[0] - rect2[0]), (point[1] - rect2[1])) for point in triangles2]

    # Create a mask by filling the triangle
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(triangles_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warp transformation to small rectangular patches
    image1_rect = image1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    image2_rect = image2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

    output_size = (rect[2], rect[3])
    warp_image1 = triangle_transformation(image1_rect, triangles1_rect, triangles_rect, output_size)
    warp_image2 = triangle_transformation(image2_rect, triangles2_rect, triangles_rect, output_size)

    # Alpha blend rectangular patches
    rect_image = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    # Copy the triangular region from the rectangular patch to the output image
    rect_mask = output_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    output_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        rect_mask * (1 - mask) + rect_image * mask


# Generate a sequence of morphed images between two input images
def morph_video(duration, frame_rate, image1, image2, landmarks1, landmarks2, triangles_list, output_size,
                output_path):
    num_images = int(duration * frame_rate)
    ffmpeg_command = [
        'ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate), '-s', f'{output_size[1]}x{output_size[0]}', '-i',
        '-',
        '-c:v', 'libx264', '-crf', '25', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', output_path
    ]

    process = Popen(ffmpeg_command, stdin=PIPE)

    for j in range(num_images):
        image1 = np.float32(image1)
        image2 = np.float32(image2)
        interpolated_landmarks = []
        alpha = j / (num_images - 1)
        morphed_frame = np.zeros(image1.shape, dtype=image1.dtype)
        # Interpolate landmarks using weighted average
        for i in range(len(landmarks1)):
            x = (1 - alpha) * landmarks1[i][0] + alpha * landmarks2[i][0]
            y = (1 - alpha) * landmarks1[i][1] + alpha * landmarks2[i][1]
            interpolated_landmarks.append((x, y))
        for i in range(len(triangles_list)):
            x, y, z = triangles_list[i]
            triangles1 = [landmarks1[x], landmarks1[y], landmarks1[z]]
            triangles2 = [landmarks2[x], landmarks2[y], landmarks2[z]]
            triangles = [interpolated_landmarks[x], interpolated_landmarks[y], interpolated_landmarks[z]]

            # Morph one triangle at a time
            morph_triangle(image1, image2, morphed_frame, triangles1, triangles2, triangles, alpha)

            pt1 = tuple(map(int, triangles[0]))
            pt2 = tuple(map(int, triangles[1]))
            pt3 = tuple(map(int, triangles[2]))

            cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)

        result_image = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
        result_image.save(process.stdin, 'JPEG')

    process.stdin.close()
    process.wait()
    return result_image
