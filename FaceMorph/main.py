import face_landmark_detection
from triangulation import make_delaunay
from face_morphing_sequence import morph_video
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# Convert cv2 image to TK
def cv2_to_imageTK(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    imagePIL = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=imagePIL)
    return imgtk


# Convert photo image to cv2
def photoimage_to_cv2(photo_image):
    # First, get the image data
    width, height = photo_image.width(), photo_image.height()
    data = photo_image.get(0, 0, width - 1, height - 1)
    data = list(data)
    img = []

    # Convert data to RGB format
    for index in range(0, len(data), 3):
        img.append(data[index:index + 3])

    # Reshape data
    img = np.array(img, dtype=np.uint8).reshape((height, width, 3))
    return img


# Load and display image on canvas
def load_and_display_image(canvas):
    filepath = filedialog.askopenfilename(title="Select an Image",
                                          filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    current_directory = os.path.normpath(os.getcwd()).replace('\\', '/')
    filepath = filepath.replace(current_directory + '/', '')
    if not filepath:  # User closed the dialog or chose no file
        return None

    image = cv2.imread(filepath)
    height, width, _ = image.shape

    # Calculate the desired canvas size (you can adjust the scale factor as needed)
    scale_factor = 0.5  # You can adjust this to control the canvas size
    canvas_width = int(width * scale_factor)
    canvas_height = int(height * scale_factor)

    # Set the canvas size to match the image
    canvas.config(width=canvas_width, height=canvas_height)

    # Resize the image to match the canvas size
    resized_image = cv2.resize(image, (canvas_width, canvas_height))

    tk_img = cv2_to_imageTK(resized_image)

    # Display the resized image on the canvas
    canvas.image = tk_img  # Keep a reference
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    return resized_image, filepath


# Get the current loaded image name on canvas1
def update_image1_wrapper():
    global img1_src
    # Update the container with the return value
    img1_src, image1_name = load_and_display_image(canvas1)
    image_paths.append(image1_name)


# Get the current loaded image name on canvas2
def update_image2_wrapper():
    global img2_src
    # Update the container with the return value
    img2_src, image2_name = load_and_display_image(canvas2)
    image_paths.append(image2_name)


def process_images():
    global size, img1, img2, points1, points2, list3
    if len(image_paths) < 2:
        print("Both images not loaded yet!")
        return
    # vec1, vec2 = landmark_detection_test.detect_landmarks(img1_src, img2_src)
    [size, img1, img2, points1, points2, list3] = face_landmark_detection.generate_face_correspondences(img1_src,
                                                                                                        img2_src)
    radius = 1
    # Draw landmarks on canvas1
    for point in points1:
        canvas1.create_oval(point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius, fill='red')
    # Draw landmarks on canvas2
    for point in points2:
        canvas2.create_oval(point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius, fill='red')


# Get the point user clicked on canvas1
def on_canvas1_click(event):
    # Capture the x and y coordinates of the click
    if canvas1.image and len(points1) != 0:
        x, y = event.x, event.y
        print(f"Clicked at x={x}, y={y}")

        # Store the coordinates in the global points list
        points1.append((x, y))

        # Optionally, draw a small circle on the canvas where the user clicked
        radius = 1
        canvas1.create_oval(x - radius, y - radius, x + radius, y + radius, fill='blue')


# Get the point user clicked on canvas2
def on_canvas2_click(event):
    # Capture the x and y coordinates of the click
    if canvas2.image and len(points2) != 0:
        x, y = event.x, event.y
        print(f"Clicked at x={x}, y={y}")

        # Store the coordinates in the global points list
        points2.append((x, y))

        # Optionally, draw a small circle on the canvas where the user clicked
        radius = 1
        canvas2.create_oval(x - radius, y - radius, x + radius, y + radius, fill='blue')


def compute_and_display_result():
    global narray
    # background points
    common_points = [
        (1, 1),
        (size[1] - 1, 1),
        ((size[1] - 1) // 2, 1),
        (1, size[0] - 1),
        (1, (size[0] - 1) // 2),
        ((size[1] - 1) // 2, size[0] - 1),
        (size[1] - 1, size[0] - 1),
        ((size[1] - 1), (size[0] - 1) // 2)
    ]
    points1.extend(common_points)
    points2.extend(common_points)
    narray = np.append(list3, common_points, axis=0)
    # Do face morphing if canvas1 and canvas2 both have images
    if canvas1.image and canvas2.image:
        doMorphing(img1_src, img2_src, 5, 20, 'outputs/output.mp4')


def doMorphing(img1, img2, duration, frame_rate, output):
    tri = make_delaunay(size, narray)

    morph_video(duration, frame_rate, img1, img2, points1, points2, tri, size, output)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Load and Display Images")
    root.geometry("1500x1000")
    image_paths = []
    # Global list to store the points (x, y coordinates) clicked by the user
    points1 = []
    points2 = []

    # Create Canvas widgets for displaying images
    canvas1 = tk.Canvas(root, bg='gray')
    canvas1.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
    canvas1.bind("<Button-1>", on_canvas1_click)

    canvas2 = tk.Canvas(root, bg='gray')
    canvas2.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
    canvas2.bind("<Button-1>", on_canvas2_click)

    # Buttons for loading images

    btn1 = tk.Button(root, text="Load Image 1", command=update_image1_wrapper)
    btn1.grid(row=1, column=0, pady=10, sticky='ew')

    btn2 = tk.Button(root, text="Load Image 2", command=update_image2_wrapper)
    btn2.grid(row=1, column=1, pady=10, sticky='ew')

    process_btn = tk.Button(root, text="Process Images", command=process_images)
    process_btn.grid(row=2, column=0, pady=10, sticky='ew')

    compute_btn = tk.Button(root, text="Compute Result", command=compute_and_display_result)
    compute_btn.grid(row=1, column=2, pady=10, sticky='ew')

    # Allow the columns to expand
    for i in range(3):
        root.grid_columnconfigure(i, weight=1)

    root.mainloop()
