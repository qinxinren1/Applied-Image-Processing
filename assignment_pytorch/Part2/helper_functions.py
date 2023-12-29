import torch
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image

def points_to_mask(image, points):
    """ Returns a segmentation mask of the points.
    
    # Parameters:
        @image: np.array size [3, H, W], the image
        @points: np.array size [N, 2], the points defining the convex hull of the mask

    # Returns: np.array size [H, W], the segmentation mask
    """
    img = np.zeros(image.shape[1:], dtype=np.uint8)
    for point in points.round():
        img[int(point[0]), int(point[1])] = 1

    return convex_hull_image(img)

def draw_points(p, t, img, p_radius=0, l_thickness=1):
    """ Draws the points p and t on the image img.

    # Parameters:
        @p: torch.tensor size [2], the current handle point p
        @t: torch.tensor size [2], the target point t
        @img: torch.tensor size [1, 3, H, W], the image to draw on
        @p_radius: int, the radius of the points p and t
        @l_thickness: int, the thickness of the line between p and t
    
    # Returns: np.array size [H, W, 3], the image with the points drawn on it
    """
    img = (img.clone().detach() + 1) * (255/2)
    img = img.clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
    img = img.copy()
    p = p.flip(0).int().cpu().numpy()
    t = t.flip(0).int().cpu().numpy()

    img = cv2.line(img, p, t, (255, 255, 255), l_thickness)
    img = cv2.circle(img, p, p_radius, (255, 0, 0), -1)
    img = cv2.circle(img, t, p_radius, (0, 0, 255), -1)

    return img

def squeeze_all(tensor):
    # NOTE: In PyTorch 2 we would not have to loop
    for i, dim in enumerate(torch.where(torch.tensor(tensor.shape) == 1)[0]):
        tensor = torch.squeeze(tensor, dim - i)

    return tensor


def read_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def select_points(image, coords, mask_points):
    """Wrapper that creates a closure for interactive point selection."""
    plt_image = image[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5

    if coords is None or mask_points is None:
        coords = []
        mask_points = []

        def onclick(event):
            ix, iy = event.xdata, event.ydata
            print (f'Point set at: (x: {ix}, y: {iy})')
            if len(coords) < 2:
                coords.append(np.array([iy, ix]))
            else:
                mask_points.append(np.array([iy, ix]))

            if len(coords) == 1:
                col = torch.tensor([1,0,0])
            elif len(coords) == 2 and len(mask_points) == 0:
                col = torch.tensor([0,0,1])
            else:
                col = torch.tensor([0,1,0])

            image[0,:,int(iy), int(ix)] = col
            imshow.set_data(image[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5)
            fig.canvas.draw()

            if len(mask_points) > 1:
                dist = np.sqrt(((mask_points[0] - mask_points[-1])**2 + 1e-6).sum()) 
                print(dist)
                if dist < 1:
                    fig.canvas.mpl_disconnect(cid)
                    plt.close()
        print("First click: handle point, second click: target point, all other clicks: mask points. "
            + "To finish mask, click on first mask point again.")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        imshow = ax.imshow(plt_image)
        plt.show()

    return coords, mask_points