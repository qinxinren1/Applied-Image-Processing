import torch
import torch.optim as optim

from helper_functions import *


def normalize(img, mean, std):
    """ Normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    # TODO: 1. Implement normalization doing channel-wise z-score normalization.
    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    img = transform_norm(img)
    return img


def content_loss(input_features, content_features, content_layers):
    """ Calculates the content loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @content_features, VGG features of the content image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @content_layers, a list containing which layers to consider for calculating
            the content loss.
    
    # Returns the content loss, a torch.tensor of size (1)
    """
    # TODO: 2. Implement the content loss given the input feature volume and the
    # content feature volume. Note that:
    # - Only the layers given in content_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.
    loss = torch.zeros(1)
    for n in content_layers:
        loss = loss + (pow((input_features[n] - content_features[n]), 2)).mean() / len(content_layers)

    return loss  # Initialize placeholder such that the code runs


def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # NOTE: Normalize by number of number of dimensions of the feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    # TODO: 3.2 Implement the calculation of the normalized gram matrix. 
    # Do not use for-loops, make use of Pytorch functionalities.
    b, c, h, w = x.size()
    features = x.view(b*c, h*w)
    g = torch.mm(features, features.t()).div(b*c*h*w)
    return g


def style_loss(input_features, style_features, style_layers):
    """ Calculates the style loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @style_features, VGG features of the style image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @style_layers, a list containing which layers to consider for calculating
            the style loss.
    
    # Returns the style loss, a torch.tensor of size (1)
    """
    # TODO: 3.1 Implement the style loss given the input feature volume and the
    # style feature volume. Note that:
    # - Only the layers given in style_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.
    # - Implement the gram_matrix function.
    # style_loss = torch.zeros(1)
    # for n in style_layers:
    #     style_loss = style_loss + ((input_features[n] - style_features[n])**2).mean() / len(style_layers)
    #
    # return style_loss  # Initialize placeholder such that the code runs
    loss = torch.zeros(1)
    for n in style_layers:
        loss = loss + (pow((gram_matrix(input_features[n]) - gram_matrix(style_features[n])), 2)).mean() / len(
            style_layers)
    return loss


def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)
    
    # Returns the total variation, a torch.tensor of size (1)
    """
    # TODO: 4. Implement the total variation loss. Normalize by tensor dimension sizes
    b, c, h, w = y.size()
    dh = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dw = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])

    tv_loss = (dh.sum() + dw.sum()) / (b * c * h * w)
    return tv_loss  # Initialize placeholder such that the code runs


def run_double_image(
        vgg_mean, vgg_std, content_img, style_img_1, style_img_2, num_steps,
        random_init, w_style_1, w_style_2, w_content, w_tv, content_layers, style_layers, device):
    # TODO: 5. Implement style transfer for two given style images.

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    # TODO: 1. Normalize Input images
    normed_style_img_1 = normalize(style_img_1, vgg_mean, vgg_std)
    normed_style_img_2 = normalize(style_img_2, vgg_mean, vgg_std)
    normed_content_img = normalize(content_img, vgg_mean, vgg_std)

    # Retrieve feature maps for content and style image
    # We do not need to calculate gradients for these feature maps
    with torch.no_grad():
        style_features_1 = model(normed_style_img_1)
        style_features_2 = model(normed_style_img_2)
        content_features = model(normed_content_img)

    # Either initialize the image from random noise or from the content image
    if random_init:
        optim_img = torch.randn(content_img.data.size(), device=device)
        optim_img = torch.nn.Parameter(optim_img, requires_grad=True)
    else:
        optim_img = torch.nn.Parameter(content_img.clone(), requires_grad=True)

    # Initialize optimizer and set image as parameter to be optimized
    optimizer = optim.LBFGS([optim_img])

    # Training Loop
    iter = [0]
    while iter[0] <= num_steps:

        def closure():

            # Set gradients to zero before next optimization step
            optimizer.zero_grad()

            # Clamp image to lie in correct range
            with torch.no_grad():
                optim_img.clamp_(0, 1)

            # Retrieve features of image that is being optimized
            normed_img = normalize(optim_img, vgg_mean, vgg_std)
            input_features = model(normed_img)

            # TODO: 2. Calculate the content loss
            if w_content > 0:
                c_loss = w_content * content_loss(input_features, content_features, content_layers)
            else:
                c_loss = 0

            # TODO: 3. Calculate the style loss
            if w_style_1 > 0:
                s_loss_1 = w_style_1 * style_loss(input_features, style_features_1, style_layers)
            else:
                s_loss_1 = 0
            if w_style_2 > 0:
                s_loss_2 = w_style_2 * style_loss(input_features, style_features_2, style_layers)
            else:
                s_loss_2 = 0

            # TODO: 4. Calculate the total variation loss
            if w_tv > 0:
                tv_loss = w_tv * total_variation_loss(normed_img)
            else:
                tv_loss = 0

            # Sum up the losses and do a backward pass
            loss = s_loss_1 + s_loss_2 + c_loss + tv_loss
            loss.backward()

            # Print losses every 50 iterations
            iter[0] += 1
            if iter[0] % 50 == 0:
                print('iter {}: | Style Loss: {:4f} | Content Loss: {:4f} | TV Loss: {:4f}'.format(
                    iter[0], s_loss_1.item(), s_loss_2.item(), c_loss.item(), tv_loss.item()))

            return loss

        # Do an optimization step as defined in our closure() function
        optimizer.step(closure)

    # Final clamping
    with torch.no_grad():
        optim_img.clamp_(0, 1)

    return optim_img
