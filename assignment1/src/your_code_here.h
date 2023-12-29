#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

#include "helpers.h"

/*
 * Utility functions.
 */

template<typename T>
int getImageOffset(const Image<T>& image, int x, int y)
{
    // Return offset of the pixel at column x and row y in memory such that 
    // the pixel can be accessed by image.data[offset].
    // 
    // Note, that the image is stored in row-first order, 
    // ie. is the order of [x,y] pixels is [0,0],[1,0],[2,0]...[0,1],[1,1][2,1],...
    //
    // Image size can be accessed using image.width and image.height.
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    const auto offset = y * image.width + x;
    return offset;
}


glm::vec2 getRGBImageMinMax(const ImageRGB& image) {

    auto min_val = 100.0f;
    auto max_val = 0.0f;
    
    // Write a code that will return minimum value (min of all color channels and pixels) and maximum value as a glm::vec2(min,max).
    
    // Note: Parallelize the code using OpenMP directives for full points.
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    #pragma omp parallel for 
    for (auto i = 0; i < image.data.size(); i++) {
        #pragma omp critical
        min_val = glm::min(glm::min(glm::min(image.data[i][0], image.data[i][1]), image.data[i][2]), min_val);
        #pragma omp critical
        max_val = glm::max(glm::max(glm::max(image.data[i][0], image.data[i][1]), image.data[i][2]), max_val);
    }

    // Return min and max value as x and y components of a vector.
    return glm::vec2(min_val, max_val);
}


ImageRGB normalizeRGBImage(const ImageRGB& image)
{
    // Create an empty image of the same size as input.
    auto result = ImageRGB(image.width, image.height);

    // Find min and max values.
    glm::vec2 min_max = getRGBImageMinMax(image);
    
    // Fill the result with normalized image values (ie, fit the image to [0,1] range).    
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    //#pragma omp parallel for
    for (auto i = 0; i < image.data.size(); i++) {
        result.data[i] = (image.data[i] - min_max[0]) / (min_max[1] - min_max[0]);
    }
    return result;
}

ImageRGB applyGamma(const ImageRGB& image, const float gamma)
{
    // Create an empty image of the same size as input.
    auto result = ImageRGB(image.width, image.height);

    // Fill the result with gamma mapped pixel values (result = image^gamma).    
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    for (auto i = 0; i < image.data.size(); i++) {
        for (auto j = 0; j < 3; j++) {
            result.data[i][j] = pow(image.data[i][j], gamma);
        }
    }
    return result;
}

/*
    Main algorithm.
*/

/// <summary>
/// Compute luminance from a linear RGB image.
/// </summary>
/// <param name="rgb">A linear RGB image</param>
/// <returns>log-luminance</returns>
ImageFloat rgbToLuminance(const ImageRGB& rgb)
{
    // RGB to luminance weights defined in ITU R-REC-BT.601 in the R,G,B order.
    const auto WEIGHTS_RGB_TO_LUM = glm::vec3(0.299f, 0.587f, 0.114f);
    // An empty luminance image.
    auto luminance = ImageFloat(rgb.width, rgb.height);
    // Fill the image by logarithmic luminace.
    // Luminance is a linear combination of the red, green and blue channels using the weights above.

    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    #pragma omp parallel for
        for (int i = 0; i < luminance.data.size(); i++) {
        luminance.data[i] = rgb.data[i].x * WEIGHTS_RGB_TO_LUM[0] + rgb.data[i].y * WEIGHTS_RGB_TO_LUM[1] + rgb.data[i].z * WEIGHTS_RGB_TO_LUM[2];
        }

    return luminance;
}




/// <summary>
/// Compute X and Y gradients of an image.
/// </summary>
/// <param name="H">H = log luminance</param>
/// <returns>grad H</returns>
ImageGradient getGradients(const ImageFloat& H)
{
    // Empty fields for X and Y gradients with the same sze as the image.
    auto grad_x = ImageFloat(H.width, H.height);
    auto grad_y = ImageFloat(H.width, H.height);

    for (auto y = 0; y < H.height; y++) {
        for (auto x = 0; x < H.width; x++) {
            // Compute X and Y gradients using right-sided forward differences:
            //      H = grad I = (I(x+1,y) - I(x, y), I(x,y+1) - I(x, y)
            // Assume zero padding => Pixels outside the image are 0 (black).
            //          => I(x,y) is zero if this pixel is outside of the image.
            // See the grad H equation of Section 5 in the paper for details.
    
            /*******
             * TODO: YOUR CODE GOES HERE!!!
             ******/
            if (x != H.width - 1 && y != H.height - 1) {
                grad_x.data[getImageOffset(grad_x, x, y)] = H.data[getImageOffset(H, x + 1, y)] - H.data[getImageOffset(H, x, y)];
                grad_y.data[getImageOffset(grad_x, x, y)] = H.data[getImageOffset(H, x, y + 1)] - H.data[getImageOffset(H, x, y)];
            } else if (x == H.width - 1 && y == H.height - 1) {
                grad_x.data[getImageOffset(grad_x, x, y)] = 0 - H.data[getImageOffset(H, x, y)];
                grad_y.data[getImageOffset(grad_x, x, y)] = 0 - H.data[getImageOffset(H, x, y)];
            } else if (x == H.width - 1) {
                grad_x.data[getImageOffset(grad_x, x, y)] = 0 - H.data[getImageOffset(H, x, y)];
                grad_y.data[getImageOffset(grad_x, x, y)] = H.data[getImageOffset(H, x, y + 1)] - H.data[getImageOffset(H, x, y)];
            } else if (y == H.height - 1) {
                grad_x.data[getImageOffset(grad_x, x, y)] = H.data[getImageOffset(H, x + 1, y)] - H.data[getImageOffset(H, x, y)];
                grad_y.data[getImageOffset(grad_x, x, y)] = 0 - H.data[getImageOffset(H, x, y)];
            }
        }
    }

    // Return both gradients in an ImageGradient struct.
    return ImageGradient(grad_x, grad_y);
}

/// <summary>
/// Computes gradient attenuation using the formula for phi_k in Sec. 4 of the paper.
/// </summary>
/// <param name="grad_H">gradients of H</param>
/// <param name="alpha_rel">alpha relative to mean grad norm</param>
/// <param name="beta">beta</param>
/// <returns>attenuation coefficients phi</returns>
ImageFloat getGradientAttenuation(const ImageGradient& grad_H, const float alpha_rel = 0.1f, const float beta = 0.35f)
{
    // EPSILON to add to the gradient norm to prevent division by zero.
    const float EPSILON = 1e-3f;
    // An empty gradient attenuation map phi.
    auto phi = ImageFloat(grad_H.x.width, grad_H.x.height);

    // Compute gradient attenuation using the formula for phi_k in Sec. 4 of the paper.
    // Step 1: Compute L2 norms of each XY gradient as grad_norm = sqrt(dx**2+dy**2)
    
    
    // Step 2: Compute mean norm of all gradients.
    float mean_grad = 0.0f;
    
    auto grad_norm = ImageFloat(grad_H.x.width, grad_H.x.height);
    for (auto i = 0; i < grad_norm.data.size(); i++) {
        grad_norm.data[i] = sqrt(pow(grad_H.x.data[i], 2) + pow(grad_H.y.data[i], 2));
        mean_grad += grad_norm.data[i];
    }
    mean_grad = mean_grad / grad_norm.data.size();
    // Step 3: Compute alpha = alpha_rel * mean_grad
    float alpha = alpha_rel * mean_grad;
    for (auto i = 0; i < grad_norm.data.size(); i++) {
        phi.data[i] = (alpha / (grad_norm.data[i] + EPSILON)) * pow(((grad_norm.data[i] + EPSILON) / alpha), beta);
    }
    // Step 4: Fill gradient attenuation field phi_k:
    //  phi_k = alpha / (grad_norm + eps) * ((grad_norm + eps) / alpha)^beta
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/


    return phi;
}

/// <summary>
/// Computes attenauted divergence div_G from attenuated gradients H as described in Sec. 5 of the paper.
/// </summary>
/// <param name="grad_H">(original) attenuated gradients of H</param>
/// <param name="phi">gradient attenuations phi</param>
/// <returns>div G</returns>
ImageFloat getAttenuatedDivergence(ImageGradient& grad_H, const ImageFloat& phi) {

    // An empty divergence field with the same size as the input.
    auto div_G = ImageFloat(phi.width, phi.height);
    // Compute attenauted divergence div_G from attenuated gradients H as described in Sec. 5 of the paper.
    // 1. Compute attentuated gradients G:
    //     G = H * phi
    // 2. Compute divergence of G using formula in Sec. 5 of the paper:
    //    div G = (G_x(x,y) - G_x(x-1,y)) + (G_y(x,y) - G_y(x,y-1))
    //
    // Assume zero padding => Pixels outside the image are 0 (black)
    //  => G_x(x,y) is zero if this pixel is outside of the image.
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    auto G_x = ImageFloat(phi.width, phi.height);
    auto G_y = ImageFloat(phi.width, phi.height);
    for (auto i = 0; i < grad_H.x.data.size(); i++) {
        G_x.data[i] = grad_H.x.data[i] * phi.data[i];
        G_y.data[i] = grad_H.y.data[i] * phi.data[i];
    }
    for (auto i = 0; i < grad_H.x.width; i++) {
        for (auto j = 0; j < grad_H.x.height; j++) {
            if (i != 0 && j != 0) {
                div_G.data[getImageOffset(phi, i, j)] = G_x.data[getImageOffset(phi, i, j)] - G_x.data[getImageOffset(phi, i - 1, j)] +
                    G_y.data[getImageOffset(phi, i, j)] - G_y.data[getImageOffset(phi, i, j-1)];
            } else if (i == 0 && j == 0) {
                div_G.data[getImageOffset(phi, i, j)] = G_x.data[getImageOffset(phi, i, j)] - 0 + 
                    G_y.data[getImageOffset(phi, i, j)] - 0;
            } else if (i == 0 && j != 0) {
                div_G.data[getImageOffset(phi, i, j)] = G_x.data[getImageOffset(phi, i, j)] - 0 + 
                    G_y.data[getImageOffset(phi, i, j)] - G_y.data[getImageOffset(phi, i, j - 1)];
            } else if (i != 0 && j == 0) {
                div_G.data[getImageOffset(phi, i, j)] = G_x.data[getImageOffset(phi, i, j)] - G_x.data[getImageOffset(phi, i - 1, j)] + 
                    G_y.data[getImageOffset(phi, i, j)] - 0;
            }
        }
    }

    return div_G;
}

/// <summary>
/// Solves poisson equation in form grad^2 I = div G.
/// </summary>
/// <param name="divergence_G">div G</param>
/// <param name="num_iters">number of iterations</param>
/// <returns>luminance I</returns>
ImageFloat solvePoisson(const ImageFloat& divergence_G, const int num_iters = 2000)
{
    // Empty solution.
    auto I = ImageFloat(divergence_G.width, divergence_G.height);
    std::fill(I.data.begin(), I.data.end(), 0.0f);

    // Another solution for the alteranting updates.
    auto I_next = ImageFloat(divergence_G.width, divergence_G.height);

    // Iterative solver.
    for (auto iter = 0; iter < num_iters; iter++)
    {
        if (iter % 500 == 0) {
            // Print progress info every 500 iteartions.
            std::cout << "[" << iter << "/" << num_iters << "] Solving Poisson equation..." << std::endl;
        }

        // Implement one step of the iterative Euler solver:
        //      I_next = ((I[x-1, y] + I[x+1, y] + I[x, y-1] + I[x, y+1]) - div_G[x,y]) / 4
        
        // Assume zero padding => Pixels outside the image are 0 (black).
        //     Eg:  I(-1,2) => 0

        // Note: Parallelize the code using OpenMP directives for full points.
    
        /*******
         * TODO: YOUR CODE GOES HERE!!!
         ******/
        #pragma omp parallel for
        for (auto i = 0; i < divergence_G.width; i++) {
        #pragma omp parallel for
            for (auto j = 0; j < divergence_G.height; j++) {
                if (i == 0 && j != 0 && j != divergence_G.height - 1) {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i + 1, j)]
                                                                          + I.data[getImageOffset(divergence_G, i, j - 1)] + I.data[getImageOffset(divergence_G, i, j + 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i == divergence_G.width - 1 && j != 0 && j != divergence_G.height - 1)
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i - 1, j)]
                                                                          + I.data[getImageOffset(divergence_G, i, j - 1)] + I.data[getImageOffset(divergence_G, i, j + 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i != 0 && i != divergence_G.width - 1 && j == 0)
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i + 1, j)]
                                                                          + I.data[getImageOffset(divergence_G, i - 1, j)] + I.data[getImageOffset(divergence_G, i, j + 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i != 0 && i != divergence_G.width - 1 && j == divergence_G.height - 1) {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i + 1, j)]
                                                                          + I.data[getImageOffset(divergence_G, i - 1, j)] + I.data[getImageOffset(divergence_G, i, j - 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i == 0 && j == 0)
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i + 1, j)] + I.data[getImageOffset(divergence_G, i, j + 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i == 0 && j == divergence_G.height - 1)
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i + 1, j)] + I.data[getImageOffset(divergence_G, i, j - 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i == divergence_G.width - 1 && j == 0)
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i - 1, j)] + I.data[getImageOffset(divergence_G, i, j + 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else if (i == divergence_G.width - 1 && j == divergence_G.height - 1)
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i - 1, j)] + I.data[getImageOffset(divergence_G, i, j - 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
                else
                {
                    I_next.data[getImageOffset(divergence_G, i, j)] = (I.data[getImageOffset(divergence_G, i - 1, j)] + I.data[getImageOffset(divergence_G, i + 1, j)]
                                                                          + I.data[getImageOffset(divergence_G, i, j - 1)] + I.data[getImageOffset(divergence_G, i, j + 1)]
                                                                          - divergence_G.data[getImageOffset(divergence_G, i, j)]) / 4;
                }
            }
        }

        // Swaps the current and next solution so that the next iteration
        // uses the new solution as input and the previous solution as output.
        std::swap(I, I_next);
    }

    // After the last "swap", I is the latest solution.
    return I;
}

ImageRGB rescaleRgbByLuminance(const ImageRGB& original_rgb, const ImageFloat& original_luminance, const ImageFloat& new_luminance, const float saturation = 0.5f)
{
    // EPSILON for thresholding the divisior.
    const float EPSILON = 1e-7f;
    // An empty RGB image for the result.
    auto result = ImageRGB(original_rgb.width, original_rgb.height);

    // Return the original_rgb rescaled to match the new luminance as in Sec. 5 of the paper:
    //
    //      result = (original_rgb / max(original_luminance, epsilon))^saturation * new_luminance

    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/
    for (auto i = 0; i < original_rgb.data.size(); i++) {
        auto temp_luminance = original_luminance.data[i];
        for (auto j = 0; j < 3; j++) {
                result.data[i][j] = pow((original_rgb.data[i][j] / std::max(temp_luminance, EPSILON)), saturation) * new_luminance.data[i];
        }
    }


    return result;
}