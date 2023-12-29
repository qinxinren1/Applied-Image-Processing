#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "helpers.h"


/*
 * Utility functions.
 */

/// <summary>
/// Bilinearly sample ImageFloat or ImageRGB using image coordinates [x,y].
/// Note that the coordinate start and end points align with the entire image and not just
/// the pixel centers (a half-pixel difference).
/// </summary>
/// <typeparam name="T">template type, can be ImageFloat or ImageRGB</typeparam>
/// <param name="image">input image</param>
/// <param name="pos">x,y position in px units</param>
/// <returns>interpolated pixel value (float or glm::vec3)</returns>
template <typename T>
inline T sampleBilinear(const Image<T>& image, const glm::vec2& pos_px)
{
    // Write a code that bilinearly interpolates values from a generic image (can contain either float or glm::vec3).
    // The pos_px input represents the (x,y) pixel coordinates of the sampled point where:
    //   [0, 0] = The left top corner of the left top (=first) pixel.
    //   [width, height] = The right bottom corner of the right bottom (=last) pixel.
    //   [0, height] = The left bottom corner of the left bottom pixel.
    //
    // Take into account the size of individual pixel and the fact, that the "value" of pixel is conceptually stored in its center.
    //      => Example 1: For pos_px between centers of pixels, the method bilinearly interpolates between 4 nearest pixel.
    //      => Example 2: For pos_px corresponding to a center of a pixel, the method needs to return an exact value of that pixel.
    //                    This is a natural property of any interpolation.
    //
    // Therefore, steps are as follows:
    //     1. Determine the 4 nearest pixels.
    //     2. Bilinearly interpolate their values based on the position of the sampling point between them.
    //
    // Note: The method is templated by parameter "T". This will be either float or glm::vec3 depending on whether the method
    // is called with ImageFloat or ImageRGB. Use either "T" or "auto" to define your variables and use glm::functions to handle both types.
    // Example:
    //    auto value = image.data[0] * 3; // both float and glm:vec3 support baisc operators
    //    T rounded_value = glm::round(image.data[0]); // glm::round will handle both glm::vec3 and float.
    // Use glm API for further reference: https://glm.g-truc.net/0.9.9/api/a00241.html
    //

    //
    //    YOUR CODE GOES HERE
    //
    int x = static_cast<int>(pos_px.x);
    int y = static_cast<int>(pos_px.y);
    int x1 = glm::round(pos_px.x);
    int y1 = glm::round(pos_px.y);
    int x2, y2;
    if (x == image.width - 1 && y != image.height - 1) {
        x1 = x - 1;
        y1 = glm::max(1, y1) - 1;
    } else if (y == image.height - 1 && x != image.width - 1) {
        y1 = y - 1;
        x1 = glm::max(1, x1) - 1;
    } else if (x1 == image.width - 1 && y1 == image.height - 1) {
        x1 = x1 - 1;
        y1 = y1 - 1;
    }
    else if (y == image.height - 1 && x == image.width - 1) {
        x1 = x - 1;
        y1 = y - 1;
    }
    else {
        x1 = glm::max(1, x1) - 1;
        y1 = glm::max(1, y1) - 1;
    }
    x2 = glm::min(x1 + 1, image.width - 1);
    y2 = glm::min(y1 + 1, image.height - 1);
    float dx = glm::abs(pos_px.x - x1);
    float dy = glm::abs(pos_px.y - y1);

    T q11 = image.data[y1 * image.width + x1];
    T q12 = image.data[y2 * image.width + x1];
    T q21 = image.data[y1 * image.width + x2];
    T q22 = image.data[y2 * image.width + x2];


    T result = (1 - dx) * (1 - dy) * q11 + dx * (1 - dy) * q21 + (1 - dx) * dy * q12 + dx * dy * q22;

    return result;
}


/*
  Core functions.
*/

/// <summary>
/// Applies the bilateral filter on the given disparity image.
/// Ignored pixels that are marked as invalid.
/// </summary>
/// <param name="disparity">The image to be filtered.</param>
/// <param name="guide">The image guide used for calculating the tonal distances between pixel values.</param>
/// <param name="size">The kernel size, which is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageFloat, the filtered disparity.</returns>
ImageFloat jointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int size, const float guide_sigma)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // We assume both images have matching dimensions.
    assert(disparity.width == guide.width && disparity.height == guide.height);

    // Rule of thumb for gaussian's std dev. 
    const float sigma = (size - 1) / 2 / 3.2f;

    // Empty output image.
    auto result = ImageFloat(disparity.width, disparity.height);

    //
    // Implement a bilateral filter of the disparity image guided by the guide.
    // Ignore all contributing pixels where disparity == INVALID_VALUE.
    //
    // 1. Iterate over all output pixels.
    // 2. For each output pixel, visit all neighboring pixels (including itself) in a size x size symmetric neighborhood. 
    //    That is the same as during convolution with a size x size symmetric filter centered around the pixel.
    // 3. If a neighbor is outside of the image or its disparity == INVALID_VALUE, skip the pixel (move to the next one).
    // 4. For each neighbor compute its weight as w_i = gauss(dist, sigma) * gauss(diff_value, guide_sigma)
    //      where
    //          * dist is the Eucledian distance between the center and current pixel in pixels (L2 norm).
    //          * diff_value is the L2 (euclidean) distance of the guide image pixel values for the center and current pixel.
    //          * gauss(x, sigma) is a Normal distribution pdf function for x=x and std.dev.= sigma which is available for use.
    // 5. Compute weighted mean of all (unskipped) neighboring pixel disparities and assign it to the output.
    // 
    // Refer to the first Lecture or recommended study books for more info on Bilateral filtering.
    // 
    // Notes:
    //   * If a pixel has no neighbor (all were skipped), assign INVALID_VALUE to the output.  
    //   * One point awarded for a correct OpenMP parallelization.

    //
    //    YOUR CODE GOES HERE
    //
    int width = disparity.width;
    int height = disparity.height;
    int neighbor_size = size / 2;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            auto weighted_sum = 0.0f;
            auto weight_accumulator = 0.0f;
            auto pixel_position = y * width + x;
            auto center_value = guide.data[pixel_position];
            for (int dy = y - neighbor_size; dy <= y + neighbor_size; dy++) {
                for (int dx = x - neighbor_size; dx <= x + neighbor_size; dx++) {
                    //if the neighbor is in the boundary
                    auto neighbor_position = dy * width + dx;
                    if (dx >= 0 && dy >= 0 && dx < width && dy < height && disparity.data[neighbor_position] != INVALID_VALUE) {
                        auto neighbor_value = guide.data[neighbor_position];
                        auto spatial_weight = gauss(glm::sqrt((dx - x) * (dx - x) + (dy - y) * (dy - y)), sigma);
                        auto intensity_weight = gauss(glm::length(center_value - neighbor_value), guide_sigma);
                        float weight = spatial_weight * intensity_weight;
                        weighted_sum += disparity.data[neighbor_position] * weight;
                        weight_accumulator += weight;
                    }
                }
            }
            result.data[pixel_position] = (weighted_sum > 0.0f) ? weighted_sum / weight_accumulator : INVALID_VALUE;
        }
    }
    //auto example = gauss(0.5f, 1.2f); // This is just an example of computing Normal pdf for x=0.5 and std.dev=1.2.


    // Return filtered disparity.
    return result;
}

/// <summary>
/// In-place normalizes and an ImageFloat image to be between 0 and 1.
/// All values marked as invalid will stay marked as invalid.
/// </summary>
/// <param name="scalar_image"></param>
/// <returns></returns>
void normalizeValidValues(ImageFloat& scalar_image)
{
    //
    // Find minimum and maximum among the VALID image values.
    // Linearly rescale the VALID image values to the [0,1] range (in-place).
    // The INVALID values remain INVALID (they are ignored).
    // 
    // Note #1: Pixel is INVALID as long as value == INVALID_VALUE.
    // Note #2: This modified the input image in-place => no "return".
    //
    
    //
    //    YOUR CODE GOES HERE
    //
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::min();
    int total_pixels = scalar_image.data.size();
    for (int i = 0; i < total_pixels; i++) {
        auto pixel_value = scalar_image.data[i];
        if (pixel_value != INVALID_VALUE) {
            min_value = std::min(min_value, pixel_value);
            max_value = std::max(max_value, pixel_value);
        }
    }
    if (max_value == min_value) {
        return;
    }
    for (int i = 0; i < total_pixels; i++) {
        auto pixelValue = scalar_image.data[i];
        if (pixelValue != INVALID_VALUE) {
            scalar_image.data[i] = (pixelValue - min_value) / (max_value - min_value);
        }
    }
}

/// <summary>
/// Converts a disparity image to a normalized depth image.
/// Ignores invalid disparity values.
/// </summary>
/// <param name="disparity">disparity in arbitrary units</param>
/// <returns>linear depth scaled from 0 to 1</returns>
ImageFloat disparityToNormalizedDepth(const ImageFloat& disparity)
{
    auto depth = ImageFloat(disparity.width, disparity.height);

    //
    // Convert disparity to a depth with unknown scale:
    //    depth_unscaled = 1.0 / disparity
    // If disparity of a pixel is invalid, set its depth also invalid (INVALID_VALUE).
    // We guarantee that all valid disparities > 0.
    //
        
    //
    //    YOUR CODE GOES HERE
    //
    int total_pixels = disparity.data.size();
    for (int i = 0; i < total_pixels; i++) {
        if (disparity.data[i] != INVALID_VALUE && disparity.data[i] > 0) {
            depth.data[i] = 1.0f / disparity.data[i];
        } else {
            depth.data[i] = INVALID_VALUE;
        }
    }
    // Rescales valid depth values to [0,1] range.
    normalizeValidValues(depth);

    return depth;
}

/// <summary>
/// Convert linear normalized depth to target pixel disparity.
/// Invalid pixels 
/// </summary>
/// <param name="depth">Normalized depth image (values in [0,1])</param>
/// <param name="iod_mm">Inter-ocular distance in mm.</param>
/// <param name="px_size_mm">Pixel size in mm.</param>
/// <param name="screen_distance_mm">Screen distance from eyes in mm.</param>
/// <param name="near_plane_mm">Near plane distance from eyes in mm.</param>
/// <param name="far_plane_mm">Far plane distance from eyes in mm.</param>
/// <returns>screen disparity in pixels</returns>
ImageFloat normalizedDepthToDisparity(
    const ImageFloat& depth, const float iod_mm,
    const float px_size_mm, const float screen_distance_mm,
    const float near_plane_mm, const float far_plane_mm)
{
    auto px_disparity = ImageFloat(depth.width, depth.height);

    //
    // Based on physical dimensions, distance, resolution (and hence pixel size) of the screen,
    // as well as physiologically determined distance between viewers pupil (IOD or IPD),
    // compute stereoscopic pixel disparities that will make the display appear at a correct depth
    // represented by linear interpolation between the near and far plane based on the depth input image.
    // 
    // Refer to Lecture 4 for formulas.
    // 
    // Example:
    //    screen distance = 600 mm, near_plane_mm = 550, far_plane == 650, depth = 0.1  
    //         => the pixel should appear 55+0.1(65-55) = 56 cm away from the user
    //         => That is 4 cm in front of the screen.
    //         => That means the pixel disparity will be a negative number ("crossed disparity").
    // 
    // Note:
    //    * All distances are measured orthogonal on the screen and are assumed constant across the screen (ignores the eccentricity variance).
    //    * Invalid pixels (depth==INVALID_VALUE) are to be marked invalid on the output as well.
    //
    
    //
    //    YOUR CODE GOES HERE
    //
    int total_pixels = depth.data.size();
    for (int i = 0; i < total_pixels; i++) {
        if (depth.data[i] != INVALID_VALUE) {
            float user_distance_mm = near_plane_mm + depth.data[i] * (far_plane_mm - near_plane_mm);
            float front_of_screen_mm = user_distance_mm - screen_distance_mm;
            px_disparity.data[i] = (iod_mm * front_of_screen_mm) / (screen_distance_mm * px_size_mm);            
        } else {
            px_disparity.data[i] = INVALID_VALUE;
        }
    }

    return px_disparity; // returns disparity measured in pixels
}

/// <summary>
/// Creates a warping grid for an image of specified height and weight.
/// It produces vertex buffer which stores 2D positions of pixel corners,
/// and index buffer which defines triangles by triplets of indices into
/// the vertex buffer (the three vertices form a triangle).
/// 
/// </summary>
/// <param name="width">Image width.</param>
/// <param name="height">Image height.</param>
/// <returns>Mesh, containing a vertex buffer and triangle index buffer.</returns>
Mesh createWarpingGrid(const int width, const int height)
{

    // Build vertex buffer.
    auto num_vertices = (width + 1) * (height + 1);
    auto vertices = std::vector<glm::vec2>(num_vertices);

    //
    // Fill the vertex buffer (vertices) with 2D coordinate of the pixel corners.
    // Expected output coordinates are:
    //   [0,0] for the left top corner of the left top (=first) pixel.
    //   [width,height] for the right bottom corner of the right bottom (=last) pixel.
    //   [0,height] for the left bottom corner of the left bottom pixel.
    //
    // The order in memory is to be the same as for images (row after row).
    //

    //
    //    YOUR CODE GOES HERE
    //
    for (int y = 0; y <= height; y++) {
        for (int x = 0; x <= width; x++) {
            vertices[(width + 1) * y + x] = glm::vec2(x, y);
        }
    }
    // Build index buffer.
    auto num_pixels = width * height;
    auto num_triangles = num_pixels * 2;
    auto triangles = std::vector<glm::ivec3>(num_triangles);

    //
    // Fill the index buffer (triangles) with indices pointing to the vertex buffer.
    // Each element of the "triangles" is an integer triplet (glm::ivec3).
    // It represents a triangle by selecting 3 vertices from the vertex buffer defining its corners in
    // a clockwise manner.
    // We need to fill the index buffer in the same order as pixels are stored in memory (that is row by row)
    // and for each pixel we should generate two triangles that together cover the are of a pixel as follows:
    //
    //   A ------- B
    //   |  *      |
    //   |    *    |
    //   |      *  |
    //   D ------- C
    //
    // Where A,B,C,D are the CORNERS of the respective pixel.
    //
    // For each such pixel, we add two triangles:
    //     glm::ivec3(A,B,C) and glm::ivec3(A,C,D)  (in this exact order)
    // where A,B,C,D are indices into the vertex buffer.
    //
    // The result should be a grid that fills an entire image and replaces each pixel with two small triangles.
    //
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //int index = (width + 1) * y + x;
            int a = (width + 1) * y + x;
            int b = (width + 1) * y + x + 1;
            int c = (width + 1) * (y + 1) + x + 1;
            int d = (width + 1) * (y + 1) + x;
            triangles[(y * width + x) * 2] = glm::ivec3(a, b, c);
            triangles[(y * width + x) * 2 + 1] = glm::ivec3(a, c, d);
        }
    }
    //
    //    YOUR CODE GOES HERE
    //
    // Combine the vertex and index buffers into a mesh.
    return Mesh { std::move(vertices), std::move(triangles) };
}

/// <summary>
/// Warps a grid based on the given disparity and scaling_factor.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="disparity">Disparity for each PIXEL.</param>
/// <param name="scaling_factor">Global scaling factor for the disparity.</param>
/// <returns>Mesh, the warped grid.</returns>
Mesh warpGrid(Mesh& grid, const ImageFloat& disparity, const float scaling_factor, const BilinearSamplerFloat& sampleBilinear)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float EDGE_EPSILON = 1e-5f * disparity.width;

    //
    // The goal is to modify the x coordinate of the grid vertices based on 
    // the scaled pixel disparity corresponding to the original location of the vertex buffer:
    // 
    // new_grid.vertex.x = grid.vertex.x + scaling_factor * sampled_disparity
    // 
    // where sampled_disparity is a bilinearly interpolated value from the disparity image
    // which we can easily obtained using our other function "sampleBilinear":
    //     sampled_disparity = sampleBilinear(disparity, grid.vertex)
    // 
    // IMPORTANT - in order to keep the grid attached to the image "frame",
    // we must not move the border vertices => do not move vertices
    // that are within EDGE_EPSILON from the image boundary 
    // OLD WRONG: in either x or y direction.
    // CORRECTION (Oct 11): in x direction. 
    // That means the grid is attached to left and right boundaries but can freely slide along the top/down boundaries.
    //

    // Here is an example use of the bilinear interpolation (using the provided function argument).
    // auto interpolated_value = sampleBilinear(disparity, glm::vec2(1.0f, 1.0f));
    // Recommended test: For a 2x2 image it SHOULD return the mean of the 4 pixels.
    
    //
    //    YOUR CODE GOES HERE
    //
    int width = disparity.width;
    int height = disparity.height;
    auto LEFT_BOUNDARY = EDGE_EPSILON;
    auto RIGHT_BOUNDARY = width - EDGE_EPSILON;
    for (int i = 0; i < grid.vertices.size(); i++) {
        if (grid.vertices[i].x < RIGHT_BOUNDARY && grid.vertices[i].x > LEFT_BOUNDARY) {
            float sampled_disparity = sampleBilinear(disparity, grid.vertices[i]);
            new_grid.vertices[i].x = grid.vertices[i].x + scaling_factor * sampled_disparity;
        }
    }
    return new_grid;
}



/// <summary>
/// Forward-warps an image based on the given disparity and warp_factor.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="disparity">Disparity of the source image in pixels.</param>
/// <param name="warp_factor">Multiplier of the disparity.</param>
/// <returns>ImageWithMask, containing the forward-warped image and a mask image. Mask=1 for valid pixels, Mask=0 for holes</returns>
ImageWithMask forwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor)
{
    // The dimensions of src image, src depth and disparity maps all match.
    assert(src_image.width == disparity.width && src_image.height == disparity.height);
    assert(src_image.width == disparity.width && src_depth.height == src_depth.height);
    
    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_mask = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth mask map with zero.
    std::fill(dst_mask.data.begin(), dst_mask.data.end(), 0.0f);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), std::numeric_limits<float>::max());

    // 
    // The goal is to forward warp the image pixels using the disparity displacement provided in 
    // the disparity map along with additional scaling factor. Furthermore, we
    // use depth information to resolve conflicts when multiple pixels attempt
    // to write a single output pixel. 
    //
    // 1. For every input pixel, compute where it should be warped to 
    //    based on the associated disparity and warp_factor. 
    //    Use standard rounding rules to obtain integer position (ie., 0.5 rounds up).
    //      x' = rounding_function(x + disparity * warp_factor)
    //      y' = y
    // 
    // 2. Check the destination depth at the [x',y'] location and compare it with the 
    //    depth of the currently warped pixel (ie., depth[x,y]).
    //
    // 3. If the currently warped pixel has a depth larger or equal to the previous value in the output depth buffer,
    //    stop here and continue with step 1 for the next pixel.
    //
    // 4. Overwrite the output buffers. This means writing:
    //    - the destination image
    //    - the destination depth map
    //    - the mask (mask->1)
    //
    // 
    // Note: Point(s) awarded for a correct and efficient parallel solution using OpenMP.
    //
    
    //
    //    YOUR CODE GOES HERE
    //
    int width = src_image.width;
    int height = src_image.height;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float disp = disparity.data[y * width + x] * warp_factor;
            int x_warp_position = std::round(x + disp);
            if (0 <= x_warp_position && x_warp_position < width) {
                float current_depth = src_depth.data[y * width + x];
                float& destination_depth = dst_depth.data[y * width + x_warp_position];
#pragma omp critical
                if (current_depth < destination_depth) {
                    dst_image.data[y * width + x_warp_position] = src_image.data[y * width + x];
                    destination_depth = current_depth;
                    dst_mask.data[y * width + x_warp_position] = 1.0f;
                }
            }
        }
    }

    // Return the warped image.
    return ImageWithMask(dst_image, dst_mask);

}


/// <summary>
/// Applies the bilateral filter on the given image to fill the holes
/// indicated by a binary mask (mask==0 -> missing pixel).
/// Keeps the pixels not marked as holes unchanged.
/// </summary>
/// <param name="img_forward">The image to be filtered and its mask.</param>
/// <param name="size">The kernel size. It is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageRGB, the filtered forward warping image.</returns>
ImageRGB inpaintHoles(const ImageWithMask& img, const int size)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // Rule of thumb for gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;

    // The output is initialized by copy of the input.
    auto result = ImageRGB(img.image);
    

    // The goal is to fill the holes in forward warping image using a bilateral filter where the mask serves as a guide.
    
    // 1. For valid pixels (mask >= 0.5) -> do nothing and keep the valid pixel.
    // 2. For invalid pixels -> replace pixel with:
    //      Filter valid neighbors in [size x size] in a symmetric neighborhood 
    //      with a Gaussian filter weight w_i = gauss(dist, sigma) where
    //          * dist is the Euclidean distance between the center and current pixel in pixels (L2 norm).
    //          * gauss(x, sigma) is a Normal distribution pdf function for x=x and std.dev.= sigma which is available for use.
    //      !!!! SKIP invalid neighbors (mask < 0.5) !!!!
    // 
    // Notes:
    //   * Keep pixels with no valid neighbors unmodified.

    //
    //    YOUR CODE GOES HERE
    //
    int width = img.image.width;
    int height = img.image.height;
    int half_size = size / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_position = y * img.image.width + x;
            if (img.mask.data[pixel_position] >= 0.5) {
                continue;
            } else {
                float sum_weights = 0.0f;
                glm::vec3 sum_colors = { 0.0f, 0.0f, 0.0f};
                for (int dy = -half_size; dy <= half_size; dy++) {
                    for (int dx = -half_size; dx <= half_size; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && img.mask.data[ny * width + nx] >= 0.5) // Valid neighbor
                        {
                            float dist = sqrt(dx * dx + dy * dy);
                            float weight = gauss(dist, sigma);
                            sum_weights += weight;
                            sum_colors += weight * img.image.data[ny * width + nx];
                        }
                    }
                }
                if (sum_weights > 0.0f) {
                    result.data[pixel_position] = sum_colors / sum_weights;
                }
            }
        }
    }

    // Return inpainted image.
    return result;
}

/// <summary>
/// Backward-warps an image using a warped mesh.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="src_grid">Source grid.</param>
/// <param name="dst_grid">The warped grid.</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>ImageRGB, the backward-warped image.</returns>
ImageRGB backwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const Mesh& src_grid, const Mesh& dst_grid,
    const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{
    // The dimensions of src image and depth match.
    assert(src_image.width == src_depth.width && src_image.height == src_depth.height);
    // We assume that both grids have the same size and also the same order (ie., there is 1:1 triangle pairing).
    // This implies that the content of index buffers of both meshes are exactly equal (we do not test it here).
    assert(src_grid.triangles.size() == dst_grid.triangles.size());

    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), 1e20f);

    //
    // This method implements mesh-based warping by rasterizing
    // each triangle of the destination grid and sampling
    // the source texture by looking up corresponding 
    // position in the source grid using barycentric coordinates.
    // 
    // 1. For every triangle in the warped grid (dst_grid), 
    //    determine X and Y ranges of destination pixel coordinates
    //    whose **centers** lie in the bounding box of the triangle.
    //    Note: Look back to createWarpingGrid() for definition
    //          of the grid vertex coordinates.
    // 
    // 2. Enumerate all candidate pixels within the bounding box.
    //    Skip pixels whose centers actually do not lie inside the triangle.
    //    Use the provided isPointInsideTriangle(pt_dst, vert_a, vert_b, vert_c)
    //    method to do the test.
    // 
    // 3. Compute barycentric coordinates of the pixel center in the triangle
    //    using the provided method bc = barycentricCoordinates(pt_dst, vert_a, vert_b, vert_c).
    // 
    // 4. Find the corresponding triangle in the original mesh. It has the same triangle index.
    // 
    // 5. Reproject the pixel center to the source grid by using the already computed barycentric
    //    coordinates:
    //        pt_src = (vert_a_src, vert_b_src, vert_c_src) * bc
    // 
    // 6. Bilinearly sample both the source depth map and the source image values at the pt_src
    //    position. 
    //    Hint: Use the sampleBilinear[RGB]() method implemented earlier (using the provided function arguments).
    // 
    // 7. Compare the depth in the source and destination depth values and implement the depth-test
    //    like in fowardWarpImage().
    //    Hint: The destination map can be accessed directly without interpolation because we are
    //          computing for an exact pixel coordinate.
    // 
    // 8. If the depth test passes, write out the destination image (dst_image) and depth (dst_depth) values. 
    //    Again, it is the same logic as in fowardWarpImage().
    // 
    // 9. Return the warped image (dst_image).
    //
    
    // Example of testing point [0.1, 0.2] is inside a triangle.
    bool is_point_inside = isPointInsideTriangle(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    // Example of computing barycentric coordinates of a point [0.1, 0.2] inside a triangle.
    glm::vec3 bc = barycentricCoordinates(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    //
    //    YOUR CODE GOES HERE
    //

    for (int i = 0; i < dst_grid.triangles.size(); i++) {
        // 1
        auto vert_a_dst = dst_grid.vertices[dst_grid.triangles[i].x];
        auto vert_b_dst = dst_grid.vertices[dst_grid.triangles[i].y];
        auto vert_c_dst = dst_grid.vertices[dst_grid.triangles[i].z];

        auto x_min = glm::min(vert_a_dst.x, glm::min(vert_b_dst.x, vert_c_dst.x));
        auto y_min = glm::min(vert_a_dst.y, glm::min(vert_b_dst.y, vert_c_dst.y));
        auto x_max = glm::max(vert_a_dst.x, glm::max(vert_b_dst.x, vert_c_dst.x));
        auto y_max = glm::max(vert_a_dst.y, glm::max(vert_b_dst.y, vert_c_dst.y));

        auto bb_min = glm::vec2(glm::floor(x_min), glm::floor(y_min));
        auto bb_max = glm::vec2(glm::ceil(x_max), glm::ceil(y_max));

        bb_min.x = glm::max(static_cast<int>(bb_min.x), 0);
        bb_min.y = glm::max(static_cast<int>(bb_min.y), 0);
        bb_max.x = glm::min(static_cast<int>(bb_max.x), dst_image.width - 1);
        bb_max.y = glm::min(static_cast<int>(bb_max.y), dst_image.height - 1);

        // 2
        for (int y = bb_min.y; y <= bb_max.y; y++) {
            for (int x = bb_min.x; x <= bb_max.x; x++) {
                auto pixel = glm::vec2(x + 0.5, y + 0.5);
                if (isPointInsideTriangle(pixel, vert_a_dst, vert_b_dst, vert_c_dst)) {
                    // 3
                    auto bc = barycentricCoordinates(pixel, vert_a_dst, vert_b_dst, vert_c_dst);
                    // 4
                    auto vert_a_src = src_grid.vertices[src_grid.triangles[i].x];
                    auto vert_b_src = src_grid.vertices[src_grid.triangles[i].y];
                    auto vert_c_src = src_grid.vertices[src_grid.triangles[i].z];
                    // 5
                    auto pt_src = vert_a_src * bc.x + vert_b_src * bc.y + vert_c_src * bc.z;
                    // 6
                    auto src_depth_value = sampleBilinear(src_depth, pt_src);
                    auto src_color_value = sampleBilinearRGB(src_image, pt_src);
                    // 7
                    if (src_depth_value < dst_depth.data[y * dst_depth.width + x]) {
                        //8
                        dst_image.data[y * dst_depth.width + x] = src_color_value;
                        dst_depth.data[y * dst_depth.width + x] = src_depth_value;
                    }
                }
            }
        }
    }

    // Return the warped image.
    return dst_image;
}

/// <summary>
/// Returns an anaglyph image.
/// </summary>
/// <param name="image_left">left RGB image</param>
/// <param name="image_right">right RGB image</param>
/// <param name="saturation">color saturation to apply</param>
/// <returns>ImageRGB, the anaglyph image.</returns>
ImageRGB createAnaglyph(const ImageRGB& image_left, const ImageRGB& image_right, const float saturation)
{
    // An empty image for the resulting anaglyph.
    auto anaglyph = ImageRGB(image_left.width, image_left.height);

    // 
    // Convert stereoscopic pair into a single anaglyph stereoscopic image
    // for viewing in red-cyan anaglyph glasses.
    // We additionally scale saturation of the image to make the image
    // more "grayscale" since colors are problematic in analglyph image
    // and increase crosstalk (ghosting).
    // 
    // For both left and rigt image:
    // 1. Convert RGB to HSV color space using the provided rgbToHsv() function.
    // 2. Scale the saturation (stored in the second (=Y) component of the vec3) by the "saturation" param.
    // 3. Convert back to RGB using the hsvToRgb().
    // 
    // Combine the two images such that:
    //    * output.red = left.red
    //    * output.green = right.green
    //    * output.blue = right.blue.
    //

    // Example: RGB->HSV->RGB should be approx identity.
    auto rgb_orig = glm::vec3(0.2, 0.6, 0.4);
    auto rgb_should_be_same = hsvToRgb(rgbToHsv(rgb_orig)); // expect rgb == rgb_2 (up to numerical precision)

    //
    //    YOUR CODE GOES HERE
    //
    for (int i = 0; i < image_left.data.size(); i++) {
        auto hsv_left = rgbToHsv(image_left.data[i]);
        auto hsv_right = rgbToHsv(image_right.data[i]);
        hsv_left.y *= saturation;
        hsv_right.y *= saturation;

        auto rgb_left = hsvToRgb(hsv_left);
        auto rgb_right = hsvToRgb(hsv_right);

        anaglyph.data[i].r = rgb_left.r;
        anaglyph.data[i].g = rgb_right.g;
        anaglyph.data[i].b = rgb_right.b;
    }

    // Returns a single analgyph image.
    return anaglyph;
}


/// <summary>
/// Rotates a grid counter-clockwise around the center by a given angle in degrees.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="center">The center of the rotation (in pixel coords).</param>
/// <param name="angle">Angle in degrees.</param>
/// <returns>Mesh, the rotated grid.</returns>
Mesh rotatedWarpGrid(Mesh& grid, const glm::vec2& center, const float& angle)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float DEGREE2RADIANS = 0.0174532925f;

    //
    // The goal is to rotate the coordinate of the grid vertices
    // counter-clockwise around the 'center' by a given angle in degrees:
    //
    // 1. Create a 3*3 matrix T, there are three steps for this matrix:
    //    Translate from origin to center,
    //    then rotate counter-clockwise by a given angles,
    //    and translate back from center to origin.
    //
    // 2. Multiply each vertex by matrix T to get its new coordinates.
    //

    //
    //    YOUR CODE GOES HERE
    //
    float rad_angle = angle * DEGREE2RADIANS;

    for (auto& vertex : new_grid.vertices) {
        glm::vec2 translated_vertex = vertex - center;

        float rotated_x = translated_vertex.x * cos(rad_angle) + translated_vertex.y * sin(rad_angle);
        float rotated_y = -translated_vertex.x * sin(rad_angle) + translated_vertex.y * cos(rad_angle);

        vertex.x = rotated_x + center.x;
        vertex.y = rotated_y + center.y;
    }

    return new_grid;
}



/// <summary>
/// Rotate an image using backward warping based on the provided meshes.
/// </summary>
/// <param name="image">input image</param>
/// <param name="src_grid">original grid</param>
/// <param name="dst_grid">rotated grid</param>
/// <param name="sampleBilinear">a function that bilinearly samples ImageFloat</param>
/// <param name="sampleBilinearRGB">a function that bilinearly samples ImageRGB</param>
/// <returns>rotated image, has the same size as input</returns>
ImageRGB rotateImage(const ImageRGB& image, const Mesh& src_grid, const Mesh& dst_grid, const BilinearSamplerFloat& sampleBilinear, const BilinearSamplerRGB& sampleBilinearRGB)
{

    //
    // Unused pixels should be black.
    // Pixel that fall outside of the image should be discarded.
    //
    //    YOUR CODE GOES HERE
    //
    ImageRGB result(image.width, image.height);
    glm::vec3 black(0.0f, 0.0f, 0.0f);

    std::fill(result.data.begin(), result.data.end(), black);

    for (int i = 0; i < dst_grid.triangles.size(); i++) {
        // 1
        auto vert_a_dst = dst_grid.vertices[dst_grid.triangles[i].x];
        auto vert_b_dst = dst_grid.vertices[dst_grid.triangles[i].y];
        auto vert_c_dst = dst_grid.vertices[dst_grid.triangles[i].z];

        auto x_min = glm::min(vert_a_dst.x, glm::min(vert_b_dst.x, vert_c_dst.x));
        auto y_min = glm::min(vert_a_dst.y, glm::min(vert_b_dst.y, vert_c_dst.y));
        auto x_max = glm::max(vert_a_dst.x, glm::max(vert_b_dst.x, vert_c_dst.x));
        auto y_max = glm::max(vert_a_dst.y, glm::max(vert_b_dst.y, vert_c_dst.y));

        auto bb_min = glm::vec2(glm::floor(x_min), glm::floor(y_min));
        auto bb_max = glm::vec2(glm::ceil(x_max), glm::ceil(y_max));

        bb_min.x = glm::max(static_cast<int>(bb_min.x), 0);
        bb_min.y = glm::max(static_cast<int>(bb_min.y), 0);
        bb_max.x = glm::min(static_cast<int>(bb_max.x), image.width - 1);
        bb_max.y = glm::min(static_cast<int>(bb_max.y), image.height - 1);

        // 2
        for (int y = bb_min.y; y <= bb_max.y; y++) {
            for (int x = bb_min.x; x <= bb_max.x; x++) {
                auto pixel = glm::vec2(x + 0.5, y + 0.5);
                if (isPointInsideTriangle(pixel, vert_a_dst, vert_b_dst, vert_c_dst)) {
                    // 3
                    auto bc = barycentricCoordinates(pixel, vert_a_dst, vert_b_dst, vert_c_dst);
                    // 4
                    auto vert_a_src = src_grid.vertices[src_grid.triangles[i].x];
                    auto vert_b_src = src_grid.vertices[src_grid.triangles[i].y];
                    auto vert_c_src = src_grid.vertices[src_grid.triangles[i].z];
                    // 5
                    auto pt_src = vert_a_src * bc.x + vert_b_src * bc.y + vert_c_src * bc.z;
                    if (pt_src.x >= 0 && pt_src.x < image.width && pt_src.y >= 0 && pt_src.y < image.height) {
                        result.data[y * image.width + x] = sampleBilinearRGB(image, pt_src);                   
                    }
                }
            }
        }
    }
    return result; // replace
}