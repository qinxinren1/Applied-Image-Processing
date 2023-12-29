#include "your_code_here.h"

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path outDirPath { OUTPUT_DIR };

/// <summary>
/// Main method. Runs default tests. Feel free to modify it, add more tests and experiments,
/// change the input images etc. The file is not part of the solution. All solutions have to 
/// implemented in "your_code_here.h".
/// </summary>
/// <returns>0</returns>
int main()
{
    std::chrono::steady_clock::time_point time_start, time_end;
    printOpenMPStatus();
    
    // 0. Load inputs from files. https://www.cs.huji.ac.il/~danix/hdr/pages/memorial.html
    auto image = ImageRGB(dataDirPath / "memorial2_half.hdr");
    image.writeToFile(outDirPath / "0_src.png");

    // 1. Normalize the image range to [0,1].
    auto image_normed = normalizeRGBImage(image);
    image_normed.writeToFile(outDirPath / "1_normalized.png");

    auto test_image = ImageRGB(dataDirPath / "expected-outputs" / "1_normalized.png");
    // 2. Apply gamma curve.
    auto image_gamma = applyGamma(image_normed, 1 / 2.2f);
    image_gamma.writeToFile(outDirPath / "2_gamma.png");

    // 2b. Apply gamma to the original image.
    auto gamma_orig = applyGamma(image, 1 / 2.2f);
    gamma_orig.writeToFile(outDirPath / "2_gamma_orig.png");

    // 3. Get luminance.
    auto luminance = rgbToLuminance(image);
    luminance.writeToFile(outDirPath / "3a_luminance.png");
    auto H = lnImage(luminance);
    H.writeToFile(outDirPath / "3b_log_luminance_H.png");

    // 4. Compute luminance gradients \nabla H (Sec. 5).
    auto gradients = getGradients(H);
    gradientsToRgb(gradients).writeToFile(outDirPath / "4_gradients_H.png");
    std::cout << "grad.x.MIN " << *std::min_element(gradients.x.data.begin(), gradients.x.data.end()) << std::endl;
    std::cout << "grad.x.MAX " << *std::max_element(gradients.x.data.begin(), gradients.x.data.end()) << std::endl;

    // 5. Compute the gradient attenuation \phi (Sec. 4).
    auto grad_atten = getGradientAttenuation(gradients);
    grad_atten.writeToFile(outDirPath / "5_attenuation_phi.png");
    std::cout << "MIN " << *std::min_element(grad_atten.data.begin(), grad_atten.data.end()) << std::endl;
    std::cout << "MAX " << *std::max_element(grad_atten.data.begin(), grad_atten.data.end()) << std::endl;

    // 6. Compute the attentuated divergence (Sec. 3 and 5).
    auto divergence = getAttenuatedDivergence(gradients, grad_atten);
    auto n = normalizeRGBImage(imageFloatToRgb(divergence));
    imageRgbToFloat(normalizeRGBImage(imageFloatToRgb(divergence))).writeToFile(outDirPath / "6_divergence_G.png");

    std::cout << "divergence.MIN " << *std::min_element(divergence.data.begin(), divergence.data.end()) << std::endl;
    std::cout << "divergence.MAX " << *std::max_element(divergence.data.begin(), divergence.data.end()) << std::endl;

    // 7. Very simplistic single-scale direct solver of the Poisson equation (Eq. 3).
    time_start = std::chrono::steady_clock::now();
    auto solved_luminance = solvePoisson(divergence);
    time_end = std::chrono::steady_clock::now();
    std::cout << "Poisson Solver | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;

    // Normalize the result to [0,1] range.
    // Note: We are using our code for RGB images here which is inefficient.
    solved_luminance = imageRgbToFloat(normalizeRGBImage(imageFloatToRgb(solved_luminance)));
    solved_luminance.writeToFile(outDirPath / "7_poisson_solution_I.png");

    // 8. Convert back to RGB.
    auto result_rgb = rescaleRgbByLuminance(image, luminance, solved_luminance);
    result_rgb.writeToFile(outDirPath / "8_result_rgb.png");

    // Test every pixel
    std::vector<std::string> name_list = { "0_src.png", "1_normalized.png", "2_gamma.png", "3a_luminance.png", "3b_log_luminance_H.png", "4_gradients_H.png", "5_attenuation_phi.png", "6_divergence_G.png", "7_poisson_solution_I.png", "8_result_rgb.png" };

    const float EPSILON = 1e-7f;
    for (auto ele : name_list) {
        ImageRGB correct_answer = ImageRGB(dataDirPath / ("expected-outputs/" + ele));
        ImageRGB student_answer = ImageRGB(outDirPath / ele);
        for (int y = 0; y < correct_answer.height; y++) {
            int break_sign = 0;
            for (int x = 0; x < correct_answer.width; x++) {
                int index = x + y * correct_answer.width;
                glm::vec3 correct_pixel = correct_answer.data[index];
                glm::vec3 student_pixel = student_answer.data[index];
                if (correct_pixel.x - student_pixel.x >= EPSILON && correct_pixel.y - student_pixel.y >= EPSILON && correct_pixel.z - student_pixel.z >= EPSILON) {
                    std::cout << ele << " are not the same starts at pixel [" << x << ", " << y << "]" << std::endl;
                    std::cout << "Correct pixel: (" << correct_pixel.x << ", " << correct_pixel.y << ", " << correct_pixel.z << ")" << std::endl;
                    std::cout << "Student pixel: (" << student_pixel.x << ", " << student_pixel.y << ", " << student_pixel.z << ")" << std::endl;
                    break_sign = 1;
                }
            }
        }
    }

    std::cout << "All done!" << std::endl;
    return 0;
}
