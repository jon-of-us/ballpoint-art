#include <iostream>
#include <opencv2/opencv.hpp>

const double relative_margin = 0.05;
const double relative_blur_kernel_size = 0.05;
const int iterations = 300;
const double cut_off_creep = 0.1;

cv::Mat convert(const cv::Mat& image) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0 / 255.0);

    cv::Mat integral = cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat hole = cv::Mat::zeros(image.size(), CV_8S);
    
    int margin = static_cast<int>(relative_margin * std::min(image.rows, image.cols));
    hole(cv::Rect(margin, margin, image.cols - 2 * margin, image.rows - 2 * margin)) = 1;

    for (int i = 0; i < iterations; ++i) {
        cv::Mat edge = normalized.mul(1 - hole);
        int blur_kernel_size = static_cast<int>(relative_blur_kernel_size * std::min(image.rows, image.cols));
        blur_kernel_size += (blur_kernel_size % 2 == 0) ? 1 : 0;

        cv::Mat blurred;
        cv::GaussianBlur(edge, blurred, cv::Size(blur_kernel_size, blur_kernel_size), 0);

        cv::Mat creep_in = blurred.mul(hole);
        cv::compare(creep_in, cut_off_creep, creep_in, cv::CMP_GT);
        hole -= creep_in;
        integral += hole;
    }

    return integral;
}

int main() {
    // Example usage:
    cv::Mat input_image = cv::imread("your_image_path.jpg", cv::IMREAD_GRAYSCALE);
    if (input_image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    cv::Mat result = convert(input_image);

    // Display or save the result as needed
    cv::imshow("Result", result);
    cv::waitKey(0);

    return 0;
}
