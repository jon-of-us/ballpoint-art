#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "integral.h"  // Assuming you have an 'integral.h' header file for the 'convert' function
#include "rawpy.h"

const int down_scale = 4;

void files_in_folder(const std::string& path, std::vector<std::string>& file_paths) {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            file_paths.push_back(entry.path().string());
        }
    }
}

cv::Mat downscaled(const cv::Mat& image, int scale = down_scale) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(image.cols / scale, image.rows / scale));
    return resized_image;
}

cv::Mat read_image(const std::string& filepath) {
    std::string extension = filepath.substr(filepath.find_last_of('.') + 1);
    cv::Mat image;

    if (extension == "jpg" || extension == "jpeg" || extension == "png") {
        image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    } else if (extension == "raw" || extension == "dng") {
        rawpy::Image raw;
        raw.open_file(filepath.c_str());
        auto rgb = raw.postprocess(rawpy::PostprocessOptions());
        cv::cvtColor(rgb, image, cv::COLOR_RGB2GRAY);
        image.convertTo(image, CV_8U, 1.0 / 256);
    }

    return image;
}

void convert_all_to_video() {
    std::vector<std::string> file_paths;
    files_in_folder("./input", file_paths);

    for (const auto& filepath : file_paths) {
        // read image to OpenCV Mat in grayscale
        cv::Mat in_arr = read_image(filepath);
        in_arr = downscaled(in_arr);

        std::string out_path = "./output/" + filepath.substr(filepath.find_last_of('/') + 1, filepath.find_last_of('.') - filepath.find_last_of('/') - 1) + ".mp4";
        cv::VideoWriter video(
            out_path,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            1.3,
            in_arr.size(),
            false
        );

        for (int scale = 0; scale <= 2500; scale += 200) {
            // cv::Mat out_arr = convert(in_arr, scale);
            cv::Mat out_image;
            cv::cvtColor(
                cv::Mat(convert(in_arr, scale)),
                out_image,
                cv::COLOR_GRAY2RGB
            );
            // cv::resize(out_image, new_shape);
            // save image
            video.write(out_image);
        }

        video.release();
        cv::destroyAllWindows();
    }
}

void convert_all_to_foto() {
    std::vector<std::string> file_paths;
    files_in_folder("./input", file_paths);

    for (const auto& filepath : file_paths) {
        // read image to OpenCV Mat in grayscale
        cv::Mat in_arr = read_image(filepath);
        in_arr = downscaled(in_arr);

        cv::Mat out_arr(convert(in_arr));
        std::string out_path = "./output/" + filepath.substr(filepath.find_last_of('/') + 1, filepath.find_last_of('.') - filepath.find_last_of('/') - 1) + ".png";

        // save image
        cv::imwrite(out_path, out_arr, {cv::IMWRITE_PNG_COMPRESSION, 0});
    }
}

int main() {
    convert_all_to_foto();
    return 0;
}
