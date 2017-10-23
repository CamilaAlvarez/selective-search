#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <chrono>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <vector>
#include <fstream>
#include "boost/filesystem.hpp"

DEFINE_string(input_images, "", "File that contains one image path per line");
DEFINE_string(base_image_dir, "", "Base directory for all images (Optional)");
DEFINE_string(output_directory, "", "Directory where regions will be saved");
DEFINE_int32(number_regions, 100, "Number of regions to retrieve");
DEFINE_double(min_area, 0.2, "Minimum area to consider");
DEFINE_bool(fast_computation, false, "Set fast computatation");

static void resize(cv::Mat &image){
    //Resize so it has same size that faster r-cnn uses
    int newHeight, newWidth;
    if(image.cols > image.rows){
        //width > height
        newWidth = 600;
        newHeight = image.rows*newWidth/image.cols;
        if(newHeight > 450){
            newHeight = 450;
            newWidth = image.cols*newHeight/image.rows;
        }
    }
    else{
        //height >= width
        newHeight = 600;
        newWidth = image.cols*newHeight/image.rows;
        if(newWidth > 450){
            newWidth = 450;
            newHeight = image.rows*newWidth/image.cols;
        }
    }
    cv::resize(image, image, cv::Size(newWidth, newHeight));
}

static void createOutputDirectoryIfNecessary(std::string &image_location, std::string &output_location){
    boost::filesystem::path image_path(image_location+".txt");
    boost::filesystem::path output_path(FLAGS_output_directory);
    boost::filesystem::path output_file = output_path / image_path;

    boost::filesystem::path final_output_dir = output_file.parent_path();
    if(!boost::filesystem::exists(final_output_dir)){
        boost::filesystem::create_directories(final_output_dir);
    }
    output_location = output_file.string();
}

int main(int argc, char *argv[]) {
    FLAGS_logtostderr = true;
    gflags::SetUsageMessage("command line brew\n"
                                    "usage: selective_search\n"
                                    "--input_images <input file>\n"
                                    "--output_directory <output directory>\n"
                                    "--number_regions <number of regions>"
                                    "--min_area <min area to consider>");
    ::google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if(FLAGS_input_images.empty() || FLAGS_output_directory.empty() ){
        gflags::ShowUsageWithFlagsRestrict(argv[0],"selective_search");
        return 1;
    }

    boost::filesystem::path output_directory(FLAGS_output_directory);
    if(boost::filesystem::is_directory(output_directory)){
        LOG(WARNING) << "OUTPUT DIRECTORY EXISTS. FILES MAY BE OVERWRITTEN.";
    }
    else{
        boost::filesystem::create_directories(output_directory);
    }

    std::ifstream images_file(FLAGS_input_images);
    CHECK(images_file.is_open()) << "COULD NOT OPEN INPUT FILE";
    std::vector<std::string> images;
    std::string line;
    while(std::getline(images_file, line)){
        images.push_back(line);
    }
    for (std::vector<std::string>::iterator it = images.begin();
         it != images.end() ; ++it) {
        std::string filename = *it;
        if(!FLAGS_base_image_dir.empty()){
            boost::filesystem::path base_dir(FLAGS_base_image_dir);
            boost::filesystem::path image_name(filename);
            image_name = base_dir / image_name;
            filename = image_name.string();

        }
        cv::Mat image = cv::imread(filename);
        LOG_IF(WARNING ,image.empty()) << "EMPTY IMAGE: " << filename;
        if(image.empty())
            continue;
        resize(image);
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss =
                cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
        // set input image on which we will run segmentation
        ss->setBaseImage(image);
        if(FLAGS_fast_computation){
            ss->switchToSelectiveSearchFast();
        }
        else{
            ss->switchToSelectiveSearchQuality();
        }
        std::vector<cv::Rect> regions;
        LOG(INFO) << "CALCULATING REGIONS for image: " << filename;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        ss->process(regions);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
        LOG(INFO) <<  "TOTAL NUMBER OF REGION PROPOSALS: " << regions.size() << " IN :" << duration << " milliseconds";
        std::string output_file;
        createOutputDirectoryIfNecessary(*it, output_file);
        std::ofstream output(output_file);
        int imageArea = image.cols*image.rows;
        int count = 0;
        LOG(INFO) << "WRITING OUTPUT";
        output << "Image size: " << image.cols << "," << image.rows << std::endl;
        output << duration << "ms" << std::endl;
        output << "x1,y1,x2,y2" << std::endl;
        for (int i = 0; i < regions.size(); ++i) {
            if(count >= FLAGS_number_regions )
                break;
            cv::Rect region = regions[i];
            if (region.area() < FLAGS_min_area*imageArea)
                continue;
            output << region.x << "," << region.y << "," << region.x+region.width << "," << region.y+region.height
                   << std::endl;
            count++;
        }

        output.close();
#ifdef DEBUG
        count = 0;
        cv::Mat imOut = image.clone();
        for(int i = 0; i < regions.size(); i++) {
            if (count >= FLAGS_number_regions) {
                break;
            }
            cv::Rect region = regions[i];
            if (region.area() < FLAGS_min_area*imageArea)
                continue;
            cv::rectangle(imOut, region, cv::Scalar(0, 255, 0));
            count++;

        }
        imshow("Output", imOut);
        cv::waitKey();
        // show output
        imshow("Output", imOut);
#endif
        LOG(INFO) << "FINISHED IMAGE: " << *it;
    }
    images_file.close();
    return 0;
}