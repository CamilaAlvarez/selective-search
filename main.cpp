#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <chrono>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <vector>
#include <map>
#include <fstream>
#include "boost/filesystem.hpp"
#ifdef _OPENMP
    #include <omp.h>
#endif

DEFINE_string(input_images, "", "File that contains one image path per line");
DEFINE_string(base_image_dir, "", "Base directory for all images (Optional)");
DEFINE_string(output_directory, "", "Directory where regions will be saved");
DEFINE_int32(number_regions, 100, "Number of regions to retrieve");
DEFINE_double(min_width, 0.2, "Minimum area to consider");
DEFINE_bool(fast_computation, false, "Set fast computatation");

static std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

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
                                    "--min_width <min area to consider>");
    ::google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if(FLAGS_input_images.empty() || FLAGS_output_directory.empty() ){
        gflags::ShowUsageWithFlagsRestrict(argv[0],"selective_search");
        return 1;
    }
    cv::setUseOptimized(true);
    cv::setNumThreads(4);
    boost::filesystem::path output_directory(FLAGS_output_directory);
    if(boost::filesystem::is_directory(output_directory)){
        LOG(WARNING) << "OUTPUT DIRECTORY EXISTS. FILES MAY BE OVERWRITTEN.";
    }
    else{
        boost::filesystem::create_directories(output_directory);
    }

    std::ifstream images_file(FLAGS_input_images);
    CHECK(images_file.is_open()) << "COULD NOT OPEN INPUT FILE";
    //std::map<std::string,std::string> images;
    std::string line;
    std::vector<std::string> images;
    std::vector<std::string> images_id;
    while(std::getline(images_file, line)){
        std::vector<std::string> splittedLine = split(line, "\t");
        images.push_back(splittedLine[1]);
	images_id.push_back(splittedLine[0]);
    }
    size_t number_images = images.size();
    #ifdef _OPENMP
        omp_lock_t lock;
        omp_init_lock(&lock);
    #endif
    #pragma omp parallel for num_threads(6)
    for (size_t i = 0; i< number_images; i++) {
        std::string filename = images[i];
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
        createOutputDirectoryIfNecessary(images[i], output_file);
        std::ofstream output(output_file);
        int imageWidth = image.cols;
        int count = 0;
        boost::filesystem::path output_path(FLAGS_output_directory);
        LOG(INFO) << "WRITING OUTPUT";
        output << "Image size: " << image.cols << "," << image.rows << std::endl;
        output << duration << "ms" << std::endl;
        output << "x1,y1,x2,y2" << std::endl;
        for (int j = 0; j < regions.size(); ++j) {
            if(count >= FLAGS_number_regions )
                break;
	    cv::Rect region = regions[j];
            if (region.width < FLAGS_min_width*imageWidth)
                continue;
            count++;
            boost::filesystem::path image_path(images_id[i]+"#"+std::to_string(count)+".jpg");
            boost::filesystem::path output_file = output_path / image_path;
            output << images_id[i] << "\t" << images_id[i] << "#" << count << "\t"
                << region.x << "," << region.y << "," << region.x+region.width << "," << region.y+region.height
                << "\t" << output_file.string() << std::endl;
            cv::imwrite(output_file.string(), image(region));
            
        }

        output.close();
#ifdef DEBUG
        count = 0;
        cv::Mat imOut = image.clone();
        for(int j = 0; j < regions.size(); j++) {
            if (count >= FLAGS_number_regions) {
                break;
            }
            cv::Rect region = regions[j];
            if (region.width < FLAGS_min_width*imageWidth)
                continue;
            cv::rectangle(imOut, region, cv::Scalar(0, 255, 0));
            count++;

        }
        imshow("Output", imOut);
        cv::waitKey();
        // show output
        imshow("Output", imOut);
#endif
        LOG(INFO) << "FINISHED IMAGE: " << images[i];
    }
    #ifdef _OPENMP
        omp_destroy_lock(&lock);
    #endif
    images_file.close();
    return 0;
}
