// GUI.hpp
#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <mutex>
#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "MyPointCloud.h"
#include "CloudGeneration.h"
#include "PointProcessing.h"
#include <Eigen/Dense>

class GUI {

    public:

        GUI(){

            // Initialize polyscope
            selectedQuantities.resize(7, 1);

            // Initialize the point cloud
            selectedFile = assetsDir + "armadillo.obj";

            pointProcessing.measureTime("[Generation] Load object", [this](){
                loadObject(mainCloud, selectedFile);
            });
            
            pointProcessing.update(mainCloud);
            polyscope_mainCloud = polyscope::registerPointCloud(mainCloudName, mainCloud.getVertices());
            addQuantities(polyscope_mainCloud, "real normals", mainCloud.getNormals());
        }

        void mainCallBack();


    private: 

        // Point Cloud Informations
        
        std::string mainCloudName = "mainCloud";
        std::string assetsDir = "assets/";
        std::string selectedFile = "";

        Scalar pointRadius    = 0.005; /// < display radius of the point cloud
        polyscope::PointCloud* polyscope_mainCloud;

        MyPointCloud mainCloud;
        PointProcessing pointProcessing;

    private:

        // Point Cloud Processing 

        // quantities to display
        // 0 : display projections
        // 1 : display normals
        // 2 : display min curvature direction
        // 3 : display max curvature direction
        // 4 : display min curvature
        // 5 : display max curvature
        // 6 : display mean curvature
        std::vector<std::string> quantityNames = {"Projections", "Normals", "Min curvature direction", "Max curvature direction", "Min curvature", "Max curvature", "Mean curvature"};
        std::vector<int> selectedQuantities; // Initialize to 1 all quantities
        
        std::string lastDryRun = "";

        bool all_computed = false;
        std::string methodName = "";

        void cloudComputing();

        void cloudComputingUpdateAll();

        template <typename FitT>
        void methodForCloudComputing(const std::string &metName);

        void cloudComputingParameters();

        void addQuantities(polyscope::PointCloud *pc, const std::string &name, const Eigen::MatrixXd &values);


}; // class GUI

#include "GUI.hpp"
