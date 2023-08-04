#pragma once

#include <igl/read_triangle_mesh.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>
#include "MyPointCloud.h"

void loadPTSObject (MyPointCloud &cloud, std::string filename){

    Eigen::MatrixXd cloudV, cloudN;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return;
    }

    std::string line;
    // Read the first line (header)
    std::getline(file, line);
    std::istringstream iss(line);
    std::vector<std::string> header;
    std::string word;
    while (iss >> word) {
        header.push_back(word);
    }

    // Find the indices of the x, y, z, nx, ny, nz in the header
    std::map<std::string, int> indices;
    for (int i = 0; i < header.size(); ++i) {
        if (header[i] == "x" || header[i] == "y" || header[i] == "z" || header[i] == "nx" || header[i] == "ny" || header[i] == "nz") {
            indices[header[i]] = i-1;
        }
    }

    // Read the rest of the file
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values(header.size());
        for (int i = 0; i < header.size(); ++i) {
            iss >> values[i];
        }

        // Add the read values to the matrices
        cloudV.conservativeResize(cloudV.rows() + 1, 3);
        cloudV.row(cloudV.rows() - 1) << values[indices["x"]], values[indices["y"]], values[indices["z"]];

        cloudN.conservativeResize(cloudN.rows() + 1, 3);
        cloudN.row(cloudN.rows() - 1) << values[indices["nx"]], values[indices["ny"]], values[indices["nz"]];
    }

    file.close();

    cloud = MyPointCloud(cloudV, cloudN);
}


void loadObject (MyPointCloud &cloud, std::string filename) {

    Eigen::MatrixXi meshF;
    Eigen::MatrixXd cloudV, cloudN;

    if (filename.substr(filename.find_last_of(".") + 1) == "pts") {
        loadPTSObject(cloud, filename);
        return;
    }
    else {
        if (filename.substr(filename.find_last_of(".") + 1) == "ply"){
            Eigen::MatrixXi cloudE;
            Eigen::MatrixXd cloudUV;
            igl::readPLY(filename, cloudV, meshF, cloudE, cloudN, cloudUV);
        }
        else {
            igl::read_triangle_mesh(filename, cloudV, meshF);
        }
        
        if ( cloudN.rows() == 0 )
            igl::per_vertex_normals(cloudV, meshF, cloudN);
        
    }
    // Check if there is mesh 
    if ( meshF.rows() == 0 && cloudN.rows() == 0 ) {
        std::cerr << "[libIGL] The mesh is empty. Aborting..." << std::endl;
        exit (EXIT_FAILURE);
    }

    // Check if normals have been properly loaded
    int nbUnitNormal = cloudN.rowwise().squaredNorm().sum();
    if ( meshF.rows() != 0 && nbUnitNormal != cloudV.rows() ) {
        std::cerr << "[libIGL] An error occurred when computing the normal vectors from the mesh. Aborting..."
                  << std::endl;
        exit (EXIT_FAILURE);
    }

    cloud = MyPointCloud(cloudV, cloudN);
}
