#include "polyscope/polyscope.h"

#include <igl/readOBJ.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"

#include <iostream>
#include <utility>

// The mesh, Eigen representation
Eigen::MatrixXd meshV;

// Options for algorithms
int iVertexSource = 7;

void addCurvatureScalar() {
    std::cerr << "Not implemented yet" << std::endl;
}

void computeDistanceFrom() {
    std::cerr << "Not implemented yet" << std::endl;
}

void computeNormals() {
    std::cerr << "Not implemented yet" << std::endl;
}

void callback() {

    static int numPoints = 2000;
    static float param = 3.14;

    ImGui::PushItemWidth(100);

    // Curvature
    if (ImGui::Button("add curvature")) {
        addCurvatureScalar();
    }

    // Normals
    if (ImGui::Button("add normals")) {
        computeNormals();
    }

    // Geodesics
    if (ImGui::Button("compute distance")) {
        computeDistanceFrom();
    }
    ImGui::SameLine();
    ImGui::InputInt("source vertex", &iVertexSource);

    ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
    // Options
    polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();

    std::string filename = "assets/bunnyhead.obj";
    std::cout << "loading: " << filename << std::endl;

    // Read the point cloud
    Eigen::MatrixXi meshF;
    igl::readOBJ(filename, meshV, meshF);

    // Register the mesh with Polyscope
    polyscope::registerPointCloud("input mesh", meshV);

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    return 0;
}