#include "polyscope/polyscope.h"

#include <igl/readOBJ.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"

#include <Ponca/Fitting>
#include <Ponca/SpatialPartitioning>
#include "poncaAdapters.hpp"

#include <iostream>
#include <utility>

// Types definition
using Scalar             = double;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTree<PPAdapter>;
using WeightConstantFunc = Ponca::DistWeightFunc<PPAdapter, Ponca::ConstantWeightKernel<Scalar> >;
using PlaneFit           = Ponca::Basket<PPAdapter, WeightConstantFunc, Ponca::CovariancePlaneFit>;


// Variables
Eigen::MatrixXd cloudV;
KdTree tree;
polyscope::PointCloud* cloud = nullptr;

// Options for algorithms
int iVertexSource  = 7;
int kNN            = 10;
Scalar pointRadius = 0.02;

// Colorize neighbors of point iVertexSource
void computeKnnFrom() {
    int nvert = tree.index_data().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(iVertexSource) = 2;
    for (int j : tree.k_nearest_neighbors(iVertexSource, kNN)){
        closest(j) = 1;
    }
    cloud->addScalarQuantity(  std::to_string(kNN) + "-neighborhood of vertex " + std::to_string(iVertexSource), closest);
}

// Fit plane on all vertices
void fitPlane() {

    int nvert = tree.index_data().size();
    Eigen::MatrixXd meshN( nvert, 3 );
    Eigen::VectorXd surfvar ( nvert );

#pragma omp parallel for
    for( int i = 0; i != nvert; ++i ){
        PlaneFit fit;
        auto n = meshN.row( i );
        fit.init( tree.point_data()[i].pos() );

        for (int j : tree.k_nearest_neighbors(i, kNN))
            fit.addNeighbor( tree.point_data()[j] );
        fit.finalize();

        n = fit.primitiveGradient();
        surfvar(i) = fit.surfaceVariation();
    }
    cloud->addVectorQuantity("PlaneFit - Normals", meshN)->setVectorLengthScale(Scalar(2)*pointRadius);
    cloud->addScalarQuantity( "PlaneFit - Surface Variation", surfvar );
}

void callback() {

    ImGui::PushItemWidth(100);

    if (ImGui::Button("fit plane")) fitPlane();
    ImGui::SameLine();
    if (ImGui::Button("show knn"))  computeKnnFrom();

    ImGui::Separator();

    ImGui::InputInt("source vertex", &iVertexSource);
    ImGui::InputInt("neighborhood size", &kNN);

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
    igl::readOBJ(filename, cloudV, meshF);

    // Build Ponca KdTree
    buildKdTree( cloudV, tree );

    // Register the point cloud with Polyscope
    cloud = polyscope::registerPointCloud("cloud", cloudV);
    cloud->setPointRadius(pointRadius);

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    return 0;
}