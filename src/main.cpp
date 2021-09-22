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
using WeightSmoothFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;
using PlaneFit           = Ponca::Basket<PPAdapter, WeightSmoothFunc, Ponca::CovariancePlaneFit>;
using APSSFit            = Ponca::Basket<PPAdapter, WeightConstantFunc, Ponca::OrientedSphereFit>;


// Variables
Eigen::MatrixXd cloudV;
KdTree tree;
polyscope::PointCloud* cloud = nullptr;

// Options for algorithms
int iVertexSource  = 7;
int kNN            = 10;
float NSize        = 0.2;
Scalar pointRadius = 0.02;

void computeNeiFrom() {
    int nvert = tree.index_data().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    WeightSmoothFunc w( NSize );

    closest(iVertexSource) = 2;
    const auto &p = tree.point_data()[iVertexSource];
    for (int j : tree.range_neighbors(iVertexSource, NSize)){
        const auto &q = tree.point_data()[j];
        closest(j) = w.w( q.pos() - p.pos(), q );
    }
    cloud->addScalarQuantity(  "range neighborhood (" + std::to_string(NSize)+ ") of vertex " + std::to_string(iVertexSource), closest);
}

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

// Traverse point cloud, compute fitting, and use functor to process it
// Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloud(const typename FitT::WeightFunction& w, Functor f){

    int nvert = tree.index_data().size();

#pragma omp parallel for
    for( int i = 0; i != nvert; ++i ){
        FitT fit;
        fit.setWeightFunc( w );
        fit.init( tree.point_data()[i].pos() );

        for (int j : tree.k_nearest_neighbors(i, kNN))
            fit.addNeighbor( tree.point_data()[j] );

        fit.finalize();
        if( fit.isStable() )
            f( i, fit );
        else
            std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
    }
}

// Fit plane on all vertices
void fitPlane() {
    int nvert = tree.index_data().size();
    Eigen::MatrixXd meshN( nvert, 3 );
    Eigen::VectorXd surfvar ( nvert );

    processPointCloud<PlaneFit>( //WeightConstantFunc(1),
                                 WeightSmoothFunc(NSize),
                                 [&meshN, &surfvar]( int i, const PlaneFit& fit){
        meshN.row( i ) = fit.primitiveGradient();
        surfvar  ( i ) = fit.surfaceVariation();
    });

    cloud->addVectorQuantity("PlaneFit - Normals", meshN)->setVectorLengthScale(Scalar(2)*pointRadius);
    cloud->addScalarQuantity( "PlaneFit - Surface Variation", surfvar );
}

template<typename FitT>
void estimateCurvature_impl(const std::string& name) {
    int nvert = tree.index_data().size();
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );

    processPointCloud<FitT>( //WeightConstantFunc(1),
                             WeightSmoothFunc(NSize),
                             [&mean, &kmin, &kmax, &dmin, &dmax]( int i, const FitT& fit){
        mean(i) = fit.kMean();
        kmax(i) = fit.k1();
        kmin(i) = fit.k2();

        dmin.row( i ) = fit.k1Direction();
        dmax.row( i ) = fit.k1Direction();
    });

    cloud->addScalarQuantity(name + " - Mean Curvature", mean);
    cloud->addScalarQuantity(name + " - K1", kmin);
    cloud->addScalarQuantity(name + " - K2", kmax);

    cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(Scalar(2)*pointRadius);
    cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(Scalar(2)*pointRadius);
}

// Compute curvature using Plane fitting
void estimateCurvatureWithPlane() {
    estimateCurvature_impl<Ponca::Basket<PPAdapter,WeightSmoothFunc,
                                         Ponca::CovariancePlaneFit,Ponca::CovariancePlaneSpaceDer,
                                         Ponca:: CurvatureEstimator>>("PSS");
}

// Compute curvature using APSS
void estimateCurvatureWithAPSS() {
//    estimateCurvature_impl<Ponca::Basket<PPAdapter, WeightConstantFunc,
//            Ponca::OrientedSphereFit, Ponca::OrientedSphereSpaceDer,
//            Ponca::CurvatureEstimator>>("APSS");
}

// Compute curvature using Algebraic Shape Operator
void estimateCurvatureWithASO() {
//    estimateCurvature_impl<Ponca::Basket<PPAdapter, WeightConstantFunc,
//            Ponca::OrientedSphereFit, Ponca::OrientedSphereSpaceDer, Ponca::MlsSphereFitDer,
//            Ponca::CurvatureEstimator>>("ASO");
}

// Generate new smoothed point cloud using Covariance Plane Fitting
void smoothCovPlane() {

}

// Generate new smoothed point cloud using Algebraic Point Set Surfaces
void smoothAPSS() {

}

void callback() {

    ImGui::PushItemWidth(100);

    ImGui::Text("Neighborhood collection");

    ImGui::InputInt("k-neighborhood size", &kNN);
    ImGui::InputFloat("neighborhood size", &NSize);
    ImGui::InputInt("source vertex", &iVertexSource);
    ImGui::SameLine();
    if (ImGui::Button("show knn"))  computeKnnFrom();
    ImGui::SameLine();
    if (ImGui::Button("show euclidean nei"))  computeNeiFrom();

    ImGui::Separator();

    ImGui::Text("Computations");
    if (ImGui::Button("fit plane")) fitPlane();
    ImGui::Separator();

    ImGui::Text("Curvature estimation");
    if (ImGui::Button("Plane"))  estimateCurvatureWithPlane();
    ImGui::SameLine();
    if (ImGui::Button("APSS"))  estimateCurvatureWithAPSS();
    ImGui::SameLine();
    if (ImGui::Button("ASO"))  estimateCurvatureWithASO();

    ImGui::Separator();
    ImGui::Text("Smoothing");
    if (ImGui::Button("Covariance Plane")) smoothCovPlane();
    if (ImGui::Button("APSS")) smoothAPSS();

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