#include "polyscope/polyscope.h"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"

#include <Ponca/Fitting>
#include <Ponca/SpatialPartitioning>
#include "poncaAdapters.hpp"

#include <iostream>
#include <utility>
#include <chrono>

// Types definition
using Scalar             = double;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTree<PPAdapter>;
using WeightConstantFunc = Ponca::DistWeightFunc<PPAdapter, Ponca::ConstantWeightKernel<Scalar> >;
using WeightSmoothFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;
using PlaneFit           = Ponca::Basket<PPAdapter, WeightSmoothFunc, Ponca::CovariancePlaneFit>;
using APSSFit            = Ponca::Basket<PPAdapter, WeightConstantFunc, Ponca::OrientedSphereFit>;


// Variables
Eigen::MatrixXd cloudV, cloudN;
KdTree tree;
polyscope::PointCloud* cloud = nullptr;

// Options for algorithms
int iVertexSource  = 7;    /// < id of the selected point
int kNN            = 10;   /// < neighborhood size (knn)
float NSize        = 0.2;  /// < neighborhood size (euclidean)
Scalar pointRadius = 0.005; /// < display radius of the point cloud


/// Convenience function measuring and printing the processing time of F
template <typename Functor>
void measureTime( const std::string &actionName, Functor F ){
    using namespace std::literals; // enables the usage of 24h, 1ms, 1s instead of
                                   // e.g. std::chrono::hours(24), accordingly

    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();
    F();
    const auto end = std::chrono::steady_clock::now();
    std::cout
            << actionName << " in "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs ≈ "
            << (end - start) / 1ms << "ms ≈ " // almost equivalent form of the above, but
            << (end - start) / 1s << "s.\n";  // using milliseconds and seconds accordingly
}

/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood() {
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

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn() {
    int nvert = tree.index_data().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(iVertexSource) = 2;
    for (int j : tree.k_nearest_neighbors(iVertexSource, kNN)){
        closest(j) = 1;
    }
    cloud->addScalarQuantity(  std::to_string(kNN) + "-neighborhood of vertex " + std::to_string(iVertexSource), closest);
}

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
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

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities_impl(const std::string& name) {
    int nvert = tree.index_data().size();
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax]() {
        processPointCloud<FitT>( //WeightConstantFunc(1),
                                 WeightSmoothFunc(NSize),
                                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax]( int i, const FitT& fit){

            mean(i) = fit.kMean();
            kmax(i) = fit.k1();
            kmin(i) = fit.k2();

            normal.row( i ) = fit.primitiveGradient();
            dmin.row( i )   = fit.k1Direction();
            dmax.row( i )   = fit.k2Direction();
        });
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax]() {
                     cloud->addScalarQuantity(name + " - Mean Curvature", mean);
                     cloud->addScalarQuantity(name + " - K1", kmin);
                     cloud->addScalarQuantity(name + " - K2", kmax);

                     cloud->addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(
                             Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                             Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                             Scalar(2) * pointRadius);
                 });
}

/// Compute curvature using Covariance Plane fitting
/// \see estimateDifferentialQuantities_impl
void estimateCurvatureWithPlane() {
    estimateDifferentialQuantities_impl<Ponca::Basket<PPAdapter, WeightSmoothFunc,
            Ponca::CovariancePlaneFit, Ponca::CovariancePlaneSpaceDer,
            Ponca::CurvatureEstimator>>("PSS");
}

/// Compute curvature using APSS
/// \see estimateDifferentialQuantities_impl
void estimateCurvatureWithAPSS() {
    estimateDifferentialQuantities_impl<Ponca::Basket<PPAdapter, WeightSmoothFunc,
            Ponca::OrientedSphereFit, Ponca::OrientedSphereSpaceDer,
            Ponca::CurvatureEstimator>>("APSS");
}

/// Compute curvature using Algebraic Shape Operator
/// \see estimateDifferentialQuantities_impl
void estimateCurvatureWithASO() {
    estimateDifferentialQuantities_impl<Ponca::Basket<PPAdapter, WeightSmoothFunc,
            Ponca::OrientedSphereFit, Ponca::OrientedSphereSpaceDer, Ponca::MlsSphereFitDer,
            Ponca::CurvatureEstimator>>("ASO");
}

// Generate new smoothed point cloud using Covariance Plane Fitting
void smoothCovPlane() {

}

// Generate new smoothed point cloud using Algebraic Point Set Surfaces
void smoothAPSS() {

}

/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    ImGui::Text("Neighborhood collection");

    ImGui::InputInt("k-neighborhood size", &kNN);
    ImGui::InputFloat("neighborhood size", &NSize);
    ImGui::InputInt("source vertex", &iVertexSource);
    ImGui::SameLine();
    if (ImGui::Button("show knn")) colorizeKnn();
    ImGui::SameLine();
    if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood();

    ImGui::Separator();

    ImGui::Text("Differential estimators");
    if (ImGui::Button("Plane (PCA)"))  estimateCurvatureWithPlane();
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

    measureTime( "[libIGL] Load Armadillo", []()
    // For convenience: use libIGL to load a mesh, and store only the vertices location and normal vector
    {
        std::string filename = "assets/armadillo.obj";
        Eigen::MatrixXi meshF;
        igl::readOBJ(filename, cloudV, meshF);
        igl::per_vertex_normals(cloudV, meshF, cloudN);
    } );
    std::cout << "Vertex count: " << cloudV.rows() << std::endl;

    // Build Ponca KdTree
    measureTime( "[Ponca] Build KdTree", []() {
        buildKdTree(cloudV, cloudN, tree);
    });

    // Register the point cloud with Polyscope
    std::cout << "Starting polyscope... " << std::endl;
    cloud = polyscope::registerPointCloud("cloud", cloudV);
    cloud->setPointRadius(pointRadius);

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    return 0;
}