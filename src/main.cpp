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
using VectorType         = Eigen::Vector<Scalar, 3>;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTree<PPAdapter>;
using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;

// Variables
Eigen::MatrixXd cloudV, cloudN;
KdTree tree;
polyscope::PointCloud* cloud = nullptr;

// Options for algorithms
int iVertexSource  = 7;     /// < id of the selected point
int kNN            = 10;    /// < neighborhood size (knn)
float NSize        = 0.1;   /// < neighborhood size (euclidean)
int mlsIter        = 3;     /// < number of moving least squares iterations
Scalar pointRadius = 0.005; /// < display radius of the point cloud


/// Convenience function measuring and printing the processing time of F
template <typename Functor>
void measureTime( const std::string &actionName, Functor F ){
    using namespace std::literals; // enables the usage of 24h instead of e.g. std::chrono::hours(24)

    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();
    F(); // run process
    const auto end = std::chrono::steady_clock::now();
    std::cout << actionName << " in " << (end - start) / 1ms << "ms.\n";
}

/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood() {
    int nvert = tree.index_data().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    SmoothWeightFunc w(NSize );

    closest(iVertexSource) = 2;
    const auto &p = tree.point_data()[iVertexSource];
    for (int j : tree.range_neighbors(iVertexSource, NSize)){
        const auto &q = tree.point_data()[j];
        closest(j) = w.w( q.pos() - p.pos(), q );
    }
    cloud->addScalarQuantity(  "range neighborhood", closest);
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
    cloud->addScalarQuantity(  "knn neighborhood", closest);
}

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloud(const typename FitT::WeightFunction& w, Functor f){

    int nvert = tree.index_data().size();

#pragma omp parallel for
    for (int i = 0; i < nvert; ++i) {
        VectorType pos = tree.point_data()[i].pos();

        for( int mm = 0; mm < mlsIter; ++mm) {
            FitT fit;
            fit.setWeightFunc(w);
            fit.init( pos );

            for (int j : tree.range_neighbors(i, NSize))
                fit.addNeighbor(tree.point_data()[j]);

            if (fit.finalize() == Ponca::STABLE){
                pos = fit.project( pos );
                if ( mm == mlsIter -1 ) // last mls step, calling functor
                    f(i, fit, pos);
            }
            else {
                std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
                break;
            }
        }
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities_impl(const std::string& name) {
    int nvert = tree.index_data().size();
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloud<FitT>(SmoothWeightFunc(NSize),
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                ( int i, const FitT& fit, const VectorType& mlsPos){

            mean(i) = fit.kMean();
            kmax(i) = fit.k1();
            kmin(i) = fit.k2();

            normal.row( i ) = fit.primitiveGradient();
            dmin.row( i )   = fit.k1Direction();
            dmax.row( i )   = fit.k2Direction();

            proj.row( i )   = mlsPos - tree.point_data()[i].pos();
        });
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
                     cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});

                     cloud->addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(
                             Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                             Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                             Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - projection", proj, polyscope::VectorType::AMBIENT);
                 });
}

/// Compute curvature using Covariance Plane fitting
/// \see estimateDifferentialQuantities_impl
void estimateDifferentialQuantitiesWithPlane() {
    estimateDifferentialQuantities_impl<Ponca::Basket<PPAdapter, SmoothWeightFunc,
            Ponca::CovariancePlaneFit, Ponca::CovariancePlaneSpaceDer,
            Ponca::CurvatureEstimator>>("PSS");
}

/// Compute curvature using APSS
/// \see estimateDifferentialQuantities_impl
void estimateDifferentialQuantitiesWithAPSS() {
    estimateDifferentialQuantities_impl<Ponca::Basket<PPAdapter, SmoothWeightFunc,
            Ponca::OrientedSphereFit, Ponca::OrientedSphereSpaceDer,
            Ponca::CurvatureEstimator>>("APSS");
}

/// Compute curvature using Algebraic Shape Operator
/// \see estimateDifferentialQuantities_impl
void estimateDifferentialQuantitiesWithASO() {
    estimateDifferentialQuantities_impl<Ponca::Basket<PPAdapter, SmoothWeightFunc,
            Ponca::OrientedSphereFit, Ponca::OrientedSphereSpaceDer, Ponca::MlsSphereFitDer,
            Ponca::CurvatureEstimator>>("ASO");
}

/// Dry run: loop over all vertices + run MLS loops without computation
/// This function is useful to monitor the KdTree performances
void mlsDryRun() {
    using DryFit = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::DryFit>;
    measureTime( "[Ponca] Dry run MLS ", []() {
                     processPointCloud<DryFit>(
                             SmoothWeightFunc(NSize), [](int, const DryFit&, const VectorType& ){ });
    });
}


/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    ImGui::Text("Neighborhood collection");

    ImGui::InputInt("k-neighborhood size", &kNN);
    ImGui::InputFloat("neighborhood size", &NSize);
    ImGui::InputInt("source vertex", &iVertexSource);
    ImGui::InputInt("Nb MLS Iterations", &mlsIter);
    ImGui::SameLine();
    if (ImGui::Button("show knn")) colorizeKnn();
    ImGui::SameLine();
    if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood();

    ImGui::Separator();

    ImGui::Text("Differential estimators");
    if (ImGui::Button("Dry Run"))  mlsDryRun();
    ImGui::SameLine();
    if (ImGui::Button("Plane (PCA)")) estimateDifferentialQuantitiesWithPlane();
    ImGui::SameLine();
    if (ImGui::Button("APSS")) estimateDifferentialQuantitiesWithAPSS();
    ImGui::SameLine();
    if (ImGui::Button("ASO")) estimateDifferentialQuantitiesWithASO();

    ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
    // Options
    polyscope::options::autocenterStructures = true;
    polyscope::options::programName = "poncascope";
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

    // Check if normals have been properly loaded
    int nbUnitNormal = cloudN.rowwise().squaredNorm().sum();
    if ( nbUnitNormal != cloudV.rows() ) {
        std::cerr << "[libIGL] An error occurred when computing the normal vectors from the mesh. Aborting..."
                  << std::endl;
        return EXIT_FAILURE;
    }

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

    return EXIT_SUCCESS;
}