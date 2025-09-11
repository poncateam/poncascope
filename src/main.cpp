#include "polyscope/polyscope.h"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#include "polyscope/point_cloud.h"

#include <Ponca/Fitting>
#include <Ponca/SpatialPartitioning>
#include "poncaAdapters.hpp"
#include "polyscopeSlicer.hpp"

#include <iostream>
#include <utility>
#include <chrono>

// Types definition
using Scalar             = double;
using VectorType         = Eigen::Vector<Scalar, 3>;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTreeSparse<PPAdapter>;
using KnnGraph           = Ponca::KnnGraph<PPAdapter>;
using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;
//using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::ExpWeightKernel<Scalar> >;

// Variables
Eigen::MatrixXd cloudV, cloudN;
KdTree tree;
KnnGraph* knnGraph {nullptr};
polyscope::PointCloud* cloud = nullptr;

// Options for algorithms
int iVertexSource  = 7;     /// < id of the selected point
int kNN            = 10;    /// < neighborhood size (knn)
float NSize        = 0.1;   /// < neighborhood size (euclidean)
int mlsIter        = 3;     /// < number of moving least squares iterations
Scalar pointRadius = 0.005; /// < display radius of the point cloud
bool useKnnGraph   = false; /// < use k-neighbor graph instead of kdtree



// Slicer
float slice    = 0.f;
int axis       = 0;
bool isHDSlicer=false;
VectorType lower, upper;


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

template <typename Functor>
void processRangeNeighbors(int i, Functor f){
    if(useKnnGraph)
        for (int j : knnGraph->range_neighbors(i, NSize)){
            f(j);
        }
    else
        for (int j : tree.range_neighbors(i, NSize)){
            f(j);
        }
}

/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood() {
    int nvert = tree.samples().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    delete knnGraph;
    knnGraph = new KnnGraph (tree, kNN);

    SmoothWeightFunc w(VectorType::Zero(), NSize );

    closest(iVertexSource) = 2;
    const auto &p = tree.points()[iVertexSource];
    processRangeNeighbors(iVertexSource, [w,p,&closest](int j){
        const auto &q = tree.points()[j];
        closest(j) = w.w( q.pos() - p.pos(), q ).first;
    });

    cloud->addScalarQuantity(  "range neighborhood", closest);
}

/// Recompute K-Neighbor graph
void recomputeKnnGraph() {
    if(useKnnGraph) {
        measureTime("[Ponca] Build KnnGraph", []() {
            delete knnGraph;
            knnGraph = new KnnGraph(tree, kNN);
        });
    }
}

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn() {
    int nvert = tree.samples().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(iVertexSource) = 2;
    processRangeNeighbors(iVertexSource, [&closest](int j){
        closest(j) = 1;
    });
    cloud->addScalarQuantity(  "knn neighborhood", closest);
}

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloud(const typename FitT::Scalar t, Functor f){
#pragma omp parallel for
    for (int i = 0; i < tree.samples().size(); ++i) {
        VectorType pos = tree.points()[i].pos();

        for( int mm = 0; mm < mlsIter; ++mm) {
            FitT fit;
            fit.setWeightFunc({pos, t});
            fit.init();

            processRangeNeighbors(i, [&fit](int j){
                fit.addNeighbor(tree.points()[j]);
            });

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
    int nvert = tree.samples().size();
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloud<FitT>(NSize,
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                ( int i, const FitT& fit, const VectorType& mlsPos){

            mean(i) = fit.kMean();
            kmax(i) = fit.kmax();
            kmin(i) = fit.kmin();

            normal.row( i ) = fit.primitiveGradient();
            dmin.row( i )   = fit.kminDirection();
            dmax.row( i )   = fit.kmaxDirection();

            proj.row( i )   = mlsPos - tree.points()[i].pos();
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

using FitDry = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::DryFit>;

using FitPlane = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::CovariancePlaneFit>;
using FitPlaneDiff = Ponca::BasketDiff<
        FitPlane,
        Ponca::DiffType::FitSpaceDer,
        Ponca::CovariancePlaneDer,
        Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

using FitAPSS = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::OrientedSphereFit>;
using FitAPSSDiff = Ponca::BasketDiff<
        FitAPSS,
        Ponca::DiffType::FitSpaceDer,
        Ponca::OrientedSphereDer,
        Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

using FitASO = FitAPSS;
using FitASODiff = Ponca::BasketDiff<
        FitASO,
        Ponca::DiffType::FitSpaceDer,
        Ponca::OrientedSphereDer, Ponca::MlsSphereFitDer,
        Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

/// Compute curvature using Covariance Plane fitting
/// \see estimateDifferentialQuantities_impl
void estimateDifferentialQuantitiesWithPlane() {
    estimateDifferentialQuantities_impl<FitPlaneDiff>("PSS");
}

/// Compute curvature using APSS
/// \see estimateDifferentialQuantities_impl
inline void estimateDifferentialQuantitiesWithAPSS() {
    estimateDifferentialQuantities_impl<FitAPSSDiff>("APSS");
}

/// Compute curvature using Algebraic Shape Operator
/// \see estimateDifferentialQuantities_impl
inline void estimateDifferentialQuantitiesWithASO() {
    estimateDifferentialQuantities_impl<FitASODiff>("ASO");
}

/// Dry run: loop over all vertices + run MLS loops without computation
/// This function is useful to monitor the KdTree performances
inline void mlsDryRun() {
    measureTime( "[Ponca] Dry run MLS ", []() {
        processPointCloud<FitDry>( NSize, [](int, const FitDry&, const VectorType& ){ });
    });
}

///Evaluate scalar field for generic FitType.
///// \tparam FitT Defines the type of estimator used for computation
template<typename FitT, bool isSigned = true>
Scalar evalScalarField_impl(const VectorType& input_pos)
{
    VectorType current_pos = input_pos;
    Scalar current_value = std::numeric_limits<Scalar>::max();
    for(int mm = 0; mm < mlsIter; ++mm)
    {
            FitT fit;
            fit.setWeightFunc({current_pos, NSize}); // weighting function using current pos (not input pos)
            auto res = fit.computeWithIds(tree.range_neighbors(current_pos, NSize), tree.points());
            if(res == Ponca::STABLE) {
            current_pos = fit.project(input_pos); // always project input pos
            current_value = isSigned ? fit.potential(input_pos) : std::abs(fit.potential(input_pos));
            // current_gradient = fit.primitiveGradient(input_pos);
        } else {
            // not enough neighbors (if far from the point cloud)
            return .0;//std::numeric_limits<Scalar>::max();
        }
    }
    return current_value;
}

/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    ImGui::Text("Neighborhood collection");
    ImGui::SameLine();
    if(ImGui::Checkbox("Use KnnGraph", &useKnnGraph)) recomputeKnnGraph();

    if(ImGui::InputInt("k-neighborhood size", &kNN)) recomputeKnnGraph();
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
    
    ImGui::Separator();
  
    ImGui::Text("Implicit function slicer");
    ImGui::SliderFloat("Slice", &slice, 0, 1.0); ImGui::SameLine();
    ImGui::Checkbox("HD", &isHDSlicer);
    ImGui::RadioButton("X axis", &axis, 0); ImGui::SameLine();
    ImGui::RadioButton("Y axis", &axis, 1); ImGui::SameLine();
    ImGui::RadioButton("Z axis", &axis, 2);
    const char* items[] = { "ASO", "APSS", "PSS"};
    static int item_current = 0;
    ImGui::Combo("Fit function", &item_current, items, IM_ARRAYSIZE(items));
    if (ImGui::Button("Update"))
    {
      switch(item_current)
      {
        case 0: registerRegularSlicer("slicer", evalScalarField_impl<FitASO, true>,lower, upper, isHDSlicer?1024:256, axis, slice); break;
        case 1: registerRegularSlicer("slicer", evalScalarField_impl<FitAPSS, true>,lower, upper, isHDSlicer?1024:256, axis, slice); break;
        case 2: registerRegularSlicer("slicer", evalScalarField_impl<FitPlane, false>,lower, upper, isHDSlicer?1024:256, axis, slice); break;
      }
    }
    ImGui::SameLine();
    ImGui::PopItemWidth();
}

int main(int argc, char **argv) {
    // Options
    polyscope::options::autocenterStructures = false;
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

    //Bounding Box (used in the slicer)
    lower = cloudV.colwise().minCoeff();
    upper = cloudV.colwise().maxCoeff();
  
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

    delete knnGraph;
    return EXIT_SUCCESS;
}
