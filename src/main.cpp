#include "polyscope/polyscope.h"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#include "polyscope/point_cloud.h"

#include "ImGuiFileDialog.h"

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
int iVertexSource    = 7;     /// < id of the selected point
int kNN              = 10;    /// < neighborhood size (knn)
int kNNGraphK        = 6;     /// < number of neighbors used to compute the knngraph
float NSize          = 0.1f;  /// < neighborhood size (euclidean)
int mlsIter          = 1;     /// < number of moving least squares iterations
float mlsEpsilon     = 0.001f; /// < motion distance stopping criterion for moving least squares
Scalar pointRadius   = 0.005; /// < display radius of the point cloud
bool useKnnGraph     = false; /// < use k-neighbor graph instead of kdtree
bool useRangeNei     = true;  /// < use range neighbors for estimators (or knn queries otherwise)
std::string lastPath = ".";   /// < last path used in file loader


// Slicer
float slice    = 0.f;
int axis       = 0;
bool isHDSlicer=false;
VectorType lower, upper;


/// Convenience function measuring and printing the processing time of F
template <typename Functor>
void measureTime( const std::string &actionName, const Functor& f){
    using namespace std::literals; // enables the usage of 24h instead of e.g. std::chrono::hours(24)

    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();
    f(); // run process
    const auto end = std::chrono::steady_clock::now();
    std::cout << actionName << " in " << (end - start) / 1ms << "ms.\n";
}

namespace internal {
    //! \brief Dispatch a lambda on a range neighbors query over either the KnnGraph or the KdTree
    template <typename Functor>
    void doOnRangeNeighbors(const int i, const Functor& f) {
        if (useKnnGraph)
            f(knnGraph->rangeNeighbors(i, NSize));
        else
            f(tree.rangeNeighbors(i, NSize));
    }

    //! \brief Dispatch a lambda on a range neighbors query over either the KnnGraph or the KdTree
    template <typename Functor>
    void doOnKNeighbors(const int i, const Functor& f) {
        f(tree.kNearestNeighbors(i, kNN));
    }
}

//! \brief Dispatch a lambda on either a range or a knn query depending on the UI.
template <typename Functor>
void doOnNeighbors(const int i, const Functor& f) {
    useRangeNei ? internal::doOnRangeNeighbors(i,f) : internal::doOnKNeighbors(i, f);
}


/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood() {
    int nvert = int(tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    SmoothWeightFunc w(tree.points()[iVertexSource], NSize );

    closest(iVertexSource) = 2;
    internal::doOnRangeNeighbors(iVertexSource, [w, &closest](auto &&neighborhood){
        for (int j : neighborhood){
            const auto &q = tree.points()[j];
            closest(j) = w( q ).first;
        }
    });

    cloud->addScalarQuantity(  "range neighborhood", closest);
}

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn() {
    int nvert = int(tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(iVertexSource) = 2;
    internal::doOnKNeighbors(iVertexSource, [&closest](auto&& neighborhood){
        for (int j : neighborhood){
            closest(j) = 1;
        }
    });

    cloud->addScalarQuantity(  "knn neighborhood", closest);
}

/// Recompute K-Neighbor graph
void recomputeKnnGraph() {
    measureTime("[Ponca] Build KnnGraph", []() {
        delete knnGraph;
        knnGraph = new KnnGraph(tree, kNNGraphK);
    });
}

using FitDry = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::DryFit>;

using FitPlane = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::CovariancePlaneFit>;
using FitPlaneDiff = Ponca::BasketDiff<
        FitPlane,
        Ponca::DiffType::FitSpaceDer,
        Ponca::CovariancePlaneDer,
        Ponca::CurvatureEstimatorDer, Ponca::NormalDerivativeWeingartenEstimator>;

using FitAPSS = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::OrientedSphereFit>;
using FitAPSSDiff = Ponca::BasketDiff<
        FitAPSS,
        Ponca::DiffType::FitSpaceDer,
        Ponca::OrientedSphereDer,
        Ponca::CurvatureEstimatorDer, Ponca::NormalDerivativeWeingartenEstimator,
        Ponca::WeingartenCurvatureEstimatorDer>;

using FitASO = FitAPSS;
using FitASODiff = Ponca::BasketDiff<
        FitASO,
        Ponca::DiffType::FitSpaceDer,
        Ponca::OrientedSphereDer, Ponca::MlsSphereFitDer,
        Ponca::CurvatureEstimatorDer, Ponca::NormalDerivativeWeingartenEstimator,
        Ponca::WeingartenCurvatureEstimatorDer>;

using FitCNCUniform = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::UniformGeneration>;
using FitCNCIndep   = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::IndependentGeneration>;
using FitCNCHex     = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::HexagramGeneration>;
using FitCNCAvgHex  = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::AvgHexagramGeneration>;

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudMLS(const typename FitT::Scalar t, const Functor& f){

    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (mlsIter, Scalar(mlsEpsilon));
#pragma omp parallel for private (mls_evaluation_scheme)
    for (int i = 0; i < tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({tree.points()[i], t});

        doOnNeighbors(i, [&](const auto& rangeNeighbors){
            mls_evaluation_scheme.computeWithIds(fit, rangeNeighbors, tree.points());
        });

        if (fit.isStable()) {
            f(i, fit);
        } else {
            std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
        }
    }
}

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudCNC(const typename FitT::Scalar t, const Functor& f){
#pragma omp parallel for
    for (int i = 0; i < tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({tree.points()[i], t});

        std::vector<int> neighbors;
        neighbors.push_back(i);
        doOnNeighbors(i, [&neighbors](auto &&neighborhood){
            for (int j : neighborhood){
                neighbors.push_back(j);
            }
        });
        fit.computeWithIds(neighbors, tree.points());

        if (fit.isStable()) {
            f(i, fit);
        } else {
            std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
        }
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities(const std::string& name) {
    int nvert = int(tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloudMLS<FitT>(NSize,
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                (const int i, const FitT& fit){

            mean(i)       = fit.kMean();
            kmax(i)       = fit.kmax();
            kmin(i)       = fit.kmin();
            dmin.row( i ) = fit.kminDirection();
            dmax.row( i ) = fit.kmaxDirection();
            normal.row(i) = fit.primitiveGradient();
            proj.row(i)   = fit.getNeighborFilter().evalPos() - tree.points()[i].pos();
        });
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
                     cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * pointRadius);
                    cloud->addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(Scalar(2) * pointRadius);
                    cloud->addVectorQuantity(name + " - projection", proj, polyscope::VectorType::AMBIENT);

                 });
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantitiesCNC(const std::string& name) {
    int nvert = int(tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &dmin, &dmax]() {
        processPointCloudCNC<FitT>(NSize,
                                [&mean, &kmin, &kmax, &dmin, &dmax]
                                (const int i, const FitT& fit){

            mean(i)         = fit.kMean();
            kmax(i)         = fit.kmax();
            kmin(i)         = fit.kmin();
            dmin.row( i )   = fit.kminDirection();
            dmax.row( i )   = fit.kmaxDirection();
        });
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &dmin, &dmax]() {
                     cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * pointRadius);
                     cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * pointRadius);
                 });
}

/// Dry run: loop over all vertices + run MLS loops without computation
/// This function is useful to monitor the KdTree performances
inline void mlsDryRun() {
    measureTime( "[Ponca] Dry run MLS ", []() {
        processPointCloudMLS<FitDry>( NSize, [](int, const FitDry&){ });
    });
}

///Evaluate scalar field for generic FitType.
///// \tparam FitT Defines the type of estimator used for computation
template<typename FitT, bool isSigned = true>
Scalar evalScalarField_impl(const VectorType& input_pos)
{
    FitT fit;
    fit.setNeighborFilter({input_pos, NSize}); // weighting function using current pos (not input pos)
    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (mlsIter, Scalar(mlsEpsilon));
    auto res = mls_evaluation_scheme.computeWithIds(fit, tree.rangeNeighbors(input_pos, NSize), tree.points());

    if(!fit.isStable()) {
        // not enough neighbors (if far from the point cloud)
        return Scalar(0); //std::numeric_limits<Scalar>::max();
    }

    const Scalar current_value = isSigned ? fit.potential(input_pos) : std::abs(fit.potential(input_pos));
    // current_gradient = fit.primitiveGradient(input_pos);
    return current_value;
}


bool loadFile(const std::string& path)
{
    std::cout << "[Poncascope] Load file " << path << std::endl;

    std::cout << cloudV.size() << std::endl;
    std::cout << cloudN.size() << std::endl;
    // first block: output cloudV and cloudN
    Eigen::MatrixXd newCloud, newNormals;
    {
        bool worked;
        Eigen::MatrixXi meshF;
        measureTime( "[libIGL] Load Armadillo", [path, &newCloud, &meshF, &worked]()
        // For convenience: use libIGL to load a mesh, and store only the vertices location and normal vector
        {
            const std::string filename = path.c_str();
            worked = igl::readOBJ(filename, newCloud, meshF);
        } );

        if (worked) {
            if (meshF.cols()==3) // we have a triangle mesh
            {
                igl::per_vertex_normals(newCloud, meshF, newNormals);
            }
        }
        else {
            std::cerr << "[libIGL] An error occurred when loading file " << path
                      << std::endl;
            return false;
        }
    }

    // Check if normals have been properly loaded
    /// \fixme : should not abort, but rather compute normals using Ponca.
    {
        int nbUnitNormal = int(newNormals.rowwise().squaredNorm().sum());
        if ( nbUnitNormal != newCloud.rows() ) {
            std::cerr << "[Poncascope] Point cloud has no normals, aborting" << std::endl;
            return false;
        }
    }

    cloudV = newCloud;
    // no need to delete the previous cloud, polyscope handles it
    cloud = polyscope::registerPointCloud("cloud", cloudV);
    cloudN = newNormals;

    std::cout << cloudV.size() << std::endl;
    std::cout << cloudN.size() << std::endl;

    // Bounding Box (used in the slicer)
    lower = cloudV.colwise().minCoeff();
    upper = cloudV.colwise().maxCoeff();

    // Build Ponca KdTree
    measureTime( "[Ponca] Build KdTree", []() {
        buildKdTree(cloudV, cloudN, tree);
    });


    // Compute default point and neighborhood size according to the mean density
    measureTime( "[Ponca] Compute point radius according to mean knn distance", []() {
        Scalar cloudMDist = 0;
        constexpr Scalar pointSizeFactor = 0.25;
        constexpr Scalar scaleFactor = 5;
#pragma omp parallel for
        for (int i = 0; i < tree.samples().size(); ++i)
        {
            Scalar pointMDist = 0;
            VectorType p = tree.points()[i].pos();
            internal::doOnKNeighbors(i, [&pointMDist,p](auto&& neighborhood){
                for (int j : neighborhood){
                    pointMDist += (p-tree.points()[j].pos()).norm();
                }
            });
#pragma omp critical
            cloudMDist += pointMDist/Scalar(kNN);
        }
        pointRadius = pointSizeFactor * cloudMDist/Scalar(tree.samples().size());
        NSize = scaleFactor * cloudMDist/Scalar(tree.samples().size());
    });



    // Register the point cloud with Polyscope
    cloud->setPointRadius(pointRadius);
    polyscope::requestRedraw();

    useKnnGraph = false;

    std::cout << "[Poncascope] Loading file succeeded"<< std::endl;

    return true;
}

/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    // open Dialog Simple
    if (ImGui::Button("Open File Dialog")) {
        IGFD::FileDialogConfig config;
        config.path = lastPath;
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", config);
    }
    // display
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            // std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();

            if (loadFile(filePathName)) lastPath = filePathName;
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::Text("Acceleration Structure");
    bool knnGraphUIChanged = ImGui::Checkbox("Use KnnGraph", &useKnnGraph);
    if (useKnnGraph)
    {
        ImGui::SameLine();
        if (ImGui::InputInt("Graph k", &kNNGraphK) || knnGraphUIChanged) // recompute when activated or changed
            recomputeKnnGraph();
    }

    ImGui::Separator();
    ImGui::Text("Neighborhood queries");
    ImGui::Checkbox("Use Range Queries", &useRangeNei);
    ImGui::SameLine();
    if (useRangeNei)
        ImGui::InputFloat("neighborhood range size", &NSize);
    else
        ImGui::InputInt("k-neighborhood size", &kNN);

    ImGui::Separator();
    ImGui::InputInt("source vertex", &iVertexSource);
    ImGui::SameLine();
    if (useRangeNei) {
        if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood();
    }
    else
        if (ImGui::Button("show knn")) colorizeKnn();

    ImGui::Separator();
    ImGui::InputInt("Nb MLS Iterations", &mlsIter);
    ImGui::InputFloat("MLS Epsilon", &mlsEpsilon);
    ImGui::Separator();

    ImGui::Text("Differential estimators");
    if (ImGui::Button("Dry Run"))  mlsDryRun();
    ImGui::SameLine();
    if (ImGui::Button("Plane (PCA)")) // Compute curvature using Covariance Plane fitting
        estimateDifferentialQuantities<FitPlaneDiff>("PSS");
    ImGui::SameLine();
    if (ImGui::Button("APSS")) // Compute curvature using APSS
        estimateDifferentialQuantities<FitAPSSDiff>("APSS");
    ImGui::SameLine();
    if (ImGui::Button("ASO")) // Compute curvature using Algebraic Shape Operator
        estimateDifferentialQuantities<FitASODiff>("ASO");

    ImGui::Text("Corrected Normal Current estimator");
    if (ImGui::Button("Uniform"))
        estimateDifferentialQuantitiesCNC<FitCNCUniform>("CNC - Uniform");
    ImGui::SameLine();
    if (ImGui::Button("Independent"))
        estimateDifferentialQuantitiesCNC<FitCNCIndep>("CNC - Independent");
    ImGui::SameLine();
    if (ImGui::Button("Hexagram"))
        estimateDifferentialQuantitiesCNC<FitCNCHex>("CNC - Hexagram");
    ImGui::SameLine();
    if (ImGui::Button("AvgHexagram"))
        estimateDifferentialQuantitiesCNC<FitCNCAvgHex>("CNC - AvgHexagram");

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
        case 0: registerRegularSlicer("slicer", evalScalarField_impl<FitASO, true>   , lower, upper, isHDSlicer?1024:256, axis, slice); break;
        case 1: registerRegularSlicer("slicer", evalScalarField_impl<FitAPSS, true>  , lower, upper, isHDSlicer?1024:256, axis, slice); break;
        case 2: registerRegularSlicer("slicer", evalScalarField_impl<FitPlane, false>, lower, upper, isHDSlicer?1024:256, axis, slice); break;
      }
    }
    ImGui::SameLine();
    ImGui::PopItemWidth();
}

int main(int /*argc*/, char** /*argv*/) {
    // Options
    polyscope::options::autocenterStructures = false;
    polyscope::options::programName = "poncascope";
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();

    loadFile("assets/armadillo.obj");

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    delete knnGraph;
    return EXIT_SUCCESS;
}
