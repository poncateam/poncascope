#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"

#include "polyscopeSlicer.hpp"

#include <iostream>
#include <utility>

// This file defines all the main types + data shared across the application components
#include "./context.hpp"
#include "./io.hpp"

Context context;

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;

/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood() {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    Context::SmoothWeightFunc w(context.tree.points()[context.iVertexSource], context.NSize );

    closest(context.iVertexSource) = 2;
    context.doOnRangeNeighbors(context.iVertexSource, [w, &closest](auto &&neighborhood){
        for (int j : neighborhood){
            const auto &q = context.tree.points()[j];
            closest(j) = w( q ).first;
        }
    });

    context.cloud->addScalarQuantity(  "range neighborhood", closest);
}

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn() {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(context.iVertexSource) = 2;
    context.doOnKNeighbors(context.iVertexSource, [&closest](auto&& neighborhood){
        for (int j : neighborhood){
            closest(j) = 1;
        }
    });

    context.cloud->addScalarQuantity(  "knn neighborhood", closest);
}

/// Recompute K-Neighbor graph
void recomputeKnnGraph() {
    measureTime("[Ponca] Build KnnGraph", []() {
        delete context.knnGraph;
        context.knnGraph = new Context::KnnGraph(context.tree, context.kNNGraphK);
    });
}

using FitDry = Ponca::Basket<Context::PPAdapter, Context::SmoothWeightFunc, Ponca::DryFit>;

using FitPlane = Ponca::Basket<Context::PPAdapter, Context::SmoothWeightFunc, Ponca::CovariancePlaneFit>;
using FitPlaneDiff = Ponca::BasketDiff<
        FitPlane,
        Ponca::DiffType::FitSpaceDer,
        Ponca::CovariancePlaneDer,
        Ponca::CurvatureEstimatorDer, Ponca::NormalDerivativeWeingartenEstimator>;

using FitAPSS = Ponca::Basket<Context::PPAdapter, Context::SmoothWeightFunc, Ponca::OrientedSphereFit>;
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

using FitCNCUniform = Ponca::CNC<Context::PPAdapter, Ponca::TriangleGenerationMethod::UniformGeneration>;
using FitCNCIndep   = Ponca::CNC<Context::PPAdapter, Ponca::TriangleGenerationMethod::IndependentGeneration>;
using FitCNCHex     = Ponca::CNC<Context::PPAdapter, Ponca::TriangleGenerationMethod::HexagramGeneration>;
using FitCNCAvgHex  = Ponca::CNC<Context::PPAdapter, Ponca::TriangleGenerationMethod::AvgHexagramGeneration>;

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudMLS(const typename FitT::Scalar t, const Functor& f){

    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.mlsIter, Scalar(context.mlsEpsilon));
#pragma omp parallel for private (mls_evaluation_scheme)
    for (int i = 0; i < context.tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({context.tree.points()[i], t});

        context.doOnNeighbors(i, [&](const auto& rangeNeighbors){
            mls_evaluation_scheme.computeWithIds(fit, rangeNeighbors, context.tree.points());
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
    for (int i = 0; i < context.tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({context.tree.points()[i], t});

        std::vector<int> neighbors;
        neighbors.push_back(i);
        context.doOnNeighbors(i, [&neighbors](auto &&neighborhood){
            for (int j : neighborhood){
                neighbors.push_back(j);
            }
        });
        fit.computeWithIds(neighbors, context.tree.points());

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
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
        processPointCloudMLS<FitT>(context.NSize,
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                (const int i, const FitT& fit){

            mean(i)       = fit.kMean();
            kmax(i)       = fit.kmax();
            kmin(i)       = fit.kmin();
            dmin.row( i ) = fit.kminDirection();
            dmax.row( i ) = fit.kmaxDirection();
            normal.row(i) = fit.primitiveGradient();
            proj.row(i)   = fit.getNeighborFilter().evalPos() - context.tree.points()[i].pos();
        });
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
                     context.cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     context.cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                     context.cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                    context.cloud->addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                    context.cloud->addVectorQuantity(name + " - projection", proj, polyscope::VectorType::AMBIENT);

                 });
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantitiesCNC(const std::string& name) {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &dmin, &dmax]() {
        processPointCloudCNC<FitT>(context.NSize,
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
                     context.cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     context.cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                     context.cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                 });
}

/// Dry run: loop over all vertices + run MLS loops without computation
/// This function is useful to monitor the KdTree performances
inline void mlsDryRun() {
    measureTime( "[Ponca] Dry run MLS ", []() {
        processPointCloudMLS<FitDry>( context.NSize, [](int, const FitDry&){ });
    });
}

///Evaluate scalar field for generic FitType.
///// \tparam FitT Defines the type of estimator used for computation
template<typename FitT, bool isSigned = true>
Scalar evalScalarField_impl(const VectorType& input_pos)
{
    FitT fit;
    fit.setNeighborFilter({input_pos, context.NSize}); // weighting function using current pos (not input pos)
    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.mlsIter, Scalar(context.mlsEpsilon));
    auto res = mls_evaluation_scheme.computeWithIds(fit, context.tree.rangeNeighbors(input_pos, context.NSize),
        context.tree.points());

    if(!fit.isStable()) {
        // not enough neighbors (if far from the point cloud)
        return Scalar(0); //std::numeric_limits<Scalar>::max();
    }

    const Scalar current_value = isSigned ? fit.potential(input_pos) : std::abs(fit.potential(input_pos));
    // current_gradient = fit.primitiveGradient(input_pos);
    return current_value;
}

/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    callback_io(context);

    ImGui::Text("Acceleration Structure");
    bool knnGraphUIChanged = ImGui::Checkbox("Use KnnGraph", &context.useKnnGraph);
    if (context.useKnnGraph)
    {
        ImGui::SameLine();
        if (ImGui::InputInt("Graph k", &context.kNNGraphK) || knnGraphUIChanged) // recompute when activated or changed
            recomputeKnnGraph();
    }

    ImGui::Separator();
    ImGui::Text("Neighborhood queries");
    ImGui::Checkbox("Use Range Queries", &context.useRangeNei);
    ImGui::SameLine();
    if (context.useRangeNei)
        ImGui::InputFloat("neighborhood range size", &context.NSize);
    else
        ImGui::InputInt("k-neighborhood size", &context.kNN);

    ImGui::Separator();
    ImGui::InputInt("source vertex", &context.iVertexSource);
    ImGui::SameLine();
    if (context.useRangeNei) {
        if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood();
    }
    else
        if (ImGui::Button("show knn")) colorizeKnn();

    ImGui::Separator();
    ImGui::InputInt("Nb MLS Iterations", &context.mlsIter);
    ImGui::InputFloat("MLS Epsilon", &context.mlsEpsilon);
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
    ImGui::SliderFloat("Slice", &context.slice, 0, 1.0); ImGui::SameLine();
    ImGui::Checkbox("HD", &context.isHDSlicer);
    ImGui::RadioButton("X axis", &context.axis, 0); ImGui::SameLine();
    ImGui::RadioButton("Y axis", &context.axis, 1); ImGui::SameLine();
    ImGui::RadioButton("Z axis", &context.axis, 2);
    const char* items[] = { "ASO", "APSS", "PSS"};
    static int item_current = 0;
    ImGui::Combo("Fit function", &item_current, items, IM_ARRAYSIZE(items));
    if (ImGui::Button("Update"))
    {
      switch(item_current)
      {
        case 0: registerRegularSlicer("slicer", evalScalarField_impl<FitASO, true>   , context.lower, context.upper, context.isHDSlicer?1024:256, context.axis, context.slice); break;
        case 1: registerRegularSlicer("slicer", evalScalarField_impl<FitAPSS, true>  , context.lower, context.upper, context.isHDSlicer?1024:256, context.axis, context.slice); break;
        case 2: registerRegularSlicer("slicer", evalScalarField_impl<FitPlane, false>, context.lower, context.upper, context.isHDSlicer?1024:256, context.axis, context.slice); break;
      }
    }
    ImGui::SameLine();
    ImGui::PopItemWidth();
}

int main(int argc, char** argv) {
    // Options
    polyscope::options::autocenterStructures = false;
    polyscope::options::programName = "poncascope";
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();

    if (argc > 1)
    {
        loadFile(argv[1],context);
    }
    else
        loadFile("assets/armadillo.obj", context);

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    delete context.knnGraph;
    return EXIT_SUCCESS;
}
