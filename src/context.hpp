#pragma once

#include <Ponca/Ponca>

#include "poncaAdapters.hpp"

#include <chrono>

namespace polyscope
{
    class PointCloud;
}

struct Context
{
    // Types definition
    using Scalar             = double;
    using VectorType         = Eigen::Vector<Scalar, 3>;
    using PPAdapter          = BlockPointAdapter<Scalar>;
    using KdTree             = Ponca::KdTreeSparse<PPAdapter>;
    using KnnGraph           = Ponca::KnnGraph<PPAdapter>;
    using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;
    //using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::ExpWeightKernel<Scalar> >;


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
    std::string loadPath = ".";   /// < last path used in file loader
    std::string savePath = "";   /// < last path used in file loader


    // Slicer
    float slice    = 0.f;
    int axis       = 0;
    bool isHDSlicer=false;
    VectorType lower, upper;

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

    //! \brief Dispatch a lambda on either a range or a knn query depending on the UI.
    template <typename Functor>
    void doOnNeighbors(const int i, const Functor& f) {
        useRangeNei ? doOnRangeNeighbors(i,f) : doOnKNeighbors(i, f);
    }
};


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
