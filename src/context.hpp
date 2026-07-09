#pragma once

#include <Ponca/Ponca>

#include "poncaAdapters.hpp"
#include "meshGraph.h"

#include <chrono>

#include "polyscope/scalar_quantity.h"
#include "polyscope/point_cloud.h"


struct Context
{
    struct Types {
        // Types definition
        using Scalar             = double;
        using VectorType         = Eigen::Vector<Scalar, 3>;
        using PPAdapter          = BlockPointAdapter<Scalar>;
        using KdTree             = Ponca::KdTreeSparse<PPAdapter>;
        using NeighborGraph      = Ponca::NeighborGraph<PPAdapter>;
        using KnnGraph           = Ponca::KnnGraph<PPAdapter>;
        using MeshGraph          = MeshNeighborGraph<PPAdapter>;
        using SmoothWeightFunc   = Ponca::DistWeightFilter<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;

        using Factory = Ponca::Factory<PPAdapter, SmoothWeightFunc, Ponca::FitSpaceDer>;
        using FitDry = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::DryFit>;

        using FitPlane = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::CovariancePlaneFit>;
        using FitPlaneDiff = Ponca::BasketDiff<
                FitPlane,
                Ponca::DiffType::FitSpaceDer,
                Ponca::CovariancePlaneDer,
                Ponca::NormalDerivativeWeingartenEstimator,
                Ponca::WeingartenCurvatureEstimatorDer>;

        using FitAPSS = Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::OrientedSphereFit>;
        using FitAPSSDiff = Ponca::BasketDiff<
                FitAPSS,
                Ponca::DiffType::FitSpaceDer,
                Ponca::OrientedSphereDer,
                Ponca::NormalDerivativeWeingartenEstimator,
                Ponca::WeingartenCurvatureEstimatorDer>;

        using FitASO = FitAPSS;
        using FitASODiff = Ponca::BasketDiff<
                FitASO,
                Ponca::DiffType::FitSpaceDer,
                Ponca::OrientedSphereDer, Ponca::MlsSphereFitDer,
                Ponca::NormalDerivativeWeingartenEstimator,
                Ponca::WeingartenCurvatureEstimatorDer>;

        using FitCNCUniform = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::UniformGeneration>;
        using FitCNCIndep   = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::IndependentGeneration>;
        using FitCNCHex     = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::HexagramGeneration>;
        using FitCNCAvgHex  = Ponca::CNC<PPAdapter, Ponca::TriangleGenerationMethod::AvgHexagramGeneration>;
    };

    struct Asset {
        // Variables
        Eigen::MatrixXd cloudV, cloudN;
        Eigen::MatrixXi meshF; // might be empty if the object loaded is not a mesh
        Types::KdTree tree;
        Types::KnnGraph* knnGraph {nullptr};
        Types::NeighborGraph* neiGraph {nullptr};
        Types::MeshGraph* meshGraph {nullptr};

        polyscope::PointCloud* cloud = nullptr;
        Types::VectorType lower, upper;

        void clear()
        {
            delete knnGraph;  knnGraph = nullptr;
            delete neiGraph;  neiGraph = nullptr;
            delete meshGraph; meshGraph = nullptr;
            cloudV = Eigen::MatrixXd();
            cloudN = Eigen::MatrixXd();
            meshF  = Eigen::MatrixXi();
            lower  = Types::VectorType();
            upper  = Types::VectorType();
            scalarQuantites.clear();
            vectorQuantites.clear();
        }

        ~Asset()
        {
            clear();
        }


        // An abstraction of the polyscope quantities that is used to track fields and simplify saving
        template <typename QtyType>
        struct QuantityHandler
        {
            QtyType* ptr {nullptr};
            std::string name;
            bool save {true};
        };
        using ScalarQuantityHandler = QuantityHandler<polyscope::PointCloudScalarQuantity>;
        std::vector<ScalarQuantityHandler> scalarQuantites;
        using VectorQuantityHandler = QuantityHandler<polyscope::PointCloudVectorQuantity>;
        std::vector<VectorQuantityHandler> vectorQuantites;

        template <typename T>
        polyscope::PointCloudScalarQuantity* addScalarQuantity(std::string name, const T& data,
                                                               polyscope::DataType type = polyscope::DataType::STANDARD)
        {
            ScalarQuantityHandler h;
            h.ptr = cloud->addScalarQuantity(name, data);
            h.name = name;
            scalarQuantites.push_back(h);
            return h.ptr;
        }

        template <typename T>
        polyscope::PointCloudVectorQuantity* addVectorQuantity(std::string name, const T& data,
                                                               polyscope::VectorType type = polyscope::VectorType::STANDARD)
        {
            VectorQuantityHandler h;
            h.ptr = cloud->addVectorQuantity(name, data, type);
            h.name = name;
            vectorQuantites.push_back(h);
            return h.ptr;
        }

    } asset;

    struct ComputeOptions
    {
        // Options for algorithms
        int iVertexSource    = 7;     /// < id of the selected point
        int kNN              = 10;    /// < neighborhood size (knn)
        float NSize          = 0.1f;  /// < neighborhood size (euclidean)
        int mlsIter          = 1;     /// < number of moving least squares iterations
        float mlsEpsilon     = 0.001f; /// < motion distance stopping criterion for moving least squares
        Types::Scalar pointRadius = 0.005; /// < display radius of the point cloud
        bool useRangeNei     = true;  /// < use range neighbors for estimators (or knn queries otherwise)
    } computeOpts;

    struct DataStructureOptions
    {
        enum TopologyMode
        {
            None              = 0, // Kdtree
            KnnGraph          = 1, // KnnGraph
            NeighborGraph     = 2, // Range NeighborGraph
            MeshGraph         = 3, // Mesh topology
            TopologyModeCount = 4  // not a true mode per se, but the number of modes
        } topoMode {None};
        static constexpr const char* TopologyNames[] =
            { "None", "Knn Graph", "Neighbor Graph", "Mesh Topology"};
        int kNNGraphK        = 6;         /// < number of neighbors used to compute the knngraph
        float neighborGraphRange =  0.1f; /// < range used to build the neighbor graph
        bool displayTopology = false;
    } dataStructureOptions;

    struct SlicerOptions
    {
        float slice    = 0.f;
        int axis       = 0;
        bool isHDSlicer=false;
    } slicerOptions;

    struct IOOptions
    {
        std::string loadPath = ".";   /// < last path used in file loader
        std::string savePath = "";    /// < last path used in file loader
    } ioOptions;

    //! \brief Dispatch a lambda on a range neighbors query over either the KnnGraph or the KdTree
    template <typename Functor>
    void doOnRangeNeighbors(const int i, const Functor& f) {
        switch (dataStructureOptions.topoMode)
        {
        case DataStructureOptions::TopologyMode::None:
            f(asset.tree.rangeNeighbors(i, computeOpts.NSize));
            break;
        case DataStructureOptions::TopologyMode::KnnGraph:
            f(asset.knnGraph->rangeNeighbors(i, computeOpts.NSize));
            break;
        case DataStructureOptions::TopologyMode::NeighborGraph:
            f(asset.neiGraph->rangeNeighbors(i, computeOpts.NSize));
            break;
        case DataStructureOptions::TopologyMode::MeshGraph:
            f(asset.meshGraph->rangeNeighbors(i, computeOpts.NSize));
            break;
        case DataStructureOptions::TopologyModeCount:
        default:
            break;
        }
    }

    //! \brief Dispatch a lambda on the active graph if it exists (otherwise does nothing)
    template <typename Functor>
    void doOnGraph(const Functor& f) {
        switch (dataStructureOptions.topoMode)
        {
        case DataStructureOptions::TopologyMode::KnnGraph:
            f(asset.knnGraph);
            break;
        case DataStructureOptions::TopologyMode::NeighborGraph:
            f(asset.neiGraph);
            break;
        case DataStructureOptions::TopologyMode::MeshGraph:
            f(asset.meshGraph);
            break;
        case DataStructureOptions::TopologyMode::None:
        case DataStructureOptions::TopologyModeCount:
        default:
            break;
        }
    }

    //! \brief Dispatch a lambda on a range neighbors query over either the KnnGraph or the KdTree
    template <typename Functor>
    void doOnKNeighbors(const int i, const Functor& f) {
        if (computeOpts.kNN > Types::KdTree::Traits::MAX_KNN_SIZE)
        {
            std::cout<< "[warning] Knn query is too large, reducing" << std::endl;
            computeOpts.kNN = Types::KdTree::Traits::MAX_KNN_SIZE;
        }
        f(asset.tree.kNearestNeighbors(i, computeOpts.kNN));
    }

    //! \brief Dispatch a lambda on either a range or a knn query depending on the UI.
    template <typename Functor>
    void doOnNeighbors(const int i, const Functor& f) {
        computeOpts.useRangeNei ? doOnRangeNeighbors(i,f) : doOnKNeighbors(i, f);
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
