#pragma once

#include "definitions.h"
#include "MyPointCloud.h"

#include <iostream>
#include <utility>
#include <chrono>

class PointProcessing {

    // Variables
    public:

        KdTree tree;                   /// < kdtree for nearest neighbors search
        KnnGraph *knnGraph = nullptr;  /// < k-neighbor graph

        // Options for algorithms
        bool  useKnnGraph    = false; /// < use k-neighbor graph instead of kdtree
        int   iVertexSource  = 7;     /// < id of the selected point
        int   kNN            = 10;    /// < neighborhood size (knn)
        int   mlsIter        = 3;     /// < number of moving least squares iterations
        float NSize          = 0.25;  /// < neighborhood size (euclidean)

        // Current Point Cloud Data

    public :

        PointProcessing() {
            // Default values
            useKnnGraph    = false;
            iVertexSource  = 7;
            kNN            = 10;
            mlsIter        = 3;
            NSize          = 0.25;
        }

        // Constructor
        PointProcessing(MyPointCloud &cloud) {

            // Default values
            useKnnGraph    = false;
            iVertexSource  = 7;
            kNN            = 10;
            mlsIter        = 3;
            NSize          = 0.25;
        
            // build kdtree and knn graph
            update(cloud);
        };

        // update the KdTree
        void update(MyPointCloud &cloud) {
            measureTime( "[Ponca] Build KdTree",
                 [this, &cloud]() {
                    buildKdTree(cloud.getVertices(), cloud.getNormals(), tree);
                });
            recomputeKnnGraph();
        }

        /// @brief Measure time of a function
        /// @tparam Functor type of the function
        /// @param actionName name of the action 
        /// @param F function to measure
        template <typename Functor>
        void measureTime( const std::string &actionName, Functor F );
        
        /// Compute the kNN graph
        void recomputeKnnGraph();

        /// @brief Compute differential quantities
        /// @tparam FitT Fit Type, \see definitions.h
        /// @param name Name of the method, to be displayed in the console
        /// @param cloud Point Cloud to process
        template<typename FitT>
        void computeDiffQuantities(const std::string &name, MyPointCloud &cloud);

        /// Dry run: loop over all vertices + run MLS loops without computation
        /// This function is useful to monitor the KdTree performances
        /// And to compute the mean number of neighbors
        void mlsDryRun();

        /// Colorize point cloud using kNN
        const Eigen::VectorXd colorizeKnn();

        /// Colorize point cloud using euclidean distance
        const Eigen::VectorXd colorizeEuclideanNeighborhood();

        const Eigen::VectorXd getVertexSourcePosition(){ return tree.point_data()[iVertexSource].pos(); }

        Scalar getMeanNeighbors() { return m_meanNeighbors; }

private :

        Scalar m_meanNeighbors = 0;

        /// @brief Loop over all points to compute differential quantities, the method differ depending on the used spatial data structure
        /// @tparam Functor type of the function
        /// @param idx index of the point to process
        /// @param f function to apply on the neighbors
        template <typename Functor>
        void processRangeNeighbors(const int &idx, const Functor f);

        /// @brief Process one point, compute fitting, and use functor to process fitting output
        /// @tparam FitT Fit Type, \see definitions.h
        /// @tparam Functor type of the function
        /// @param idx index of the point to process
        /// @param w weight function
        /// @param f function to apply on the fitting
        template<typename FitT, typename Functor>
        void processOnePoint(const int &idx, const typename FitT::WeightFunction& w, Functor f);

        /// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
        /// \note Functor is called only if fit is stable
        /// @tparam FitT Fit Type, \see definitions.h
        /// @tparam Functor type of the function
        /// @param w weight function
        /// @param f function to apply on the fitting
        template<typename FitT, typename Functor>
        void processPointCloud(const typename FitT::WeightFunction& w, Functor f);
        
}; // class PointProcessing

#include "PointProcessing.hpp"
