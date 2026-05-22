#pragma once


#include "Ponca/src/SpatialPartitioning/NeighborGraph/abstractNeighborGraph.h"
#include "igl/vertex_triangle_adjacency.h"

/// \brief Neighbor Graph constructed from a mesh
template <typename _Traits>
class MeshNeighborGraphBase : public Ponca::StaticNeighborGraphBase<_Traits>
{
public:
    WRITE_NEIGHBOR_GRAPH_ALIASES

private:
    using Base    = Ponca::StaticNeighborGraphBase<Traits>;
    using Buffers = typename Base::Buffers;

public:
    /// \brief Build a neighbor graph by connecting all points using Euclidean range queries: two points
    /// \f[\mathbf{p,q}\f] are connected if \f[\mathbf{p}-\mathbf{q}<\text{range}\f]
    ///
    /// \note Each point might have a different number of neighbors
    /// \note Empty neighborhood are not checked
    ///
    /// \param _kdtree Reference to the KdTree
    /// \param range Distance threshold used to connect points
    ///
    /// \warning Stores a const reference to kdtree.point_data()
    /// \warning KdTreeTraits compatibility is checked with static assertion
    template <typename KdTreeTraits>
    PONCA_MULTIARCH_HOST inline MeshNeighborGraphBase(const Ponca::KdTreeBase<KdTreeTraits>& _kdtree,
        const Eigen::MatrixXi& faces)
        : Base(_kdtree.points())
    {
        Base::m_bufs.points_size = _kdtree.pointCount();
#define CHECK_TRAITS_TYPENAME_COMPAT(A, B)                                                            \
static_assert(std::is_same_v<A, B> || std::is_convertible_v<A, B> || std::is_convertible_v<B, A>, \
              "KdTreeTraits::DataPoint is not equal to Traits::DataPoint");

        static_assert(std::is_same_v<typename Traits::DataPoint, typename KdTreeTraits::DataPoint>,
                      "KdTreeTraits::DataPoint is not equal to Traits::DataPoint");

        CHECK_TRAITS_TYPENAME_COMPAT(typename Traits::PointContainer, typename KdTreeTraits::PointContainer)
        CHECK_TRAITS_TYPENAME_COMPAT(typename Traits::IndexContainer, typename KdTreeTraits::IndexContainer)

#undef CHECK_TRAITS_TYPENAME_COMPAT

        // We need to account for the entire point set, irrespectively of the sampling. This is because the kdtree
        // (kNearestNeighbors) return ids of the entire point set, not it sub-sampled list of ids.
        // \fixme Update API to properly handle kdtree subsampling
        const int cloudSize = _kdtree.pointCount();
        {
            const int samplesSize = _kdtree.sampleCount();
            PONCA_ASSERT(cloudSize == samplesSize);
        }

        // use double-index vectors to avoid to handle indices shift when inserting elements
        // (to be converted at the end of the process as Base::m_bufs.ranges
        std::vector<std::vector<int>> explicitRanges;
        explicitRanges.resize(cloudSize);

        int nbVertInFace = faces.cols();

        // loop over the faces to fill explicitRanges ranges
// #pragma omp parallel for collapse(3)
        for (int fi = 0; fi != faces.rows(); ++fi)
        {
            // here we could use shorter loops and insert elements in both current and candidate buffers,
            // Instead, we use this simple implementation that is slightly slower.
            for(int iCurrent = 0; iCurrent != nbVertInFace; ++iCurrent)
                for(int iCandidate = 0; iCandidate < nbVertInFace; ++iCandidate)
                {
                    int current   = faces.row(fi)(iCurrent);
                    int candidate = faces.row(fi)(iCandidate);
                    auto& r = explicitRanges[current];

// #pragma omp critical
                    {
                        if (std::find(r.begin(), r.end(), candidate) == r.end())
                            r.push_back(candidate);
                    }
                }
        }

        // flatten explicitRanges to Base::m_bufs
        Base::m_bufs.ranges.resize(cloudSize + 1); // we need one more index (see StaticNeighborGraphBase::endId)
        Base::m_bufs.ranges[0] = 0;                // first element starts at 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            for (const auto& n : explicitRanges[i])
            {
                Base::m_bufs.indices.push_back(n);
            }
            Base::m_bufs.ranges[i + 1] = Base::m_bufs.indices.size();
        }
        Base::m_bufs.indices_size = Base::m_bufs.indices.size();
    }
};

/*!
 * \brief Public interface for the MeshNeighborGraphBase datastructure.
 *
 * Provides default implementation of the MeshNeighborGraphBase
 */
template <typename DataPoint>
using MeshNeighborGraph = MeshNeighborGraphBase<Ponca::NeighborGraphDefaultTraits<DataPoint>>;


