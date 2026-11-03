#include "marchingCubes.hpp"

#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/sparse_volume_grid.h"
#include <igl/marching_cubes.h>
#include <imgui.h>
#include <algorithm>
#include <limits>
#include <omp.h>

template<typename Fit>
void RunMarchingCubes(Context& context, const char* name)
{
    using Scalar     = typename Fit::Scalar;
    using Point      = typename Fit::DataPoint;
    using VectorType = typename Fit::VectorType;

    // Note: There are a few hardcoded numbers (the number of vertices per cube)
    static_assert(Point::Dim == 3 && "MarchingCube only available in 3D !");

    const size_t N = context.marchingCubesOptions.N;
    const size_t M = N - 1;
    
    const Scalar t = context.computeOpts.NSize;
    const VectorType min = context.asset.cloudV.colwise().minCoeff().transpose();
    const VectorType max = context.asset.cloudV.colwise().maxCoeff().transpose();

    // Pad the bounding box so the surface is not clipped at the boundary
    const VectorType lower = min - VectorType::Constant(t);
    const VectorType upper = max + VectorType::Constant(t);
    const VectorType step  = (upper - lower) / Scalar(N - 1);

    // Data formats expected by igl
    Eigen::Matrix<Scalar , Eigen::Dynamic, 3> Grid(N * N * N, 3);
    Eigen::Matrix<Scalar , Eigen::Dynamic, 1> Data(N * N * N);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> Stables(N * N * N);
    
    // Vector for dynamically sized data
    std::vector<size_t> offsets;

    // Data format for polyscope
    std::vector<glm::ivec3> cells;
    std::vector<Scalar> values;

    #pragma omp parallel
    {        
        #pragma omp single
        offsets.assign(omp_get_num_threads() + 1, 0);

        const int tid = omp_get_thread_num();
        size_t count = 0;

        // Compute potentials and stable-ness
        #pragma omp for schedule(static)
        for (size_t i = 0; i < N * N * N; ++i)
        {
            const size_t x = i % N;
            const size_t y = (i / N) % N;
            const size_t z = i / (N * N);

            const VectorType p = lower + step.cwiseProduct(VectorType(x, y, z));
            Grid.row(i) = p.transpose();

            Fit fit;
            fit.init();
            fit.setNeighborFilter({p, t});
            fit.computeWithIds(context.asset.tree.rangeNeighbors(p, t), context.asset.tree.points());

            Stables(i) = fit.isStable();

            Data(i) = Stables(i) ? fit.potential(p) : 0;
            Data(i) = fit.isSigned() ? Data(i) : std::abs(Data(i));
            
            count += Stables(i);
        }
        
        offsets[tid + 1] = count;

        #pragma omp barrier
        #pragma omp single
        {
            for (size_t i = 1; i < offsets.size(); ++i)
                offsets[i] += offsets[i - 1];
                
            cells.resize(offsets.back());
            values.resize(offsets.back());
        }

        size_t pos = offsets[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < N * N * N; ++i)
        {
            if (!Stables(i))
                continue;

            cells[pos]  = glm::ivec3(i % N, (i / N) % N, i / (N * N));
            values[pos] = Data(i);

            pos++;
        }
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> V;
    Eigen::MatrixXi F;
    igl::marching_cubes(Data, Grid, N, N, N, Scalar(0), V, F);

    polyscope::registerSurfaceMesh(name, V, F);

    // Shift the lower point so that evaluation are at the center of each cube
    const glm::vec3 pstep  = glm::vec3(step.x(), step.y(), step.z());
    const glm::vec3 plower = glm::vec3(lower.x(), lower.y(), lower.z()) - 0.5f * pstep;

    auto* grid = polyscope::registerSparseVolumeGrid(std::string(name) + "_grid", plower, pstep, cells);
    grid->addCellScalarQuantity("potential", values, polyscope::DataType::SYMMETRIC)->setEnabled(true);
}

void callback_marchingcubes(Context& context)
{
    using namespace Ponca;

    static auto filter = Context::Types::Factory::Filter<ProjectionOperatorProvider>();
    static auto names = filter.GetNames();
    static int currentMethod = 0;

    ImGui::SeparatorText("Marching cube");
    ImGui::Combo("Fit function", &currentMethod, names.data(), names.size());

    if (ImGui::InputInt("Grid size", &context.marchingCubesOptions.N))
        context.marchingCubesOptions.N = std::clamp(context.marchingCubesOptions.N, 3, 1024);

    if (ImGui::Button("Run marching cubes"))
    {
        filter.foreach([&](const auto& x) {
            using FitType = std::remove_cv_t<decltype(x.object)>;
            if (x.idx == currentMethod)
                RunMarchingCubes<FitType>(context, x.name);
        });
    }
}