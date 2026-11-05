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

    // In the following: cube refers to data for the marching cube and cells for polyscope grid

    // Data formats expected by igl
    Eigen::Matrix<Scalar , Eigen::Dynamic, 3> cubeGrid(N * N * N, 3);
    Eigen::Matrix<Scalar , Eigen::Dynamic, 1> cubeData(N * N * N);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> cubeStables(N * N * N);
    Eigen::Matrix<size_t , Eigen::Dynamic, 8> cubeIndices;

    // Shift in indices for for cube grids, in igl order 
    const size_t iglCubeIndicesOffsets[8] = {
        0, 1, 1 + N, N,
        N * N, 1 + N * N, 1 + N + N * N, N + N * N 
    };

    // Data format for polyscope
    std::vector<glm::ivec3> cells;
    std::vector<Scalar> cellValues;

    // Offset within threads
    std::vector<size_t> cellOffsets;
    std::vector<size_t> cubeOffsets;


    #pragma omp parallel
    {        
        #pragma omp single
        {
            cellOffsets.assign(omp_get_num_threads() + 1, 0);
            cubeOffsets.assign(omp_get_num_threads() + 1, 0);
        }
        
        const int tid = omp_get_thread_num();
        size_t cellCount = 0;
        size_t cubeCount = 0;

        // Compute potentials and stable-ness
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N * N * N; ++i)
        {
            const size_t x = i % N;
            const size_t y = (i / N) % N;
            const size_t z = i / (N * N);

            const VectorType p = lower + step.cwiseProduct(VectorType(x, y, z));
            cubeGrid.row(i) = p.transpose();

            Fit fit;
            fit.init();
            fit.setNeighborFilter({p, t});
            fit.computeWithIds(context.asset.tree.rangeNeighbors(p, t), context.asset.tree.points());

            cubeStables(i) = fit.isStable();

            cubeData(i) = cubeStables(i) ? fit.potential(p) : (fit.isSigned() ? std::numeric_limits<Scalar>::max() : 0);
            
            if (context.marchingCubesOptions.keepSign)
                cubeData(i) = std::abs(cubeData(i));
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < N * N * N; ++i)
        {
            const size_t x = i % N;
            const size_t y = (i / N) % N;
            const size_t z = i / (N * N);

            cellCount += cubeStables(i);

            bool ok = cubeStables(i) && x < M && y < M && z < M;
            for (int j = 1; j < 8 && ok; ++j)
                ok = cubeStables(i + iglCubeIndicesOffsets[j]);

            if (ok)
                cubeCount ++;
        }

        cellOffsets[tid + 1] = cellCount;
        cubeOffsets[tid + 1] = cubeCount;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i < cellOffsets.size(); ++i)
            {
                cellOffsets[i] += cellOffsets[i - 1];
                cubeOffsets[i] += cubeOffsets[i - 1];
            }

            cells.resize(cellOffsets.back());
            cellValues.resize(cellOffsets.back());
            cubeIndices.resize(cubeOffsets.back(), 8);
        }

        size_t cellPos = cellOffsets[tid];
        size_t cubePos = cubeOffsets[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < N * N * N; ++i)
        {
            if (!cubeStables(i))
                continue;

            const size_t x = i % N;
            const size_t y = (i / N) % N;
            const size_t z = i / (N * N);

            cells[cellPos] = glm::ivec3(x, y, z);
            cellValues[cellPos] = cubeData(i);
            cellPos++;

            // We would cache this results, but this is quite a cheap loop
            // and at some point we should stop allocating memory :) 
            bool ok = x < M && y < M && z < M;
            for (int j = 1; j < 8 && ok; ++j)
                ok = cubeStables(i + iglCubeIndicesOffsets[j]);

            if (!ok)
                continue;

            for (int j = 0; j < 8; ++j)
                cubeIndices(cubePos, j) = i + iglCubeIndicesOffsets[j];

            cubePos++;
        }
    }

    // Only emit geometry if the method is signed. The marching cube requires
    // notion of interior which unsigned method can not provide
    if (context.marchingCubesOptions.displayMesh)
    {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 3> V;
        Eigen::MatrixXi F;
        igl::marching_cubes(cubeData, cubeGrid, cubeIndices, Scalar(0), V, F);

        polyscope::registerSurfaceMesh(name, V, F);
    }

    if (context.marchingCubesOptions.displayCube)
    {
        // Shift the lower point so that evaluation are at the center of each cube
        const glm::vec3 pstep  = glm::vec3(step.x(), step.y(), step.z());
        const glm::vec3 plower = glm::vec3(lower.x(), lower.y(), lower.z()) - 0.5f * pstep;

        auto* grid = polyscope::registerSparseVolumeGrid(std::string(name) + "_grid", plower, pstep, cells);
        grid->addCellScalarQuantity("potential", cellValues, polyscope::DataType::SYMMETRIC)->setEnabled(true);
    }
    
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

    ImGui::Checkbox("Keep sign", &context.marchingCubesOptions.keepSign);
    ImGui::Checkbox("Display mesh", &context.marchingCubesOptions.displayMesh);    
    ImGui::Checkbox("Display cubes", &context.marchingCubesOptions.displayCube);

    if (ImGui::Button("Run marching cubes"))
    {
        filter.foreach([&](const auto& x) {
            using FitType = std::remove_cv_t<decltype(x.object)>;
            if (x.idx == currentMethod)
                RunMarchingCubes<FitType>(context, x.name);
        });
    }
}