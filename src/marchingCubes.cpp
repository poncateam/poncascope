#include "marchingCubes.hpp"

#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include <igl/marching_cubes.h>
#include <imgui.h>
#include <limits>

template<typename Fit>
void RunMarchingCubes(Context& context, const char* name)
{
    using Scalar     = typename Fit::Scalar;
    using VectorType = typename Fit::VectorType;

    // Pad the bounding box so the surface is not clipped at the boundary
    const Scalar t = context.computeOpts.NSize;
    const VectorType min = context.asset.cloudV.colwise().minCoeff().transpose();
    const VectorType max = context.asset.cloudV.colwise().maxCoeff().transpose();
    const VectorType lower = min - VectorType::Constant(t);
    const VectorType upper = max + VectorType::Constant(t);

    const size_t N = context.marchingCubesOptions.N;
    const VectorType step = (upper - lower) / Scalar(N - 1);

    // Data formats expected by igl
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> Grid(N * N * N, 3);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Data(N * N * N);

    #pragma omp parallel for
    for (size_t i = 0; i < N * N * N; ++i) {
        const size_t x = i % N;
        const size_t y = (i / N) % N;
        const size_t z = i / (N * N);
        
        const VectorType p = lower + step.cwiseProduct(VectorType(x, y, z));
        Grid.row(i) = p.transpose();

        Fit fit;
        fit.init();
        fit.setNeighborFilter({p, t});
        fit.computeWithIds(context.asset.tree.rangeNeighbors(p, t), context.asset.tree.points());

        Data(i) = fit.isStable() ? fit.potential(p) : std::numeric_limits<Scalar>::max();
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> V;
    Eigen::MatrixXi F;
    igl::marching_cubes(Data, Grid, N, N, N, Scalar(0), V, F);

    polyscope::registerSurfaceMesh(name, V, F);
}

void callback_marchingcubes(Context& context)
{
    using namespace Ponca;

    static auto filter = Context::Types::Factory::Filter<ProjectionOperatorProvider>();
    static auto names = filter.GetNames();
    static int currentMethod = 0;

    ImGui::SeparatorText("Marching cube");
    ImGui::Combo("Fit function", &currentMethod, names.data(), names.size());

    ImGui::InputInt("Grid size", &context.marchingCubesOptions.N, 3, 1024);

    if (ImGui::Button("Run marching cubes"))
    {
        filter.foreach([&](const auto& x) {
            using FitType = std::remove_cv_t<decltype(x.object)>;
            if (x.idx == currentMethod)
                RunMarchingCubes<FitType>(context, x.name);
        });
    }
}