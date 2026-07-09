#include "projection.hpp"

#include "estimators.hpp"

#include "polyscope/point_cloud.h"

#include <imgui.h>

using namespace Ponca;
using Scalar     = Context::Types::Scalar;
using VectorType = Context::Types::VectorType;

static int currentProj = 0;

/// Project the entire geometry on the given surface
template <typename FitT>
void projectPointCloud(Context& context, const std::string& name)
{
    Eigen::MatrixXd projected = Eigen::MatrixXd(context.asset.cloudV.rows(), context.asset.cloudV.cols());
    projected.setZero();

    measureTime( "[Ponca] Project point cloud " + name,
                 [&projected, &context]() {
        processPointCloudMLS<FitT>(context.computeOpts.NSize,
                                [&projected, &context]
                                (const int i, const FitT& fit){
            projected.row(i) = fit.project(context.asset.cloudV.row(i));
        }, context);
    });

    polyscope::registerPointCloud(name, projected);

}

/// Project the neighborhood on the local primitive fitted at the evaluation point
template <typename FitT>
void projectNeighborhood(Context& context, const std::string& name)
{
    std::vector<Context::Types::VectorType> vertices;

    int i = context.computeOpts.iVertexSource;
    Scalar t = context.computeOpts.NSize;
    const Context::Types::VectorType q = context.asset.tree.points()[i].pos();

    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.computeOpts.mlsIter, Scalar(context.computeOpts.mlsEpsilon));

    FitT fit;
    fit.setNeighborFilter({q, t});

    context.doOnNeighbors(i, [&](const auto& rangeNeighbors){
        mls_evaluation_scheme.computeWithIds(fit, rangeNeighbors, context.asset.tree.points());
    });

    if (fit.isStable()) {
        context.doOnNeighbors(i, [&vertices, &fit, &context](auto&& neighborhood){
            for (const auto& j : neighborhood)
                vertices.push_back(fit.project(context.asset.tree.points()[j].pos()));
        });
    }

    polyscope::registerPointCloud(name + "_local", vertices);
}

void callback_projection(Context& context)
{
    static auto filter = Context::Types::Factory::Filter<NotDerivativesProvider, ProjectionOperatorProvider>();
    static auto names = filter.GetNames();

    ImGui::SeparatorText("Projection type");
    ImGui::Combo("Fit function", &currentProj, names.data(), names.size());

    ImGui::SeparatorText("Projection type");
    if (ImGui::Button("Project neighborhood on local primitive"))
    {
        filter.foreach([&](const auto& x) {
            using FitType = std::remove_cv_t<decltype(x.object)>;
            if (x.idx == currentProj)
                projectNeighborhood<FitType>(context, x.name);
        });
    }
    if (ImGui::Button("Project All"))
    {
        filter.foreach([&](const auto& x) {
            using FitType = std::remove_cv_t<decltype(x.object)>;
            if (x.idx == currentProj)
                projectPointCloud<FitType>(context, x.name);
        });
    }
}
