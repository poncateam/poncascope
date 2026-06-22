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
    ImGui::SeparatorText("Projection type");
    const char* items[] = { "PSS", "APSS"};
    ImGui::Combo("Fit function", &currentProj, items, IM_ARRAYSIZE(items));

    ImGui::SeparatorText("Projection type");
    if (ImGui::Button("Project neighborhood on local primitive"))
    {
        switch (currentProj)
        {
            case 0: projectNeighborhood<Context::Types::FitPlane>(context, items[0]) ; break;
            case 1: projectNeighborhood<Context::Types::FitAPSS>(context, items[1]) ; break;
            default: break;
        };
    }
    if (ImGui::Button("Project All"))
    {
        switch (currentProj)
        {
        case 0: projectPointCloud<Context::Types::FitPlane>(context, items[0]) ; break;
        case 1: projectPointCloud<Context::Types::FitAPSS>(context, items[1]) ; break;
        default: break;
        };
    }


}

