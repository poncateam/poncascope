#include "datastructures.hpp"
#include "imgui.h"

#include "polyscope/point_cloud.h"


/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood(Context& context) {
    int nvert = int(context.asset.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    Context::Types::SmoothWeightFunc w(context.asset.tree.points()[context.computeOpts.iVertexSource], context.computeOpts.NSize );

    closest(context.computeOpts.iVertexSource) = 2;
    context.doOnRangeNeighbors(context.computeOpts.iVertexSource, [w, &closest, &context](auto &&neighborhood){
        for (int j : neighborhood){
            const auto &q = context.asset.tree.points()[j];
            closest(j) = w( q ).first;
        }
    });

    context.asset.addScalarQuantity(  "range neighborhood", closest);
}

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn(Context& context) {
    int nvert = int(context.asset.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(context.computeOpts.iVertexSource) = 2;
    context.doOnKNeighbors(context.computeOpts.iVertexSource, [&closest](auto&& neighborhood){
        for (int j : neighborhood){
            closest(j) = 1;
        }
    });

    context.asset.addScalarQuantity(  "knn neighborhood", closest);
}

/// Recompute K-Neighbor graph
void recomputeKnnGraph(Context& context) {
    measureTime("[Ponca] Build KnnGraph", [&context]() {
        delete context.asset.knnGraph;
        context.asset.knnGraph = new Context::Types::KnnGraph(context.asset.tree, context.computeOpts.kNNGraphK);
    });
}

void callback_datastructures(Context& context)
{
    ImGui::SeparatorText("Acceleration Structure");
    bool knnGraphUIChanged = ImGui::Checkbox("Use KnnGraph", &context.computeOpts.useKnnGraph);
    if (context.computeOpts.useKnnGraph)
    {
        ImGui::SameLine();
        if (ImGui::InputInt("Graph k", &context.computeOpts.kNNGraphK) || knnGraphUIChanged) // recompute when activated or changed
            recomputeKnnGraph(context);
    }

    ImGui::SeparatorText("Neighborhood queries");
    ImGui::Checkbox("Use Range Queries", &context.computeOpts.useRangeNei);
    ImGui::SameLine();
    if (context.computeOpts.useRangeNei)
        ImGui::InputFloat("neighborhood range size", &context.computeOpts.NSize);
    else
        ImGui::InputInt("k-neighborhood size", &context.computeOpts.kNN);

    ImGui::SeparatorText("Neighborhood display");
    ImGui::InputInt("source vertex", &context.computeOpts.iVertexSource);
    ImGui::SameLine();
    if (context.computeOpts.useRangeNei) {
        if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood(context);
    }
    else
        if (ImGui::Button("show knn")) colorizeKnn(context);
}
