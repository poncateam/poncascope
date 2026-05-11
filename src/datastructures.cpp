#include "datastructures.hpp"
#include "imgui.h"

#include "polyscope/point_cloud.h"


/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood(Context& context) {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    Context::SmoothWeightFunc w(context.tree.points()[context.iVertexSource], context.NSize );

    closest(context.iVertexSource) = 2;
    context.doOnRangeNeighbors(context.iVertexSource, [w, &closest, &context](auto &&neighborhood){
        for (int j : neighborhood){
            const auto &q = context.tree.points()[j];
            closest(j) = w( q ).first;
        }
    });

    context.cloud->addScalarQuantity(  "range neighborhood", closest);
}

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn(Context& context) {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(context.iVertexSource) = 2;
    context.doOnKNeighbors(context.iVertexSource, [&closest](auto&& neighborhood){
        for (int j : neighborhood){
            closest(j) = 1;
        }
    });

    context.cloud->addScalarQuantity(  "knn neighborhood", closest);
}

/// Recompute K-Neighbor graph
void recomputeKnnGraph(Context& context) {
    measureTime("[Ponca] Build KnnGraph", [&context]() {
        delete context.knnGraph;
        context.knnGraph = new Context::KnnGraph(context.tree, context.kNNGraphK);
    });
}

void callback_datastructures(Context& context)
{
    ImGui::SeparatorText("Acceleration Structure");
    bool knnGraphUIChanged = ImGui::Checkbox("Use KnnGraph", &context.useKnnGraph);
    if (context.useKnnGraph)
    {
        ImGui::SameLine();
        if (ImGui::InputInt("Graph k", &context.kNNGraphK) || knnGraphUIChanged) // recompute when activated or changed
            recomputeKnnGraph(context);
    }

    ImGui::SeparatorText("Neighborhood queries");
    ImGui::Checkbox("Use Range Queries", &context.useRangeNei);
    ImGui::SameLine();
    if (context.useRangeNei)
        ImGui::InputFloat("neighborhood range size", &context.NSize);
    else
        ImGui::InputInt("k-neighborhood size", &context.kNN);

    ImGui::SeparatorText("Neighborhood display");
    ImGui::InputInt("source vertex", &context.iVertexSource);
    ImGui::SameLine();
    if (context.useRangeNei) {
        if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood(context);
    }
    else
        if (ImGui::Button("show knn")) colorizeKnn(context);
}
