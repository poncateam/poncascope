#include "datastructures.hpp"

#include "context.hpp"
#include "imgui.h"
#include "polyscope/curve_network.h"

#include "polyscope/point_cloud.h"


/// Generate a curve network for the selected point
void generateLocalTopologyDisplay(Context& context, const std::string &name)
{
    std::vector<Context::Types::VectorType> vertices;
    std::vector<std::array<size_t, 2>> edges;
    std::vector<double> distances;

    const Context::Types::VectorType q = context.asset.tree.points()[context.computeOpts.iVertexSource].pos();

    vertices.push_back(q);
    distances.push_back(0);

    context.doOnNeighbors(context.computeOpts.iVertexSource, [&vertices,&context,&edges,&distances,&q](auto&& neighborhood){
        for (int j : neighborhood){
            edges.push_back({0,vertices.size()});
            vertices.emplace_back(context.asset.tree.points()[j].pos());
            distances.push_back((q-vertices.back()).norm());
        }
    });
    polyscope::CurveNetwork* c = polyscope::registerCurveNetwork(name, vertices, edges);
    c->addNodeScalarQuantity("Distances",distances);
}

/// Generate a curve network of the entire neighbor graph
void generateFullTopologyDisplay(Context& context)
{
    std::vector<std::array<int, 2>> edges;
    std::string name = context.dataStructureOptions.topoMode == Context::DataStructureOptions::KnnGraph
        ? "Knn Graph"
        : "Range Neighbor Graph";

    for (int i = 0; i != context.asset.tree.pointCount(); ++i)
    {
        context.doOnGraph([&edges,&i](auto& graph)
        {
            for (int j : graph->oneConnectedNeighbors(i))
                edges.push_back({i,j});
        });
    }
    polyscope::CurveNetwork* c = polyscope::registerCurveNetwork(name, context.asset.cloudV, edges);

}

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

    context.asset.addScalarQuantity(  "range neighborhood pts " + std::to_string(context.computeOpts.iVertexSource), closest);

    generateLocalTopologyDisplay(context, "range links pts " + std::to_string(context.computeOpts.iVertexSource));
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

    context.asset.addScalarQuantity(  "knn neighborhood pts " + std::to_string(context.computeOpts.iVertexSource), closest);

    generateLocalTopologyDisplay(context, "knn links pts " + std::to_string(context.computeOpts.iVertexSource));
}

/// Recompute K-Neighbor graph
void recomputeTopology(Context& context) {
    measureTime("[Ponca] Build Graph", [&context]() {
        // delete existing datastructures
        delete context.asset.knnGraph;
        delete context.asset.neiGraph;

        context.asset.knnGraph = nullptr;
        context.asset.neiGraph = nullptr;

        switch (context.dataStructureOptions.topoMode)
        {
        case Context::DataStructureOptions::KnnGraph:
            context.asset.knnGraph = new Context::Types::KnnGraph(context.asset.tree, context.dataStructureOptions.kNNGraphK);
            break;
        case Context::DataStructureOptions::NeighborGraph:
            context.asset.neiGraph = new Context::Types::NeighborGraph(context.asset.tree, context.dataStructureOptions.neighborGraphRange);
            break;
        case Context::DataStructureOptions::None:
        case Context::DataStructureOptions::TopologyModeCount:
        default:
            ;
        }
    });
    if (context.dataStructureOptions.displayTopology) generateFullTopologyDisplay(context);
}

void callback_datastructures(Context& context)
{
    ImGui::SeparatorText("Topology");
    int currentTopologyMode = context.dataStructureOptions.topoMode;
    const char* topologyMode[] = { "None", "Knn Graph", "Neighbor Graph" };
    if (ImGui::Combo("Restrict queries to topology", &currentTopologyMode, topologyMode, Context::DataStructureOptions::TopologyModeCount))
    {
        if (currentTopologyMode != context.dataStructureOptions.topoMode)
        {

            context.dataStructureOptions.topoMode = Context::DataStructureOptions::TopologyMode(currentTopologyMode);
            if (currentTopologyMode != Context::DataStructureOptions::TopologyMode::None) // recompute when activated
                recomputeTopology(context);
        }
    }

    bool changed = false;
    if (currentTopologyMode == Context::DataStructureOptions::TopologyMode::KnnGraph)
        changed = ImGui::InputInt("k-neighborhood size", &context.dataStructureOptions.kNNGraphK);
    else if (currentTopologyMode == Context::DataStructureOptions::TopologyMode::NeighborGraph)
        changed = ImGui::InputFloat("k-neighborhood size", &context.dataStructureOptions.neighborGraphRange);

    if (currentTopologyMode != Context::DataStructureOptions::TopologyMode::None)
        if (ImGui::Checkbox("Display topology", &context.dataStructureOptions.displayTopology))
        {
            generateFullTopologyDisplay(context);
        }

    if (changed) recomputeTopology(context);

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
