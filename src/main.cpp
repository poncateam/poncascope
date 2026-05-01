#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"

#include "polyscopeSlicer.hpp"

#include <iostream>
#include <utility>

// This file defines all the main types + data shared across the application components
#include "./context.hpp"
#include "./io.hpp"
#include "./estimators.hpp"

Context context;

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;

/// Show in polyscope the euclidean neighborhood of the selected point (iVertexSource), with smooth weighting function
void colorizeEuclideanNeighborhood() {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    Context::SmoothWeightFunc w(context.tree.points()[context.iVertexSource], context.NSize );

    closest(context.iVertexSource) = 2;
    context.doOnRangeNeighbors(context.iVertexSource, [w, &closest](auto &&neighborhood){
        for (int j : neighborhood){
            const auto &q = context.tree.points()[j];
            closest(j) = w( q ).first;
        }
    });

    context.cloud->addScalarQuantity(  "range neighborhood", closest);
}

/// Show in polyscope the knn neighborhood of the selected point (iVertexSource)
void colorizeKnn() {
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
void recomputeKnnGraph() {
    measureTime("[Ponca] Build KnnGraph", []() {
        delete context.knnGraph;
        context.knnGraph = new Context::KnnGraph(context.tree, context.kNNGraphK);
    });
}

///Evaluate scalar field for generic FitType.
///// \tparam FitT Defines the type of estimator used for computation
template<typename FitT, bool isSigned = true>
Scalar evalScalarField_impl(const VectorType& input_pos)
{
    FitT fit;
    fit.setNeighborFilter({input_pos, context.NSize}); // weighting function using current pos (not input pos)
    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.mlsIter, Scalar(context.mlsEpsilon));
    auto res = mls_evaluation_scheme.computeWithIds(fit, context.tree.rangeNeighbors(input_pos, context.NSize),
        context.tree.points());

    if(!fit.isStable()) {
        // not enough neighbors (if far from the point cloud)
        return Scalar(0); //std::numeric_limits<Scalar>::max();
    }

    const Scalar current_value = isSigned ? fit.potential(input_pos) : std::abs(fit.potential(input_pos));
    // current_gradient = fit.primitiveGradient(input_pos);
    return current_value;
}

/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    callback_io(context);

    ImGui::Text("Acceleration Structure");
    bool knnGraphUIChanged = ImGui::Checkbox("Use KnnGraph", &context.useKnnGraph);
    if (context.useKnnGraph)
    {
        ImGui::SameLine();
        if (ImGui::InputInt("Graph k", &context.kNNGraphK) || knnGraphUIChanged) // recompute when activated or changed
            recomputeKnnGraph();
    }

    ImGui::Separator();
    ImGui::Text("Neighborhood queries");
    ImGui::Checkbox("Use Range Queries", &context.useRangeNei);
    ImGui::SameLine();
    if (context.useRangeNei)
        ImGui::InputFloat("neighborhood range size", &context.NSize);
    else
        ImGui::InputInt("k-neighborhood size", &context.kNN);

    ImGui::Separator();
    ImGui::InputInt("source vertex", &context.iVertexSource);
    ImGui::SameLine();
    if (context.useRangeNei) {
        if (ImGui::Button("show euclidean nei")) colorizeEuclideanNeighborhood();
    }
    else
        if (ImGui::Button("show knn")) colorizeKnn();

    ImGui::Separator();
    ImGui::InputInt("Nb MLS Iterations", &context.mlsIter);
    ImGui::InputFloat("MLS Epsilon", &context.mlsEpsilon);
    ImGui::Separator();

    callback_estimators(context);

    ImGui::Separator();

    ImGui::Text("Implicit function slicer");
    ImGui::SliderFloat("Slice", &context.slice, 0, 1.0); ImGui::SameLine();
    ImGui::Checkbox("HD", &context.isHDSlicer);
    ImGui::RadioButton("X axis", &context.axis, 0); ImGui::SameLine();
    ImGui::RadioButton("Y axis", &context.axis, 1); ImGui::SameLine();
    ImGui::RadioButton("Z axis", &context.axis, 2);
    const char* items[] = { "ASO", "APSS", "PSS"};
    static int item_current = 0;
    ImGui::Combo("Fit function", &item_current, items, IM_ARRAYSIZE(items));
    if (ImGui::Button("Update"))
    {
      switch(item_current)
      {
        case 0: registerRegularSlicer("slicer", evalScalarField_impl<Context::FitASO, true>   , context.lower, context.upper, context.isHDSlicer?1024:256, context.axis, context.slice); break;
        case 1: registerRegularSlicer("slicer", evalScalarField_impl<Context::FitAPSS, true>  , context.lower, context.upper, context.isHDSlicer?1024:256, context.axis, context.slice); break;
        case 2: registerRegularSlicer("slicer", evalScalarField_impl<Context::FitPlane, false>, context.lower, context.upper, context.isHDSlicer?1024:256, context.axis, context.slice); break;
      }
    }
    ImGui::SameLine();
    ImGui::PopItemWidth();
}

int main(int argc, char** argv) {
    // Options
    polyscope::options::autocenterStructures = false;
    polyscope::options::programName = "poncascope";
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();

    if (argc > 1)
    {
        loadFile(argv[1],context);
    }
    else
        loadFile("assets/armadillo.obj", context);

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    delete context.knnGraph;
    return EXIT_SUCCESS;
}
