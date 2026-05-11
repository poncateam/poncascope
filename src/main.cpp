#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"

#include <iostream>
#include <utility>

// This file defines all the main types + data shared across the application components
#include "./context.hpp"
#include "./io.hpp"
#include "./estimators.hpp"
#include "./slicer.hpp"
#include "datastructures.hpp"

Context context;

/// Define Polyscope callbacks
void callback() {

    ImGui::PushItemWidth(100);

    callback_io(context);
    callback_datastructures(context);
    callback_estimators(context);
    callback_slicer(context);

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
