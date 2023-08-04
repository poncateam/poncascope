#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"

#include <iostream>
#include <utility>
#include <chrono>
#include "defines.h"

// gui stuff as a global variable to be able to call it from the callback
std::unique_ptr<GUI> gui;

void callback(){
    gui->mainCallBack();
}

int main(int argc, char **argv) {

    // Options
    polyscope::options::autocenterStructures = false;
    polyscope::options::programName = "poncascope";
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;
    polyscope::options::groundPlaneEnabled = false;
    polyscope::view::bgColor = std::array<float, 4> {0.185, 0.185, 0.185, 0};


    // Initialize polyscope
    polyscope::init();

    // Instantiate the GUI
    gui = std::make_unique<GUI>(GUI());

    // Add the callback
    polyscope::state::userCallback = callback;

    // Show the gui
    polyscope::show();

    return EXIT_SUCCESS;
}
