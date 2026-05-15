#pragma once

#include "./context.hpp"
#include <string>

// Populate polyscope callback
void callback_datastructures(Context& context);

void generateLocalTopologyDisplay(Context& context, const std::string &name);
void generateFullTopologyDisplay(Context& context);

