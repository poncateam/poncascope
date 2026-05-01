#pragma once


#include "./context.hpp"


bool loadFile(const std::string& path, Context& context);

// Populate polyscope callback
void callback_io(Context& context);

