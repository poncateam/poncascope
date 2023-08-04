void GUI::mainCallBack(){

    // Create a window
    ImGui::PushItemWidth(100);
    // The size of the window, the position is set by default
    ImGui::SetNextWindowSize(ImVec2(300, 600), ImGuiCond_FirstUseEver);

    cloudComputing();

}

void GUI::cloudComputingUpdateAll (){
    if (!all_computed) return;

    pointProcessing.measureTime("[Polyscope] Update diff quantities and projection", [this](){
        
        for (int i = 0; i < selectedQuantities.size(); ++i) {
            if (selectedQuantities[i]){
                std::string completeName = "[" + methodName + "] "+ quantityNames[i];
                addQuantities(polyscope_mainCloud,completeName, mainCloud.getDiffQuantities().getByName(quantityNames[i]));
            }
        }
    });

    all_computed = false;
    methodName = "";
}


template <typename FitT>
void GUI::methodForCloudComputing(const std::string& metName){
    std::string buttonName_all = metName;
    if (ImGui::Button(buttonName_all.c_str())){
        methodName = metName;
        all_computed = true;
        pointProcessing.computeDiffQuantities<FitT>(metName, mainCloud);
    }
}

void GUI::cloudComputing(){

    cloudComputingParameters();

    ImGui::Text("Differential estimators");

    if (ImGui::Button("Dry Run")){
        pointProcessing.mlsDryRun();
        lastDryRun = "Last time run : " + std::to_string(pointProcessing.getMeanNeighbors()) + " number of neighbors (mean) \n";
    }
    ImGui::SameLine();
    ImGui::Text(lastDryRun.c_str());

    methodForCloudComputing<basket_planeFit>("Plane (PCA)");

    ImGui::SameLine();

    methodForCloudComputing<basket_AlgebraicPointSetSurfaceFit>("APSS");

    ImGui::SameLine();

    methodForCloudComputing<basket_AlgebraicShapeOperatorFit>("ASO");

    cloudComputingUpdateAll();
}

void GUI::cloudComputingParameters(){

    ImGui::Text("Neighborhood collection");
    ImGui::SameLine();
    if(ImGui::Checkbox("Use KnnGraph", &pointProcessing.useKnnGraph))
        pointProcessing.recomputeKnnGraph();

    ImGui::InputInt("k-neighborhood size", &pointProcessing.kNN);
    ImGui::InputFloat("neighborhood size", &pointProcessing.NSize);
    ImGui::InputInt("source vertex", &pointProcessing.iVertexSource);
    ImGui::InputInt("Nb MLS Iterations", &pointProcessing.mlsIter);
    ImGui::SameLine();
    if (ImGui::Button("show knn")) addQuantities(polyscope_mainCloud, "knn", pointProcessing.colorizeKnn());
    ImGui::SameLine();
    if (ImGui::Button("show euclidean nei")) addQuantities(polyscope_mainCloud, "euclidean nei", pointProcessing.colorizeEuclideanNeighborhood());

    ImGui::Separator();

}

void GUI::addQuantities(polyscope::PointCloud *pc, const std::string &name, const Eigen::MatrixXd &values){
    if (values.cols() == 1){
        // Make values beeing a vector
        Eigen::VectorXd valuesVec = values.col(0);
        auto quantity = pc->addScalarQuantity(name, valuesVec);
        // Set bound [-5, 5] for the scalar quantity
        if (name != "knn" && name != "euclidean nei"){
            quantity->setMapRange(std::pair<double,double>(-5,5));
            quantity->setColorMap("coolwarm");
        }
        else {
            quantity->setColorMap("turbo");
        }
    }
    else 
        pc->addVectorQuantity(name, values);
}
