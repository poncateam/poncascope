#include "io.hpp"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#include "polyscope/point_cloud.h"
#include "happly.h"

#include "ImGuiFileDialog.h"

#include <filesystem>
#include <string>

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;

bool loadObjUsingLibIGL(const std::string& path, Context& context, Eigen::MatrixXd &coords, Eigen::MatrixXd &normals)
{
    bool worked;
    Eigen::MatrixXi meshF;
    measureTime( "[libIGL] obj file loading", [path, &coords, &meshF, &worked]()
    // For convenience: use libIGL to load a mesh, and store only the vertices location and normal vector
    {
        const std::string filename = path.c_str();
        worked = igl::readOBJ(filename, coords, meshF);
    } );

    if (worked) {
        if (meshF.cols()==3) // we have a triangle mesh
        {
            igl::per_vertex_normals(coords, meshF, normals);
        }
    }
    else {
        std::cerr << "[libIGL] An error occurred when loading file " << path
                  << std::endl;
        return false;
    }
    return true;
}

// Simple constexpr hash function
// Source: https://www.reddit.com/r/cpp/comments/jkw84k/strings_in_switch_statements_using_constexp/
constexpr size_t hash(const char* str){
    const long long p = 131;
    const long long m = 4294967291; // 2^32 - 5, largest 32 bit prime
    long long total = 0;
    long long current_multiplier = 1;
    for (int i = 0; str[i] != '\0'; ++i){
        total = (total + current_multiplier * str[i]) % m;
        current_multiplier = (current_multiplier * p) % m;
    }
    return total;
}

bool loadFile(const std::string& path, Context& context)
{
    std::cout << "[Poncascope] Load file " << path << std::endl;
    Eigen::MatrixXd newCloud, newNormals;

    std::filesystem::path filePath(path);
    bool loaded = false;

    const std::string ext = filePath.extension().string();
    switch (hash(ext.c_str()))
    {
    case hash(".obj"):
        loaded = loadObjUsingLibIGL(path, context, newCloud, newNormals);
        break;
    default:
        loaded = false;
    }

    if (!loaded) return false;

    // Check if normals have been properly loaded
    /// \fixme : should not abort, but rather compute normals using Ponca.
    {
        int nbUnitNormal = int(newNormals.rowwise().squaredNorm().sum());
        if ( nbUnitNormal != newCloud.rows() ) {
            std::cerr << "[Poncascope] Point cloud has no normals, aborting" << std::endl;
            return false;
        }
    }

    context.cloudV = newCloud;
    // no need to delete the previous cloud, polyscope handles it
    context.scalarQuantites.clear();
    context.vectorQuantites.clear();
    context.cloud = polyscope::registerPointCloud("cloud", context.cloudV);
    context.cloudN = newNormals;

    // Bounding Box (used in the slicer)
    context.lower = context.cloudV.colwise().minCoeff();
    context.upper = context.cloudV.colwise().maxCoeff();

    // Build Ponca KdTree
    measureTime( "[Ponca] Build KdTree", [&context]() {
        buildKdTree(context.cloudV, context.cloudN, context.tree);
    });


    // Compute default point and neighborhood size according to the mean density
    measureTime( "[Ponca] Compute point radius according to mean knn distance", [&context]() {
        Scalar cloudMDist = 0;
        constexpr Scalar pointSizeFactor = 0.25;
        constexpr Scalar scaleFactor = 5;
#pragma omp parallel for
        for (int i = 0; i < context.tree.samples().size(); ++i)
        {
            Scalar pointMDist = 0;
            VectorType p = context.tree.points()[i].pos();
            context.doOnKNeighbors(i, [&pointMDist,p,&context](auto&& neighborhood){
                for (int j : neighborhood){
                    pointMDist += (p-context.tree.points()[j].pos()).norm();
                }
            });
#pragma omp critical
            cloudMDist += pointMDist/Scalar(context.kNN);
        }
        context.pointRadius = pointSizeFactor * cloudMDist/Scalar(context.tree.samples().size());
        context.NSize = scaleFactor * cloudMDist/Scalar(context.tree.samples().size());
    });

    // Be sure that the KnnGraph is invalidated
    delete context.knnGraph;
    context.useKnnGraph = false;
    context.knnGraph = nullptr;

    // Register the point cloud with Polyscope
    context.cloud->setPointRadius(context.pointRadius);
    polyscope::requestRedraw();

    std::cout << "[Poncascope] Loading file succeeded"<< std::endl;

    return true;
}

bool saveFile(const std::string& path, Context& context)
{
    happly::PLYData plyOut;
    plyOut.comments.push_back("File generated with Poncascope (https://github.com/poncateam/poncascope)");

    int nbVert = context.cloud->points.size();

    // compute number of properties to export
    int nbPropS = 0;
    int nbPropV = 0;
    for (const auto& handler: context.scalarQuantites) if (handler.save) ++nbPropS;
    for (const auto& handler: context.vectorQuantites) if (handler.save) ++nbPropV;

    auto addVectorData = [&plyOut](int n, const std::vector<glm::vec3>& h, const std::string& name)
    {
        std::vector<double> xPos(n);
        std::vector<double> yPos(n);
        std::vector<double> zPos(n);
        for (size_t i = 0; i < n; i++) {
            const auto& v = h[i];
            xPos[i] = v[0];
            yPos[i] = v[1];
            zPos[i] = v[2];
        }

        // Store
        plyOut.addElement(name, n);
        plyOut.getElement(name).addProperty<double>("x", xPos);
        plyOut.getElement(name).addProperty<double>("y", yPos);
        plyOut.getElement(name).addProperty<double>("z", zPos);
    };


    addVectorData(nbVert, context.cloud->points.data, "vertex");
    for (int i = 0; i != nbPropS; ++i)
    {
        std::string name = context.scalarQuantites[i].name;
        name.erase(std::remove_if(name.begin(), name.end(), isspace), name.end());
        plyOut.getElement("vertex").addProperty<float>(name,
            context.scalarQuantites[i].ptr->quantity.values.data);
    }
    for (int i = 0; i != nbPropV; ++i)
    {
        std::string name = context.vectorQuantites[i].name;
        name.erase(std::remove_if(name.begin(), name.end(), isspace), name.end());
        addVectorData(nbVert, context.vectorQuantites[i].ptr->vectors.data, name);
    }

    plyOut.write(path, happly::DataFormat::Binary);

    return true;
}

void callback_io(Context& context)
{
    // open Dialog Simple
    if (ImGui::Button("Open File Dialog")) {
        IGFD::FileDialogConfig config;
        config.path = context.loadPath;
        config.flags = ImGuiFileDialogFlags_DisableCreateDirectoryButton
            | ImGuiFileDialogFlags_ReadOnlyFileNameField;
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", config);
    }
    // display
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            // std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();

            if (loadFile(filePathName, context)) context.loadPath = filePathName;
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }

    if (context.cloud != nullptr)
    {
        ImGui::SameLine();
        if (ImGui::Button("Save File")) {
            IGFD::FileDialogConfig config;

            // if never saved before, compute a path according to loaded path
            if (context.savePath.empty())
            {
                std::filesystem::path filePath(context.loadPath);
                filePath.replace_extension(".ply");
                context.savePath = filePath.string();
            }
            config.path = context.savePath;
            config.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
            ImGuiFileDialog::Instance()->OpenDialog("SaveFileDlgKey", "Save File as...", ".ply", config);
        }
        // display
        if (ImGuiFileDialog::Instance()->Display("SaveFileDlgKey")) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                // std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                context.savePath = filePathName;
                ImGuiFileDialog::Instance()->Close();
                ImGui::OpenPopup("Choose export options");
            } else
                ImGuiFileDialog::Instance()->Close();
        }
    }

    if (ImGui::BeginPopupModal("Choose export options", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::SeparatorText("Scalar quantities");
        for (auto& handler: context.scalarQuantites) ImGui::Checkbox(handler.name.c_str(), &(handler.save));

        ImGui::SeparatorText("Vector quantities");
        for (auto& handler: context.vectorQuantites) ImGui::Checkbox(handler.name.c_str(), &(handler.save));

        if (ImGui::Button("Save")) {
            saveFile(context.savePath, context);
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

