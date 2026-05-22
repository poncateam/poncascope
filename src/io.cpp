#include "io.hpp"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#include "polyscope/point_cloud.h"
#include "happly.h"

#include "ImGuiFileDialog.h"

#include <filesystem>
#include <string>

using namespace Ponca;
using Scalar     = Context::Types::Scalar;
using VectorType = Context::Types::VectorType;

bool loadObjUsingLibIGL(const std::string& path, Context& context,
                        Eigen::MatrixXd &coords, Eigen::MatrixXd &normals,
                        Eigen::MatrixXi &meshF)
{
    bool worked;
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

bool loadPlyUsingHapply(const std::string& path, Context& context,
                        Eigen::MatrixXd &coords, Eigen::MatrixXd &normals,
                        Eigen::MatrixXi &meshF)
{
    happly::PLYData plyIn (path);


    std::vector<double> xPos = plyIn.getElement("vertex").getProperty<double>("x");
    std::vector<double> yPos = plyIn.getElement("vertex").getProperty<double>("y");
    std::vector<double> zPos = plyIn.getElement("vertex").getProperty<double>("z");

    coords.resize(int(xPos.size()),3);

    std::vector<double> nxPos, nyPos, nzPos;
    bool hasNormals = plyIn.getElement("vertex").hasProperty("nx");
    if (hasNormals)
    {
        nxPos = plyIn.getElement("vertex").getProperty<double>("nx");
        nyPos = plyIn.getElement("vertex").getProperty<double>("ny");
        nzPos = plyIn.getElement("vertex").getProperty<double>("nz");
    }
    bool hasValidNormals = xPos.size() == nxPos.size();
    if (hasValidNormals)
        normals.resize(int(nxPos.size()),3);

    for (int i = 0; i < int(xPos.size()); i++) {
        coords.row(i) << xPos[i], yPos[i], zPos[i];
        if (hasValidNormals)
            normals.row(i) << nxPos[i], nyPos[i], nzPos[i];
    }

    if (plyIn.hasElement("face"))
    {
        auto faceList = plyIn.getFaceIndices<int>();
        // we assume to have faces with the same size (otherwise the process stops)
        size_t nbFaces  = faceList.size();
        size_t faceSize =  faceList[0].size();

        meshF.resize(nbFaces, faceSize);

        for (int f = 0; f != faceList.size(); ++ f)
        {
            if (faceList[f].size() != faceSize)
            {
                meshF = Eigen::MatrixXi();
                break;
            }

            using MapType = Eigen::Map<const Eigen::MatrixXi>;
            meshF.row(f) = MapType(faceList[f].data(), 1, faceSize);
        }

        if (meshF.cols()==3) // we have a triangle mesh
        {
            igl::per_vertex_normals(coords, meshF, normals);
        }
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
    Eigen::MatrixXi newFaces;

    std::filesystem::path filePath(path);
    bool loaded = false;

    const std::string ext = filePath.extension().string();
    switch (hash(ext.c_str()))
    {
    case hash(".obj"):
        loaded = loadObjUsingLibIGL(path, context, newCloud, newNormals, newFaces);
        break;
    case hash(".ply"):
        loaded = loadPlyUsingHapply(path, context, newCloud, newNormals, newFaces);
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
    // clear previous asset
    context.asset.clear();

    context.asset.cloudV = newCloud;
    context.asset.cloudN = newNormals;
    context.asset.meshF  = newFaces;
    // no need to delete the previous cloud, polyscope handles it
    context.asset.cloud  = polyscope::registerPointCloud("cloud", context.asset.cloudV);

    // Bounding Box (used in the slicer)
    context.asset.lower = context.asset.cloudV.colwise().minCoeff();
    context.asset.upper = context.asset.cloudV.colwise().maxCoeff();

    // Build Ponca KdTree
    measureTime( "[Ponca] Build KdTree", [&context]() {
        buildKdTree(context.asset.cloudV, context.asset.cloudN, context.asset.tree);
    });


    // Compute default point and neighborhood size according to the mean density
    measureTime( "[Ponca] Compute point radius according to mean knn distance", [&context]() {
        Scalar cloudMDist = 0;
        constexpr Scalar pointSizeFactor = 0.25;
        constexpr Scalar scaleFactor = 5;
#pragma omp parallel for
        for (int i = 0; i < context.asset.tree.samples().size(); ++i)
        {
            Scalar pointMDist = 0;
            VectorType p = context.asset.tree.points()[i].pos();
            context.doOnKNeighbors(i, [&pointMDist,p,&context](auto&& neighborhood){
                for (int j : neighborhood){
                    pointMDist += (p-context.asset.tree.points()[j].pos()).norm();
                }
            });
#pragma omp critical
            cloudMDist += pointMDist/Scalar(context.computeOpts.kNN);
        }
        context.computeOpts.pointRadius = pointSizeFactor * cloudMDist/Scalar(context.asset.tree.samples().size());
        context.computeOpts.NSize = scaleFactor * cloudMDist/Scalar(context.asset.tree.samples().size());
    });

    // Register the point cloud with Polyscope
    context.asset.cloud->setPointRadius(context.computeOpts.pointRadius);
    context.asset.cloud->setPointRenderMode(context.asset.cloud->nPoints() > 400000
        ? polyscope::PointRenderMode::Quad
        : polyscope::PointRenderMode::Sphere );
    polyscope::requestRedraw();

    std::cout << "[Poncascope] Loading file succeeded"<< std::endl;

    return true;
}

bool saveFile(const std::string& path, Context& context)
{
    happly::PLYData plyOut;
    plyOut.comments.push_back("File generated with Poncascope (https://github.com/poncateam/poncascope)");

    int nbVert = context.asset.cloud->points.size();

    // compute number of properties to export
    int nbPropS = 0;
    int nbPropV = 0;
    for (const auto& handler: context.asset.scalarQuantites) if (handler.save) ++nbPropS;
    for (const auto& handler: context.asset.vectorQuantites) if (handler.save) ++nbPropV;

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


    addVectorData(nbVert, context.asset.cloud->points.data, "vertex");
    for (int i = 0; i != nbPropS; ++i)
    {
        std::string name = context.asset.scalarQuantites[i].name;
        name.erase(std::remove_if(name.begin(), name.end(), isspace), name.end());
        plyOut.getElement("vertex").addProperty<float>(name,
            context.asset.scalarQuantites[i].ptr->quantity.values.data);
    }
    for (int i = 0; i != nbPropV; ++i)
    {
        std::string name = context.asset.vectorQuantites[i].name;
        name.erase(std::remove_if(name.begin(), name.end(), isspace), name.end());
        addVectorData(nbVert, context.asset.vectorQuantites[i].ptr->vectors.data, name);
    }

    // implementation is inspired from addFaceIndices, adapted to Eigen matrices
    int nbFaces = context.asset.meshF.rows();
    if (nbFaces != 0)
    {
        plyOut.addElement("face", nbFaces);

        // Cast to 32 bit
        std::vector<std::vector<int>> intInds;
        for (int f = 0; f != context.asset.meshF.rows(); f++) {
            std::vector<int> face;
            for (int i = 0; i!= context.asset.meshF.row(f).cols(); ++i)
            {
                face.push_back(context.asset.meshF.row(f)(i));
            }
            intInds.push_back(face);
        }

        // Store
        plyOut.getElement("face").addListProperty<int>("vertex_indices", intInds);
    }

    plyOut.write(path, happly::DataFormat::Binary);

    return true;
}

void callback_io(Context& context)
{
    // open Dialog Simple
    if (ImGui::Button("Open File Dialog")) {
        IGFD::FileDialogConfig config;
        config.path = context.ioOptions.loadPath;
        config.flags = ImGuiFileDialogFlags_DisableCreateDirectoryButton
            | ImGuiFileDialogFlags_ReadOnlyFileNameField;
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj,.ply", config);
    }
    // display
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            // std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();

            if (loadFile(filePathName, context)) context.ioOptions.loadPath = filePathName;
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }

    if (context.asset.cloud != nullptr)
    {
        ImGui::SameLine();
        if (ImGui::Button("Save File")) {
            IGFD::FileDialogConfig config;

            // if never saved before, compute a path according to loaded path
            if (context.ioOptions.savePath.empty())
            {
                std::filesystem::path filePath(context.ioOptions.loadPath);
                filePath.replace_extension(".ply");
                context.ioOptions.savePath = filePath.string();
            }
            config.path = context.ioOptions.savePath;
            config.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
            ImGuiFileDialog::Instance()->OpenDialog("SaveFileDlgKey", "Save File as...", ".ply", config);
        }
        // display
        if (ImGuiFileDialog::Instance()->Display("SaveFileDlgKey")) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                // std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                context.ioOptions.savePath = filePathName;
                ImGuiFileDialog::Instance()->Close();
                ImGui::OpenPopup("Choose export options");
            } else
                ImGuiFileDialog::Instance()->Close();
        }
    }

    if (ImGui::BeginPopupModal("Choose export options", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::SeparatorText("Scalar quantities");
        for (auto& handler: context.asset.scalarQuantites) ImGui::Checkbox(handler.name.c_str(), &(handler.save));

        ImGui::SeparatorText("Vector quantities");
        for (auto& handler: context.asset.vectorQuantites) ImGui::Checkbox(handler.name.c_str(), &(handler.save));

        if (ImGui::Button("Save")) {
            saveFile(context.ioOptions.savePath, context);
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

