#include "io.hpp"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#include "polyscope/point_cloud.h"

#include "ImGuiFileDialog.h"

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;

bool loadFile(const std::string& path, Context& context)
{
    std::cout << "[Poncascope] Load file " << path << std::endl;
    // first block: output cloudV and cloudN
    Eigen::MatrixXd newCloud, newNormals;
    {
        bool worked;
        Eigen::MatrixXi meshF;
        measureTime( "[libIGL] obj file loading", [path, &newCloud, &meshF, &worked]()
        // For convenience: use libIGL to load a mesh, and store only the vertices location and normal vector
        {
            const std::string filename = path.c_str();
            worked = igl::readOBJ(filename, newCloud, meshF);
        } );

        if (worked) {
            if (meshF.cols()==3) // we have a triangle mesh
            {
                igl::per_vertex_normals(newCloud, meshF, newNormals);
            }
        }
        else {
            std::cerr << "[libIGL] An error occurred when loading file " << path
                      << std::endl;
            return false;
        }
    }

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

void callback_io(Context& context)
{
    // open Dialog Simple
    if (ImGui::Button("Open File Dialog")) {
        IGFD::FileDialogConfig config;
        config.path = context.lastPath;
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj", config);
    }
    // display
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            // std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();

            if (loadFile(filePathName, context)) context.lastPath = filePathName;
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }
}
