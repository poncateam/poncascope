#include "io.hpp"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

#define TINYGLTF3_IMPLEMENTATION
#define TINYGLTF3_ENABLE_FS          // enable file I/O
#define TINYGLTF3_ENABLE_STB_IMAGE   // enable image decoding
#include "tiny_gltf_v3.h"

#include "polyscope/point_cloud.h"

#include "ImGuiFileDialog.h"

#include <filesystem>
#include <string>

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;


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

bool loadObjUsingLibIGL(const std::string& path, const std::string& /*dir*/,
                        Context& context,
                        Eigen::MatrixXd &coords, Eigen::MatrixXd &normals)
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

bool loadGLTF(const std::string& path, const std::string& dir,
              Context& context,
              Eigen::MatrixXd &coords, Eigen::MatrixXd &normals)
{
    /* Read file into memory */
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "[TinyGLTF] Cannot open " <<path << std::endl;
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }

    std::vector<uint8_t> data((size_t)sz);
    size_t rd = fread(data.data(), 1, (size_t)sz, f);
    fclose(f);
    if ((long)rd != sz) { return false; }

    tg3_parse_options opts;
    tg3_parse_options_init(&opts);

    tg3_model model;
    tg3_error_stack errors;
    tg3_error_stack_init(&errors);

    tg3_error_code err = tg3_parse_auto(&model, &errors,
                                         data.data(), data.size(),
                                         dir.c_str(),
                                         (uint32_t)dir.size(),
                                         &opts);
    if (err != TG3_OK) return false;

    auto searchForPointsAndNormals = [&model,&coords, &normals]()
    {
        for (int m = 0; m != model.meshes_count; m++)
        {
            for (int p = 0; p != model.meshes[m].primitives_count; p++)
            {
                const tg3_primitive& prim = model.meshes[m].primitives[p];
                if (prim.mode == TG3_MODE_POINTS)
                {
                    bool found = false;
                    for (int a=0; a != prim.attributes_count; ++a)
                    {
                        auto& attrib = prim.attributes[a];
                        tg3_accessor accessor = model.accessors[attrib.value];



                        switch (hash(attrib.key.data))
                        {
                        case hash("POSITION"):
                            if (accessor.type == TG3_TYPE_VEC3)
                            {
                                auto count = accessor.count / 3; //VEC3
                                auto bufferViewId = accessor.buffer_view;
                                auto bufferView = model.buffer_views[bufferViewId];
                                auto buffer = model.buffers[bufferViewId];

                                // 1. Calculate the start pointer
                                const float* bufferPtr = reinterpret_cast<const float*>(
                                    buffer.data.data + bufferView.byte_offset + accessor.byte_offset);

                                // 2. Determine stride (if byteStride is 0, it's tightly packed)
                                int strideBytes = bufferView.byte_stride ? bufferView.byte_stride : (3 * sizeof(float));
                                int strideFloats = strideBytes / sizeof(float) - 3; // 3 is the length of VEC3

                                using StrideMap = Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::ColMajor>,
                                                              0, Eigen::Stride<Eigen::Dynamic, 1>>;
                                StrideMap eigenPositions(
                                    bufferPtr,
                                    3,
                                    accessor.count,
                                    Eigen::Stride<Eigen::Dynamic, 1>(strideFloats, 1)
                                );
                                // auto stride = model.buffer_views[bufferViewId].byte_stride;
                                // const auto *buf = (const float*)(model.buffers[bufferViewId].data.data[bufferView.byte_offset + accessor.byte_offset]);
                                coords = eigenPositions.transpose().cast<double>();
                                normals = coords.rowwise().normalized();;
                                found = true;
                            }
                            break;
                        default:
                            break;
                        }
                    }
                    if (found) return true;
                }

            }

        }
        return false;
    };

    return searchForPointsAndNormals();
}

bool loadFile(const std::string& path, Context& context)
{
    std::cout << "[Poncascope] Load file " << path << std::endl;
    Eigen::MatrixXd newCloud, newNormals;

    std::filesystem::path filePath(path);
    bool loaded = false;

    const std::string ext = filePath.extension().string();
    const std::string dir = filePath.parent_path().string();
    switch (hash(ext.c_str()))
    {
    case hash(".obj"):
        loaded = loadObjUsingLibIGL(path, dir, context, newCloud, newNormals);
        break;
    case hash(".gltf"):
    case hash(".glb"):
        loaded = loadGLTF(path, dir, context, newCloud, newNormals);
        break;
    default:
        loaded = false;
    }

    std::cout << newCloud.rows() << " " << newCloud.cols() << std::endl;

    if (!loaded) return false;

    // Check if normals have been properly loaded
    /// \fixme : should not abort, but rather compute normals using Ponca.
    // {
    //     int nbUnitNormal = int(newNormals.rowwise().squaredNorm().sum());
    //     if ( nbUnitNormal != newCloud.rows() ) {
    //         std::cerr << "[Poncascope] Point cloud has no normals, aborting" << std::endl;
    //         return false;
    //     }
    // }

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
