#include "estimators.hpp"

#include "polyscope/point_cloud.h"

#include <imgui.h>

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;


/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudMLS(const typename FitT::Scalar t, const Functor& f, Context& context){

    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.mlsIter, Scalar(context.mlsEpsilon));
#pragma omp parallel for private (mls_evaluation_scheme)
    for (int i = 0; i < context.tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({context.tree.points()[i], t});

        context.doOnNeighbors(i, [&](const auto& rangeNeighbors){
            mls_evaluation_scheme.computeWithIds(fit, rangeNeighbors, context.tree.points());
        });

        if (fit.isStable()) {
            f(i, fit);
        } else {
            std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
        }
    }
}

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudCNC(const typename FitT::Scalar t, const Functor& f, Context& context){
#pragma omp parallel for
    for (int i = 0; i < context.tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({context.tree.points()[i], t});

        std::vector<int> neighbors;
        neighbors.push_back(i);
        context.doOnNeighbors(i, [&neighbors](auto &&neighborhood){
            for (int j : neighborhood){
                neighbors.push_back(j);
            }
        });
        fit.computeWithIds(neighbors, context.tree.points());

        if (fit.isStable()) {
            f(i, fit);
        } else {
            std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
        }
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantities(const std::string& name, Context& context) {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj, &context]() {
        processPointCloudMLS<FitT>(context.NSize,
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj, &context]
                                (const int i, const FitT& fit){

            mean(i)       = fit.kMean();
            kmax(i)       = fit.kmax();
            kmin(i)       = fit.kmin();
            dmin.row( i ) = fit.kminDirection();
            dmax.row( i ) = fit.kmaxDirection();
            normal.row(i) = fit.primitiveGradient();
            proj.row(i)   = fit.getNeighborFilter().evalPos() - context.tree.points()[i].pos();
        }, context);
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj, &context]() {
                     context.cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     context.cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                     context.cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                    context.cloud->addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                    context.cloud->addVectorQuantity(name + " - projection", proj, polyscope::VectorType::AMBIENT);

                 });
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantitiesCNC(const std::string& name, Context& context) {
    int nvert = int(context.tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &dmin, &dmax, &context]() {
        processPointCloudCNC<FitT>(context.NSize,
                                [&mean, &kmin, &kmax, &dmin, &dmax]
                                (const int i, const FitT& fit){

            mean(i)         = fit.kMean();
            kmax(i)         = fit.kmax();
            kmin(i)         = fit.kmin();
            dmin.row( i )   = fit.kminDirection();
            dmax.row( i )   = fit.kmaxDirection();
        }, context);
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &dmin, &dmax, &context]() {
                     context.cloud->addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     context.cloud->addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     context.cloud->addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                     context.cloud->addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * context.pointRadius);
                 });
}

/// Dry run: loop over all vertices + run MLS loops without computation
/// This function is useful to monitor the KdTree performances
inline void mlsDryRun(Context& context) {
    measureTime( "[Ponca] Dry run MLS ", [&context]() {
        processPointCloudMLS<Context::FitDry>( context.NSize, [](int, const Context::FitDry&){ }, context);
    });
}


void callback_estimators(Context& context)
{
    ImGui::Separator();
    ImGui::InputInt("Nb MLS Iterations", &context.mlsIter);
    ImGui::InputFloat("MLS Epsilon", &context.mlsEpsilon);
    ImGui::Separator();

    ImGui::Text("Differential estimators");
    if (ImGui::Button("Dry Run"))  mlsDryRun(context);
    ImGui::SameLine();
    if (ImGui::Button("Plane (PCA)")) // Compute curvature using Covariance Plane fitting
        estimateDifferentialQuantities<Context::FitPlaneDiff>("PSS",context);
    ImGui::SameLine();
    if (ImGui::Button("APSS")) // Compute curvature using APSS
        estimateDifferentialQuantities<Context::FitAPSSDiff>("APSS",context);
    ImGui::SameLine();
    if (ImGui::Button("ASO")) // Compute curvature using Algebraic Shape Operator
        estimateDifferentialQuantities<Context::FitASODiff>("ASO",context);

    ImGui::Text("Corrected Normal Current estimator");
    if (ImGui::Button("Uniform"))
        estimateDifferentialQuantitiesCNC<Context::FitCNCUniform>("CNC - Uniform",context);
    ImGui::SameLine();
    if (ImGui::Button("Independent"))
        estimateDifferentialQuantitiesCNC<Context::FitCNCIndep>("CNC - Independent",context);
    ImGui::SameLine();
    if (ImGui::Button("Hexagram"))
        estimateDifferentialQuantitiesCNC<Context::FitCNCHex>("CNC - Hexagram",context);
    ImGui::SameLine();
    if (ImGui::Button("AvgHexagram"))
        estimateDifferentialQuantitiesCNC<Context::FitCNCAvgHex>("CNC - AvgHexagram",context);
}
