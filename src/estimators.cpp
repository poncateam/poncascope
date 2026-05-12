#include "estimators.hpp"

#include "polyscope/point_cloud.h"

#include <imgui.h>

using namespace Ponca;
using Scalar     = Context::Types::Scalar;
using VectorType = Context::Types::VectorType;


/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudMLS(const typename FitT::Scalar t, const Functor& f, Context& context){

    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.computeOpts.mlsIter, Scalar(context.computeOpts.mlsEpsilon));
#pragma omp parallel for private (mls_evaluation_scheme)
    for (int i = 0; i < context.asset.tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({context.asset.tree.points()[i], t});

        context.doOnNeighbors(i, [&](const auto& rangeNeighbors){
            mls_evaluation_scheme.computeWithIds(fit, rangeNeighbors, context.asset.tree.points());
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
void processPointCloudSinglePass(const typename FitT::Scalar t, const Functor& f, Context& context){
#pragma omp parallel for
    for (int i = 0; i < context.asset.tree.samples().size(); ++i) {
        FitT fit;
        fit.setNeighborFilter({context.asset.tree.points()[i], t});

        std::vector<int> neighbors;
        neighbors.push_back(i);
        context.doOnNeighbors(i, [&neighbors](auto &&neighborhood){
            for (int j : neighborhood){
                neighbors.push_back(j);
            }
        });
        fit.computeWithIds(neighbors, context.asset.tree.points());

        if (fit.isStable()) {
            f(i, fit);
        } else {
            std::cerr << "Warning: fit " << i << " is not stable" << std::endl;
        }
    }
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<ProvidesPrincipalCurvatures FitT>
void estimateDifferentialQuantities(const std::string& name, Context& context) {
    int nvert = int(context.asset.tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj, &context]() {
        processPointCloudMLS<FitT>(context.computeOpts.NSize,
                                [&mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj, &context]
                                (const int i, const FitT& fit){

            mean(i)       = fit.kMean();
            kmax(i)       = fit.kmax();
            kmin(i)       = fit.kmin();
            dmin.row( i ) = fit.kminDirection();
            dmax.row( i ) = fit.kmaxDirection();
            normal.row(i) = fit.primitiveGradient();
            proj.row(i)   = fit.getNeighborFilter().evalPos() - context.asset.tree.points()[i].pos();
        }, context);
    });

    measureTime( "[Polyscope] Update differential quantities",
                 [&name, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj, &context]() {
                     context.asset.addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     context.asset.addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     context.asset.addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     context.asset.addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * context.computeOpts.pointRadius);
                     context.asset.addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * context.computeOpts.pointRadius);
                    context.asset.addVectorQuantity(name + " - normal", normal)->setVectorLengthScale(
                        Scalar(2) * context.computeOpts.pointRadius);
                    context.asset.addVectorQuantity(name + " - projection", proj, polyscope::VectorType::AMBIENT);

                 });
}

/// Generic processing function: traverse point cloud and compute mean, first and second curvatures + their direction
/// \tparam FitT Defines the type of estimator used for computation
template<typename FitT>
void estimateDifferentialQuantitiesCNC(const std::string& name, Context& context) {
    int nvert = int(context.asset.tree.samples().size());
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd dmin( nvert, 3 ), dmax( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [&mean, &kmin, &kmax, &dmin, &dmax, &context]() {
        processPointCloudSinglePass<FitT>(context.computeOpts.NSize,
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
                     context.asset.addScalarQuantity(name + " - Mean Curvature", mean)->setMapRange({-10,10});
                     context.asset.addScalarQuantity(name + " - K1", kmin)->setMapRange({-10,10});
                     context.asset.addScalarQuantity(name + " - K2", kmax)->setMapRange({-10,10});
                     context.asset.addVectorQuantity(name + " - K1 direction", dmin)->setVectorLengthScale(
                        Scalar(2) * context.computeOpts.pointRadius);
                     context.asset.addVectorQuantity(name + " - K2 direction", dmax)->setVectorLengthScale(
                        Scalar(2) * context.computeOpts.pointRadius);
                 });
}

/// Dry run: loop over all vertices + run MLS loops without computation
/// This function is useful to monitor the KdTree performances
inline void mlsDryRun(Context& context) {
    measureTime( "[Ponca] Dry run", [&context]() {
        processPointCloudSinglePass<Context::Types::FitDry>( context.computeOpts.NSize, [](int, const Context::Types::FitDry&){ }, context);
    });
}


void callback_estimators(Context& context)
{
    ImGui::SeparatorText("Utils");
    if (ImGui::Button("Dry Run"))  mlsDryRun(context);
    ImGui::Separator();

    ImGui::Text("MLS");
    ImGui::InputInt("Nb Iterations", &context.computeOpts.mlsIter);
    ImGui::SameLine();
    ImGui::InputFloat("Epsilon", &context.computeOpts.mlsEpsilon);
    if (ImGui::Button("Plane (PCA)")) // Compute curvature using Covariance Plane fitting
        estimateDifferentialQuantities<Context::Types::FitPlaneDiff>("PSS",context);
    ImGui::SameLine();
    if (ImGui::Button("APSS")) // Compute curvature using APSS
        estimateDifferentialQuantities<Context::Types::FitAPSSDiff>("APSS",context);
    ImGui::SameLine();
    if (ImGui::Button("ASO")) // Compute curvature using Algebraic Shape Operator
        estimateDifferentialQuantities<Context::Types::FitASODiff>("ASO",context);

    ImGui::Text("Corrected Normal Current estimator");
    if (ImGui::Button("Uniform"))
        estimateDifferentialQuantitiesCNC<Context::Types::FitCNCUniform>("CNC - Uniform",context);
    ImGui::SameLine();
    if (ImGui::Button("Independent"))
        estimateDifferentialQuantitiesCNC<Context::Types::FitCNCIndep>("CNC - Independent",context);
    ImGui::SameLine();
    if (ImGui::Button("Hexagram"))
        estimateDifferentialQuantitiesCNC<Context::Types::FitCNCHex>("CNC - Hexagram",context);
    ImGui::SameLine();
    if (ImGui::Button("AvgHexagram"))
        estimateDifferentialQuantitiesCNC<Context::Types::FitCNCAvgHex>("CNC - AvgHexagram",context);
}
