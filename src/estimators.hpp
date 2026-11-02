#pragma once

#include <Ponca/Ponca>
#include "./context.hpp"

// Populate polyscope callback
void callback_estimators(Context& context);

/// Generic processing function: traverse point cloud, compute fitting, and use functor to process fitting output
/// \note Functor is called only if fit is stable
template<typename FitT, typename Functor>
void processPointCloudMLS(const typename FitT::Scalar t, const Functor& f, Context& context){

    using Scalar = typename FitT::Scalar;
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
