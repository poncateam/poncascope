#pragma once

#include "Ponca/Fitting"
#include "Ponca/SpatialPartitioning"
#include "poncaAdapters.hpp"

// Types definition

using Scalar             = double;
using VectorType         = Eigen::Matrix<Scalar, 3,1>;
using PPAdapter          = BlockPointAdapter<Scalar>;
using KdTree             = Ponca::KdTree<PPAdapter>;
using KnnGraph           = Ponca::KnnGraph<PPAdapter>;

// Weighting functions

using SmoothWeightFunc   = Ponca::DistWeightFunc<PPAdapter, Ponca::SmoothWeightKernel<Scalar> >;
using ConstWeightFunc    = Ponca::DistWeightFunc<PPAdapter, Ponca::ConstantWeightKernel<Scalar> >;

// Fitting methods

using basket_dryFit                      =  Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::DryFit>;            

using basket_AlgebraicShapeOperatorFit   =  Ponca::BasketDiff<
                                            Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::OrientedSphereFit>,
                                            Ponca::DiffType::FitSpaceDer,
                                            Ponca::OrientedSphereDer, Ponca::MlsSphereFitDer,
                                            Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

using basket_AlgebraicPointSetSurfaceFit =  Ponca::BasketDiff<
                                            Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::OrientedSphereFit>,
                                            Ponca::DiffType::FitSpaceDer,
                                            Ponca::OrientedSphereDer,
                                            Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;

using basket_planeFit                    =  Ponca::BasketDiff<
                                                Ponca::Basket<PPAdapter, SmoothWeightFunc, Ponca::CovariancePlaneFit>,
                                                Ponca::DiffType::FitSpaceDer,
                                                Ponca::CovariancePlaneDer,
                                                Ponca::CurvatureEstimatorBase, Ponca::NormalDerivativesCurvatureEstimator>;
