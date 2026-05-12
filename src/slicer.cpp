#include "slicer.hpp"

#include <vector>
#include <array>
#include <polyscope/surface_mesh.h>
#include <polyscope/polyscope.h>

using namespace Ponca;
using Scalar     = Context::Scalar;
using VectorType = Context::VectorType;

///Evaluate scalar field for generic FitType.
///// \tparam FitT Defines the type of estimator used for computation
template<typename FitT, bool isSigned = true>
Scalar evalScalarField_impl(const VectorType& input_pos, Context& context)
{
    FitT fit;
    fit.setNeighborFilter({input_pos, context.NSize}); // weighting function using current pos (not input pos)
    Ponca::MLSEvaluationScheme<Scalar> mls_evaluation_scheme (context.mlsIter, Scalar(context.mlsEpsilon));
    auto res = mls_evaluation_scheme.computeWithIds(fit, context.tree.rangeNeighbors(input_pos, context.NSize),
        context.tree.points());

    if(!fit.isStable()) {
        // not enough neighbors (if far from the point cloud)
        return Scalar(0); //std::numeric_limits<Scalar>::max();
    }

    const Scalar current_value = isSigned ? fit.potential(input_pos) : std::abs(fit.potential(input_pos));
    // current_gradient = fit.primitiveGradient(input_pos);
    return current_value;
}

/**
 * Create and register a polyscope surface mesh that slices a given implicit function.
 * This function uses a regular grid and evaluates the implicit function at the grid vertex positions.
 *
 * The implicit function could be a C/C++ function or any lambda taking as input a Point and returning a double.
 *
 * Note: if openmp is available, the implicit function is evaluated in parallel.
 *
 * @param name the name of the slicer
 * @param implicit the implicit function that maps points (type Point) to scalars.
 * @param nbSteps step size for the regular grid construction (eg 256)
 *
 * @return a pointer to the polyscope surface mesh object
 *
 * @author David Coeurjolly <david.coeurjolly@cnrs.fr>
 */
template<typename Functor>
polyscope::SurfaceMesh* registerRegularSlicer(const std::string &name,
                                              const Functor &implicit,
                                              size_t nbSteps,
                                              Context& context)
{
  size_t sliceid = static_cast<size_t>(std::floor(context.slice*nbSteps));

  auto dim1 = (context.axis+1)%3;
  auto dim2 = (context.axis+2)%3;

  double du = (context.upper[dim1]-context.lower[dim1])/(double)nbSteps;
  double dv = (context.upper[dim2]-context.lower[dim2])/(double)nbSteps;
  double dw = (context.upper[context.axis]-context.lower[context.axis])/(double)nbSteps;

  double u = context.lower[dim1];
  double v = context.lower[dim2];
  double w = context.lower[context.axis] + sliceid*dw;

  VectorType p;
  VectorType vu,vv;
  switch (context.axis) {
    case 0: p=VectorType(w,u,v); vu=VectorType(0,du,0); vv=VectorType(0,0,dv);break;
    case 1: p=VectorType(u,w,v); vu=VectorType(du,0,0); vv=VectorType(0,0,dv);break;
    case 2: p=VectorType(u,v,w); vu=VectorType(du,0,0); vv=VectorType(0,dv,0);break;
  }

  std::vector<VectorType> vertices(nbSteps*nbSteps);
  std::vector<double> values(nbSteps*nbSteps);
  std::vector<std::array<size_t,4>> faces;
  faces.reserve(nbSteps*nbSteps);
  std::array<size_t,4> face;

  //Regular grid construction
  for(size_t id=0; id < nbSteps*nbSteps; ++id)
  {
    auto i = id % nbSteps;
    auto j = id / nbSteps;
    p = context.lower + i*vu + j*vv;
    p[context.axis] += sliceid*dw;
    vertices[id] = p;
    face = { id, id+1, id+1+nbSteps, id+nbSteps };
    if (((i+1) < nbSteps) && ((j+1)<nbSteps))
      faces.push_back(face);
  }

  //Evaluating the implicit function (in parallel using openmp)
#pragma omp parallel for default(none) shared(nbSteps,values,vertices,implicit, context)
  for(int id=0; id < nbSteps*nbSteps; ++id)
    values[id]  = implicit(vertices[id], context);

  //Polyscope registration
  auto psm = polyscope::registerSurfaceMesh(name, vertices,faces);
  psm->addVertexScalarQuantity("values",values)->setEnabled(true);
  return psm;
}

void callback_slicer(Context& context)
{
    ImGui::SeparatorText("Implicit function slicer");
    ImGui::SliderFloat("Slice", &context.slice, 0, 1.0); ImGui::SameLine();
    ImGui::Checkbox("HD", &context.isHDSlicer);
    ImGui::RadioButton("X axis", &context.axis, 0); ImGui::SameLine();
    ImGui::RadioButton("Y axis", &context.axis, 1); ImGui::SameLine();
    ImGui::RadioButton("Z axis", &context.axis, 2);
    const char* items[] = { "ASO", "APSS", "PSS"};
    static int item_current = 0;
    ImGui::Combo("Fit function", &item_current, items, IM_ARRAYSIZE(items));

    unsigned long size = context.isHDSlicer?1024:256;
    if (ImGui::Button("Update"))
    {
        switch(item_current)
        {
        case 0: registerRegularSlicer("slicer", evalScalarField_impl<Context::FitASO, true>   , size, context); break;
        case 1: registerRegularSlicer("slicer", evalScalarField_impl<Context::FitAPSS, true>  , size, context); break;
        case 2: registerRegularSlicer("slicer", evalScalarField_impl<Context::FitPlane, false>, size, context); break;
        }
    }
    ImGui::SameLine();
}
