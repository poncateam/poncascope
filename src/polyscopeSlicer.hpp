//MIT License
//
//Copyright (c) 2023 Ponca Development Group
//
//        Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//        copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//        copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

/// \file This file contains a slicer of the implicit function as a Polyscope surface mesh
/// \author David Coeurjolly <david.coeurjolly@cnrs.fr >

#pragma once
#include <vector>
#include <array>
#include <polyscope/surface_mesh.h>
#include <polyscope/polyscope.h>


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
 * @param lowerBound the bbox lower bound of the implicit function domain.
 * @param upperBound the bbox upper bound of the implicit function domain.
 * @param nbSteps step size for the regular grid construction (eg 256)
 * @param axis the axis to slice {0,1,2}.
 * @param slice the relative position of the slice in [0,1]
 *
 * @return a pointer to the polyscope surface mesh object
 */
template<typename Point,typename Functor>
polyscope::SurfaceMesh* registerRegularSlicer(const std::string &name,
                                              const Functor &implicit,
                                              const Point &lowerBound,
                                              const Point &upperBound,
                                              size_t nbSteps,
                                              size_t axis,
                                              float slice=0.0)
{
  size_t sliceid = static_cast<size_t>(std::floor(slice*nbSteps));
  
  auto dim1 = (axis+1)%3;
  auto dim2 = (axis+2)%3;
  
  double du = (upperBound[dim1]-lowerBound[dim1])/(double)nbSteps;
  double dv = (upperBound[dim2]-lowerBound[dim2])/(double)nbSteps;
  double dw = (upperBound[axis]-lowerBound[axis])/(double)nbSteps;
  
  double u = lowerBound[dim1];
  double v = lowerBound[dim2];
  double w = lowerBound[axis] + sliceid*dw;
  
  Point p;
  Point vu,vv;
  switch (axis) {
    case 0: p=Point(w,u,v); vu=Point(0,du,0); vv=Point(0,0,dv);break;
    case 1: p=Point(u,w,v); vu=Point(du,0,0); vv=Point(0,0,dv);break;
    case 2: p=Point(u,v,w); vu=Point(du,0,0); vv=Point(0,dv,0);break;
  }
  
  std::vector<Point> vertices(nbSteps*nbSteps);
  std::vector<double> values(nbSteps*nbSteps);
  std::vector<std::array<size_t,4>> faces;
  faces.reserve(nbSteps*nbSteps);
  std::array<size_t,4> face;
  
  //Regular grid construction
  for(size_t id=0; id < nbSteps*nbSteps; ++id)
  {
    auto i = id % nbSteps;
    auto j = id / nbSteps;
    p = lowerBound + i*vu + j*vv;
    p[axis] += sliceid*dw;
    vertices[id] = p;
    face = { id, id+1, id+1+nbSteps, id+nbSteps };
    if (((i+1) < nbSteps) && ((j+1)<nbSteps))
      faces.push_back(face);
  }
  
  //Evaluating the mplicit function (in parallel using openmp)
#pragma omp parallel for
  for(int id=0; id < nbSteps*nbSteps; ++id)
    values[id]  = implicit(vertices[id]);

  //Polyscope registration
  auto psm = polyscope::registerSurfaceMesh(name, vertices,faces);
  psm->addVertexScalarQuantity("values",values)->setEnabled(true);
  return psm;
}
