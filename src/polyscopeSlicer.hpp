#pragma once
#include <vector>
#include <functional>
#include <polyscope/surface_mesh.h>
#include <polyscope/polyscope.h>

template<typename Point,typename Functor>
struct PolyscopeSlicer
{
  
  Functor  _implicit;
  Point _lowerBound;
  Point _upperBound;
  size_t _nbSteps;
  size_t _axis;
  size_t _slice;
  std::string _name;
  
  PolyscopeSlicer(std::string name,
                  Functor &implicit,
                  const Point &lowerBound,
                  const Point &upperBound,
                  size_t nbSteps,
                  size_t axis): _name(name),_implicit(implicit),_lowerBound(lowerBound),
  _upperBound(upperBound),_nbSteps(nbSteps),_axis(axis)
  {
  }
    
  polyscope::SurfaceMesh* regiterSlicer(size_t slice=0)
  {
    _slice = slice;
    auto dim1 = (_axis+1)%3;
    auto dim2 = (_axis+2)%3;
    
    double du = (_upperBound[dim1]-_lowerBound[dim1])/(double)_nbSteps;
    double dv = (_upperBound[dim2]-_lowerBound[dim2])/(double)_nbSteps;
    double dw = (_upperBound[_axis]-_lowerBound[_axis])/(double)_nbSteps;
    
    double u = _lowerBound[dim1];
    double v = _lowerBound[dim2];
    double w = _lowerBound[_axis] + _slice*dw;
    
    Point p;
    Point vu,vv;
    switch (_axis) {
      case 0: p=Point(w,u,v); vu=Point(0,du,0); vv=Point(0,0,dv);break;
      case 1: p=Point(u,w,v); vu=Point(du,0,0); vv=Point(0,0,dv);break;
      case 2: p=Point(u,v,w); vu=Point(du,0,0); vv=Point(0,dv,0);break;
    }
    
    std::vector<Point> vertices(_nbSteps*_nbSteps);
    std::vector<double> values(_nbSteps*_nbSteps);
    std::vector<std::array<size_t,4>> faces;
    std::array<size_t,4> face;
    
    for(size_t id=0; id < _nbSteps*_nbSteps; ++id)
    {
      auto i = id % _nbSteps;
      auto j = id / _nbSteps;
      p = _lowerBound + i*vu + j*vv;
      p[_axis] += _slice*dw;
      
      vertices[id] = p;
      values[id]  = _implicit(p);
      face = { id, id+1, id+1+_nbSteps, id+_nbSteps };
      if (((i+1) < _nbSteps) && ((j+1)<_nbSteps))
        faces.push_back(face);
    }
    std::cout<<"vertices: "<<vertices.size()<<std::endl;
    auto psm = polyscope::registerSurfaceMesh(_name, vertices,faces);
    psm->addVertexScalarQuantity("values",values)->setEnabled(true);
    return psm;
  }
  
  
  void updateSlice(size_t slice)
  {
    auto dim1 = (_axis+1)%3;
    auto dim2 = (_axis+2)%3;
    double du = (_upperBound[dim1]-_lowerBound[dim1])/(double)_nbSteps;
    double dv = (_upperBound[dim2]-_lowerBound[dim2])/(double)_nbSteps;
    double dw = (_upperBound[_axis]-_lowerBound[_axis])/(double)_nbSteps;
    double u = _lowerBound[dim1];
    double v = _lowerBound[dim2];
    double w = _lowerBound[_axis] + slice*dw;
    
    Point p;
    Point vu,vv;
    switch (_axis) {
      case 0: p=Point(w,u,v); vu=Point(0,du,0); vv=Point(0,0,dv);break;
      case 1: p=Point(u,w,v); vu=Point(du,0,0); vv=Point(0,0,dv);break;
      case 2: p=Point(u,v,w); vu=Point(du,0,0); vv=Point(0,dv,0);break;
    }
    
    std::vector<Point> vertices(_nbSteps*_nbSteps);
    std::vector<double> values(_nbSteps*_nbSteps);
    for(size_t id=0; id < _nbSteps*_nbSteps; ++id)
    {
      auto i = id % _nbSteps;
      auto j = id / _nbSteps;
      p = _lowerBound + i*vu + j*vv;
      vertices[id] = p;
      values[id]  = _implicit(p);
    }
    //auto psm = polyscope::registerSurfaceMesh("Slice "+std::to_string(axis), vertices,faces);
    polyscope::getSurfaceMesh(_name)->updateVertexPositions(vertices);
    polyscope::getSurfaceMesh(_name)->addVertexDistanceQuantity("values",values);
  }
};



template<typename Point, typename Functor>
PolyscopeSlicer<Point,Functor> makeSlicer(std::string name,
                                          Functor & implicit,
                                          const Point &lowerBound,
                                          const Point &upperBound,
                                          size_t nbSteps,
                                          size_t axis)
{
  return PolyscopeSlicer<Point,Functor>(name,
                                        implicit,
                                        lowerBound,
                                        upperBound,
                                        nbSteps,
                                        axis);
}
