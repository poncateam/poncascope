
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <random>
#include <string>
#include <iterator>

//Ponca
#include <Ponca/Fitting>
#include "Eigen/Eigen"
#include <Ponca/src/SpatialPartitioning/KdTree/kdTree.h>

// Polyscope
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/curve_network.h"

using namespace std;
using namespace Ponca;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DIMENSION 3

/*
   \brief Variant of the MyPoint class allowing to work with external raw data.
 
   Using this approach, ones can use the patate library with already existing
   data-structures and without any data-duplication.
 
   In this example, we use this class to find Normals and curvature  from a polygon file using KdTree.
 */
// This class defines the input data format
 
class MyPoint
{
public:
  enum {Dim = 3};
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Dim, 1>   VectorType;
  typedef Eigen::Matrix<Scalar, Dim, Dim> MatrixType;
 
  PONCA_MULTIARCH inline MyPoint(const VectorType& _pos    = VectorType::Zero(),
                                 const VectorType& _normal = VectorType::Zero())
    : m_pos(_pos), m_normal(_normal) {}
 
  PONCA_MULTIARCH inline const VectorType& pos()    const { return m_pos; }
  PONCA_MULTIARCH inline const VectorType& normal() const { return m_normal; }
 
  PONCA_MULTIARCH inline VectorType& pos()    { return m_pos; }
  PONCA_MULTIARCH inline VectorType& normal() { return m_normal; }
 
 
private:
  VectorType m_pos, m_normal;
};

typedef MyPoint::Scalar Scalar;
typedef MyPoint::VectorType VectorType;


// Define related structure
typedef DistWeightFunc<MyPoint, SmoothWeightKernel<Scalar>> WeightFunc;


int knei = 50;
Scalar tmax = 10;

// to compute normals generalised using template and functors using KNN
template<typename FitT, typename Functor>
void computeKnn(const vector<MyPoint> points, const KdTree<MyPoint>& structure, Functor f)
{

    for(int i = 0; i < points.size(); i++){
        // set evaluation 
       // point and scale at the ith coordinate
        const VectorType& p = points.at(i).pos();
        // Here we now perform the fit
        FitT _fit;
        // Set a weighting function instance
        _fit.setWeightFunc(WeightFunc(tmax));
        // Set the evaluation position
        _fit.init(p);
        for( auto idx : structure.k_nearest_neighbors(p, knei) ){
            _fit.addNeighbor( points[idx] );
        }

        _fit.finalize();
	     f( i, _fit );
    }
}

// Your callback functions
void myCallback()
{

    // number of point cloud can be changed according to suitability
    int num_points = 50000;
    // Since options::openImGuiWindowForUserCallback == true by default,
    // we can immediately start using ImGui commands to build a UI

    ImGui::PushItemWidth(100); // Make ui elements 100 pixels wide,
                               // instead of full width. Must have
                               // matching PopItemWidth() below.

    if (ImGui::TreeNode("Fitting"))
    {
        /*LINE FITTING*/
        if (ImGui::TreeNode("Line Fitting"))
        {
            
            ImGui::InputInt("Variable K", &knei);          // set a float variable
            ImGui::InputDouble("Scalar attribute", &tmax);  // set a double variable

            // random points on point cloud
            vector<VectorType> positions;
            // set the structure/quadric for the fit
            for(int i =0 ;i < num_points; i++){
                VectorType p = VectorType::Random();
                p[2] =- 0.5;
                p[0] = -2*(p[1]) + 0.5;
                positions.push_back(p);
            }

            /* visualize! */
            polyscope::registerPointCloud("Line positions", positions);
            // Buttons for projections
            if (ImGui::Button("Find Projections"))
            {
                typedef Basket<MyPoint, WeightFunc, CovarianceLineFit> Linefit;
                
                vector<MyPoint> points;
                for(int i = 0; i < num_points; i++){
                    points.push_back({positions[i], {0,0,0}});
                }

                KdTree<MyPoint> kdtree(points);
                
                std::vector< VectorType > projections(points.size()); 

                computeKnn<Linefit>(points, kdtree, [&points, &projections]( int i,  Linefit& _fit ) 
                {
                    if(_fit.isStable() )
                    {
                        projections[i] = _fit.project(points[i].pos());
                    } 
                });

                 // visualise projections
                 polyscope::registerPointCloud("Line projection", projections);
            }

            ImGui::TreePop();
        }

        /*PLANE FITTING*/
        if (ImGui::TreeNode("Plane Fitting"))
        {
            
            ImGui::InputInt("Variable K", &knei);          // set a float variable
            ImGui::InputDouble("Scalar attribute", &tmax);  // set a double variable

            // random points on point cloud
            vector<VectorType> positions;
            // set the structure/quadric for the fit
            for(int i =0 ;i < num_points; i++){
                VectorType p = VectorType::Random();
               
                p[2] = -(3*p[0] + 2*p[1]);
                positions.push_back(p);
            }

            /* visualize! */
            polyscope::registerPointCloud("Plane positions", positions);

             // Buttons for projections
            if (ImGui::Button("Find Projections"))
            {
                typedef Basket<MyPoint, WeightFunc, CovariancePlaneFit> PlaneFit;

                vector<MyPoint> points;
                for(int i = 0; i < num_points; i++){
                    points.push_back({positions[i], {0,0,0}});
                }

                KdTree<MyPoint> kdtree(points);
                
                std::vector< VectorType > projections(points.size()); 

                computeKnn<PlaneFit>(points, kdtree, [&points, &projections]( int i,  PlaneFit& _fit ) 
                {
                    if(_fit.isStable() )
                    {
                        projections[i] = _fit.project(points[i].pos());
                    } 
                });
                 // visualise projections
                 polyscope::registerPointCloud("Plane projection", projections);
            }

            ImGui::TreePop();
        }
        /*SPHERE FITTING*/
        if (ImGui::TreeNode("Sphere Fitting"))
        {
            
            ImGui::InputInt("Variable K", &knei);          // set a float variable
            ImGui::InputDouble("Scalar attribute", &tmax);  // set a double variable

            // random points on point cloud
            vector<VectorType> positions;
            // set the structure/quadric for the fit
            for(int i =0 ;i < num_points; i++){
                VectorType p = VectorType::Random();
               
                p[2] = sqrt(1 -p[0] * p[0] - p[1] * p[1]);
                positions.push_back(p);
            }

            /* visualize! */
            polyscope::registerPointCloud("Sphere positions", positions);
             // Buttons for projections
            if (ImGui::Button("Find Projections"))
            {
                typedef Basket<MyPoint, WeightFunc, OrientedSphereFit> SphereFit;

               

                vector<MyPoint> points;
                for(int i = 0; i < num_points; i++){
                    points.push_back({positions[i], {0,0,0}});
                }

                KdTree<MyPoint> kdtree(points);
                
                std::vector< VectorType > projections(points.size()); 

                 /*Note : We need to compute normal for sphere  by covariance plane fit*/
                typedef Basket<MyPoint, WeightFunc, CovariancePlaneFit> PlaneFit;

                computeKnn<PlaneFit>(points, kdtree, [&points]( int i,  PlaneFit& _fit ) 
                {
                    if(_fit.isStable() )
                    {
                        points[i].normal() = _fit.primitiveGradient();
                    } 
                });

                computeKnn<SphereFit>(points, kdtree, [&points, &projections]( int i,  SphereFit& _fit )
                {
                    if(_fit.isStable() )
                    {
                        projections[i] = _fit.project(points[i].pos());
                    } 
                });
                 // visualise projections
                 polyscope::registerPointCloud("Sphere projection", projections);
             
            }

            ImGui::TreePop();
        }

        /*SURFACE FITTING*/
        if (ImGui::TreeNode("Surface Fitting"))
        {
            
            ImGui::InputInt("Variable K", &knei);          // set a float variable
            ImGui::InputDouble("Scalar attribute", &tmax);  // set a double variable

            // random points on point cloud
            vector<VectorType> positions;
            // set the structure/quadric for the fit
            for(int i =0 ;i < num_points; i++){
                VectorType p = VectorType::Random();
              
                p[2] = -(p[0] + p[1] * p[1]);
                positions.push_back(p);
            }

            /* visualize! */
            polyscope::registerPointCloud("Surface positions", positions);
             // Buttons for projections
            if (ImGui::Button("Find Projections"))
            {
                typedef Basket<MyPoint, WeightFunc, LeastSquareSurfaceFit> SurfaceFit;

                vector<MyPoint> points;
                for(int i = 0; i < num_points; i++){
                    points.push_back({positions[i], {0,0,0}});
                }

                KdTree<MyPoint> kdtree(points);
                
                std::vector< VectorType > projections(points.size()); 

                computeKnn<SurfaceFit>(points, kdtree, [&points, &projections]( int i,  SurfaceFit& _fit )
                {
                    if(_fit.isStable() )
                    {
                        projections[i] = _fit.project(points[i].pos());
                    } 
                });
                 // visualise projections
                 polyscope::registerPointCloud("Surface projection", projections);
            }

            ImGui::TreePop();
        }

        ImGui::TreePop();
    }

    ImGui::PopItemWidth();
}


int main(int argc, char **argv)
{   
    // Options
    polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;
    // Initialize polyscope
    polyscope::init();
    // Add the callback
    polyscope::state::userCallback = myCallback;
    // Show the gui
    polyscope::show();
    return 0;
}
