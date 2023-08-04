#pragma once

#include <Eigen/Dense>

class DiffQuantities {

    public:

        DiffQuantities(){
            m_vertices = Eigen::MatrixXd::Zero(0, 0);
            m_normals = Eigen::MatrixXd::Zero(0, 0);
            m_kMinDir = Eigen::MatrixXd::Zero(0, 0);
            m_kMaxDir = Eigen::MatrixXd::Zero(0, 0);
            m_kMin = Eigen::VectorXd::Zero(0);
            m_kMax = Eigen::VectorXd::Zero(0);
            m_kMean = Eigen::VectorXd::Zero(0);
        }

        DiffQuantities(const Eigen::MatrixXd &vertices, const Eigen::MatrixXd &normals, const Eigen::MatrixXd &kMinDir, const Eigen::MatrixXd &kMaxDir, const Eigen::VectorXd &kMin, const Eigen::VectorXd &kMax, const Eigen::VectorXd &kMean){
            m_vertices = vertices;
            m_normals = normals;
            m_kMinDir = kMinDir;
            m_kMaxDir = kMaxDir;
            m_kMin = kMin;
            m_kMax = kMax;
            m_kMean = kMean;
        }

        ~DiffQuantities(){
            m_vertices.resize(0, 0);
            m_normals.resize(0, 0);
            m_kMinDir.resize(0, 0);
            m_kMaxDir.resize(0, 0);
            m_kMin.resize(0);
            m_kMax.resize(0);
            m_kMean.resize(0);
        }

        void clear(){
            m_vertices.resize(0, 0);
            m_normals.resize(0, 0);
            m_kMinDir.resize(0, 0);
            m_kMaxDir.resize(0, 0);
            m_kMin.resize(0);
            m_kMax.resize(0);
            m_kMean.resize(0);
        }
    
        const Eigen::MatrixXd & getVertices(){
            return m_vertices;
        }

        const Eigen::MatrixXd & getNormals(){
            return m_normals;
        }

        const Eigen::MatrixXd & getKMinDir(){
            return m_kMinDir;
        }

        const Eigen::MatrixXd & getKMaxDir(){
            return m_kMaxDir;
        }

        const Eigen::VectorXd & getKMin(){
            return m_kMin;
        }

        const Eigen::VectorXd & getKMax(){
            return m_kMax;
        }

        const Eigen::VectorXd & getKMean(){
            return m_kMean;
        }

        const Eigen::MatrixXd getByName (const std::string &name){
            if (name == "Projections") return m_vertices;
            if (name == "Normals") return m_normals;
            if (name == "Min curvature direction") return m_kMinDir;
            if (name == "Max curvature direction") return m_kMaxDir;
            if (name == "Min curvature") return m_kMin;
            if (name == "Max curvature") return m_kMax;
            if (name == "Mean curvature") return m_kMean;
            return m_vertices;
        }

    private:

        Eigen::MatrixXd m_vertices;
        Eigen::MatrixXd m_normals;
        Eigen::MatrixXd m_kMinDir;
        Eigen::MatrixXd m_kMaxDir;
        Eigen::VectorXd m_kMin;
        Eigen::VectorXd m_kMax;
        Eigen::VectorXd m_kMean;

}; // class DiffQuantities

class MyPointCloud {

    public:

        MyPointCloud(){
            m_vertices = Eigen::MatrixXd::Zero(0, 0);
            m_normals = Eigen::MatrixXd::Zero(0, 0);
            m_size = 0;
        }

        MyPointCloud(Eigen::MatrixXd & vertices, Eigen::MatrixXd & normals){
            // Assert for the size of the matrix
            assert(vertices.rows() == normals.rows());
            assert(vertices.cols() == 3);
            assert(normals.cols() == 3);
            m_vertices = vertices;
            m_normals = normals;
            m_size = vertices.rows();
            updateBoundingBox();
        }

        ~MyPointCloud(){
            m_diffQuantities.clear();
            m_vertices.resize(0, 0);
            m_normals.resize(0, 0);
        }

        void clear(){
            m_diffQuantities.clear();
            m_vertices.resize(0, 0);
            m_normals.resize(0, 0);
        }

        void setDiffQuantities(const DiffQuantities & quantities){
            m_diffQuantities = quantities;
        }

        void setVertices(Eigen::MatrixXd &points){
            // Assert for the size of the matrix
            assert(points.rows() == m_size);
            assert(points.cols() == 3);
            m_vertices = points;
            updateBoundingBox();
        }

        void setNormals(Eigen::MatrixXd &normals){
            // Assert for the size of the matrix
            assert(normals.rows() == m_size);
            assert(normals.cols() == 3);
            m_normals = normals;
        }

        DiffQuantities & getDiffQuantities(){
            return m_diffQuantities;
        }

        Eigen::MatrixXd getVertices(){
            return m_vertices;
        }

        Eigen::MatrixXd getNormals(){
            return m_normals;
        }

        int getSize(){
            return m_size;
        }

        Eigen::VectorXd getMin(){
            return m_min;
        }

        Eigen::VectorXd getMax(){
            return m_max;
        }

    private:

        Eigen::MatrixXd m_vertices;
        Eigen::MatrixXd m_normals;
        DiffQuantities  m_diffQuantities;
        int m_size;

        Eigen::VectorXd m_min;
        Eigen::VectorXd m_max;

        void updateBoundingBox(){
            m_min = m_vertices.colwise().minCoeff();
            m_max = m_vertices.colwise().maxCoeff();
        }

}; // class MyPointCloud
