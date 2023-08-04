template <typename Functor>
void 
PointProcessing::measureTime( const std::string &actionName, Functor F ){
    using namespace std::literals; // enables the usage of 24h instead of e.g. std::chrono::hours(24)
    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();
    F(); // run process
    const auto end = std::chrono::steady_clock::now();
    std::cout << actionName << " in " << (end - start) / 1ms << "ms.\n";
}

template <typename Functor>
void PointProcessing::processRangeNeighbors(const int &idx, const Functor f){
    if(useKnnGraph)
        for (int j : knnGraph->range_neighbors(idx, NSize)){
            f(j);
        }
    else
        for (int j : tree.range_neighbors(idx, NSize)){
            f(j);
        }
}

template<typename FitT, typename Functor>
void PointProcessing::processOnePoint(const int &idx, const typename FitT::WeightFunction& w, Functor f){
        VectorType pos = tree.point_data()[idx].pos();
        for( int mm = 0; mm < mlsIter; ++mm) {
            FitT fit;
            fit.setWeightFunc(w);
            fit.init( pos );
            
            // Ponca::FIT_RESULT res = fit.computeWithIds(tree.range_neighbors(idx, NSize), tree.point_data() );
            
            // Loop until fit not need other pass, same process as fit.computeWithIds
            Ponca::FIT_RESULT res;
            do {
                fit.startNewPass();
                processRangeNeighbors(idx, [this, &fit](int j) {
                    fit.addNeighbor(tree.point_data()[j]);
                });
                res = fit.finalize();
            } while (res == Ponca::NEED_OTHER_PASS);
            
            if (res == Ponca::STABLE){

                pos = fit.project( pos );
                if ( mm == mlsIter -1 ) // last mls step, calling functor
                    f(idx, fit, pos);
            }
            else {
                std::cerr << "[Ponca][Warning] fit " << idx << " is not stable" << std::endl;
            }
        }
}

template<typename FitT, typename Functor>
void PointProcessing::processPointCloud(const typename FitT::WeightFunction& w, Functor f){

    int nvert = tree.index_data().size();
    // Traverse point cloud
    #pragma omp parallel for
    for (int i = 0; i < nvert; ++i) {
        processOnePoint<FitT, Functor>(i, w, f);
    }
}

template<typename FitT>
void
PointProcessing::computeDiffQuantities(const std::string &name, MyPointCloud &cloud) {
    
    int nvert = tree.index_data().size();

    // Allocate memory
    Eigen::VectorXd mean ( nvert ), kmin ( nvert ), kmax ( nvert );
    Eigen::MatrixXd normal( nvert, 3 ), dmin( nvert, 3 ), dmax( nvert, 3 ), proj( nvert, 3 );

    measureTime( "[Ponca] Compute differential quantities using " + name,
                 [this, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]() {
                    processPointCloud<FitT>(SmoothWeightFunc(NSize),
                                [this, &mean, &kmin, &kmax, &normal, &dmin, &dmax, &proj]
                                ( int i, const FitT& fit, const VectorType& mlsPos){

                                    mean(i) = fit.kMean();
                                    
                                    kmax(i) = fit.kmax();
                                    kmin(i) = fit.kmin();

                                    normal.row( i ) = fit.primitiveGradient();
                                    dmin.row( i )   = fit.kminDirection();
                                    dmax.row( i )   = fit.kmaxDirection();

                                    proj.row( i )   = mlsPos - tree.point_data()[i].pos();
                                    // proj.row( i )   = mlsPos;
                                });
                    });
    
    // Add differential quantities to the cloud
    cloud.setDiffQuantities(DiffQuantities(proj, normal,dmin, dmax, kmin, kmax, mean));
}

const Eigen::VectorXd PointProcessing::colorizeKnn() {

    int nvert = tree.index_data().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    closest(iVertexSource) = 2;
    processRangeNeighbors(iVertexSource, [&closest](int j){
        closest(j) = 1;
    });
    
    return closest;
}

const Eigen::VectorXd PointProcessing::colorizeEuclideanNeighborhood() {

    int nvert = tree.index_data().size();
    Eigen::VectorXd closest ( nvert );
    closest.setZero();

    SmoothWeightFunc w(NSize);

    closest(iVertexSource) = 2;
    const auto &p = tree.point_data()[iVertexSource];
    processRangeNeighbors(iVertexSource, [this, w,p,&closest](int j){
        const auto &q = tree.point_data()[j];
        closest(j) = w.w( q.pos() - p.pos(), q ).first;
    });
    return closest;
}

void PointProcessing::recomputeKnnGraph() {
    if(useKnnGraph) {
        measureTime("[Ponca] Build KnnGraph", [this]() {
            delete knnGraph;
            knnGraph = new KnnGraph(tree, kNN);
        });
    }
}

void PointProcessing::mlsDryRun() {
    int nvert = tree.index_data().size();
    m_meanNeighbors = Scalar(0);

    // Allocate memory
    measureTime( "[Ponca] Compute differential quantities using dry fit",
                 [this]() {
                    processPointCloud<basket_dryFit>(SmoothWeightFunc(NSize),
                                [this]
                                ( int, const basket_dryFit& fit, const VectorType&){
                                    m_meanNeighbors += fit.getNumNeighbors();
                    });
                });

    m_meanNeighbors /= nvert;    
}
