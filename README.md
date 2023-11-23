# poncascope
Application demonstrating how easily to combine:
 - Ponca: for POiNt Cloud Analysis and acceleration structures (kdtree) [https://github.com/poncateam/ponca]
 - libIGL: for data loading [https://github.com/libigl/libigl]
 - Polyscope: for the GUI [https://github.com/nmwsharp/polyscope]

With the current version, you will be able to:
 - compute and visualise differential quantities (normal vectors, mean curvature, principal curvature),
 - compare several differential estimators based on Moving Least Squares reconstruction,
 - play with reconstruction parameters, control timings, and more...

Computations are all done using Ponca on polyscope datastructures (see code for more details on data biding). Spatial queries are accelerated using Ponca Kdtree.

## Compilation instructions
```bash
git clone https://github.com/poncateam/poncascope.git # Fetch repository
cd poncascope
git submodule update --recursive --init               # Get dependencies: Polyscope, Ponca
mkdir build && cd build                               # Goto to compilation directory
cmake ../ -DCMAKE_BUILD_TYPE=Release                  # Configure in release mode
make                                                  # Compile
```

## Gallery

### Main features
[![Alt text](https://user-images.githubusercontent.com/6310221/134690163-f8ea4965-2e6c-4a84-9caa-d553fbe4e40c.png)](https://youtu.be/WRqO93rEy6s)

### Mean curvature estimation
<img width="1165" alt="image" src="https://user-images.githubusercontent.com/6310221/134543845-2f9094dd-1025-482a-b735-504b9cd8c5cd.png">

### Principale curvatures estimation
<img width="1165" alt="image" src="https://user-images.githubusercontent.com/6310221/134542628-bbce2151-b6b8-43b1-82d0-b869e5ef373a.png">
