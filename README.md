## SHREC 2025: Protein Shape Classification

This repository contains the NIT Tsuyama College team's codebase for
the [SHREC 2025: Protein Shape Classification track](http://shrec2025.drugdesign.fr/).

### Environment

We run the codebase on Ubuntu 24.04.2 LTS in WSL2 of Windows 11.
Dependencies are as follows:

- Point Cloud Library (PCL) 1.14.0 or later
- Eigen 3.4.0 or later
- CMake 3.15 or later
- Python 3.13.2 or later
- scikit-learn 1.6.1 or later

### Usage

Build the codebase and prepare the dataset as follows:

```bash
$ cd ~
$ git clone https://github.com/yoshoku/shrec2025_protein.git
$ cd shrec2025_protein/bin
$ cmake ../src
$ make
$ cd ../dataset
$ wget http://shrec2025.drugdesign.fr/files/train_set.csv
$ wget http://shrec2025.drugdesign.fr/files/test_set.csv
$ cd train_set
$ wget http://shrec2025.drugdesign.fr/files/train_set_vtk.tar.gz
$ tar xvzf train_set_vtk.tar.gz
$ cd ../test_set
$ wget http://shrec2025.drugdesign.fr/files/test_set_vtk.tar.gz
$ tar xvzf test_set_vtk.tar.gz
```

Execute the following command to generate the feature files from the VTK files and the submission file:

```bash
$ cd ~/shrec2025_protein/dataset/train_set
$ find . -name "*.vtk" | sed -e s/\.vtk// | awk '{print $1 ".vtk " $1 ".dat"}' | xargs -t -n 2 ../../bin/vtk2feat.bin
$ cd ../test_set
$ find . -name "*.vtk" | sed -e s/\.vtk// | awk '{print $1 ".vtk " $1 ".dat"}' | xargs -t -n 2 ../../bin/vtk2feat.bin
$ cd ../../ml
$ python main.py > submission.csv
```

### License

This project is licensed under the [MIT License](https://github.com/yoshoku/shrec2025_protein/blob/main/LICENSE.txt).
In addition, the codebase uses the [sobol sequence generator](https://people.sc.fsu.edu/~jburkardt/cpp_src/sobol/sobol.html) that is implemented by John Burkardt.
It is licensed under the MIT Licsense.
