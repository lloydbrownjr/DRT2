# DRT2
A distributed real-time ray tracer.

## Building
Currently only building on bridges 2
```
module load cuda
module load anaconda3
conda create –name ray
conda activate ray
conda install -c conda-forge nccl
mkdir build
cd build
cmake ..
make
```

## Running
```
sbatch job-rt
```