# DRT2
A distributed real-time ray tracer.

## Building
Currently only building on bridges 2
```
module load cuda
module load mvapich2/2.3.5-gcc8.3.1
mkdir build
cd build
cmake ..
make
```

## Running
```
sbatch job-rt
```