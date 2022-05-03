# DRT2
A distributed real-time ray tracer.

## Building
Currently only building on bridges 2
```
module load cuda
mkdir build
cd build
cmake ..
make
```

## Running
To run the basic frame generation test
```
sbatch job-rt
```
To benchmark frame generation throughput
```
sbatch job-rt-bench
```