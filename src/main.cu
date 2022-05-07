#include <iostream>
#include <algorithm>
#include <string>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <vector>
#include "../common/vec3.h"
#include "../common/ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "cuda_errors.h"
#include "../common/options.h"
#include "cuda_runtime.h"
// #include "nccl.h"
#include "/opt/packages/mvapich2/intel/2.3.5-intel20.4/include/mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <unordered_map>


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// This method determines the color of a ray going through the scene by tracing it through the scene and hitting objects.
// It has been modified to use CUDA as described below.
// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state, int init_x, int init_y) {
    int i_local = threadIdx.x + blockIdx.x * blockDim.x;
    int j_local = threadIdx.y + blockIdx.y * blockDim.y;
    int i = i_local + init_x;
    int j = j_local + init_y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    int pixel_index_local = j_local*max_x + i_local;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index_local]);
}

__global__ void render(vec3 *frame_buffer, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state, int init_x, int init_y) {
    int i_local = threadIdx.x + blockIdx.x * blockDim.x + init_x;
    int j_local = threadIdx.y + blockIdx.y * blockDim.y + init_y;
    int i = i_local + init_x;
    int j = j_local + init_y;
    if((i >= max_x) || (j >= max_y)) return;
    // int pixel_index = j*max_x + i;
    int pixel_index_local = j_local*max_x + i_local;
    curandState local_rand_state = rand_state[pixel_index_local];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index_local] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    frame_buffer[pixel_index_local] = col;
}

__global__ void render_vec3_t(vec3_t *frame_buffer, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state, int init_x, int init_y) {
    int i_local = threadIdx.x + blockIdx.x * blockDim.x + init_x;
    int j_local = threadIdx.y + blockIdx.y * blockDim.y + init_y;
    int i = i_local + init_x;
    int j = j_local + init_y;
    if((i >= max_x) || (j >= max_y)) return;
    // int pixel_index = j*max_x + i;
    int pixel_index_local = j_local*max_x + i_local;
    curandState local_rand_state = rand_state[pixel_index_local];
    vec3_t col{0,0,0};
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        auto col_addition = color(r, world, &local_rand_state);
        col.x += col_addition.x();
        col.y += col_addition.y();
        col.z += col_addition.z();
    }
    rand_state[pixel_index_local] = local_rand_state;
    col.x = sqrt(col.x / float(ns));
    col.y = sqrt(col.y / float(ns));
    col.z = sqrt(col.z / float(ns));
    frame_buffer[pixel_index_local] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

vec3 camera_move_vector() {
    return vec3(0,0,-0.1);
}

// Moves the camera's origin to create a new scene
__global__ void move_cam(camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->origin += vec3(0,0,-0.1);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

// Writes the image to a ppm file.
void write_frame_buffer(vec3 *frame_buffer, int nx, int ny, int max_x, int max_y) {
    FILE *f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", max_x, max_y);
    for(int j=0; j < max_y; j++) {
        for(int i=0; i < max_x; i++) {
            vec3 col = frame_buffer[j*max_x + i];
            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);
            fprintf(f, "%d %d %d ", ir, ig, ib);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Writes the image to a ppm file.
void write_frame_buffer_arr(vec3 **frame_buffer_arr, int nx, int ny, int max_x, int max_y) {
    int size = sizeof frame_buffer_arr / sizeof frame_buffer_arr[0];
    int slice_height = max_y/size;
    FILE *f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", max_x, max_y);
    for(int j=0; j < max_y; j++) {
        for(int i=0; i < max_x; i++) {
            int slice = j/slice_height;
            vec3 col = frame_buffer_arr[slice][(j-(slice*slice_height))*max_x + i];
            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);
            fprintf(f, "%d %d %d ", ir, ig, ib);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Renders a single image and writes it to a ppm file.
void test_render(int image_height, int image_width, int samples_per_pixel) {
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width*image_height;
    size_t frame_buffer_size = num_pixels*sizeof(vec3);

    // allocate frame_buffer
    vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, frame_buffer_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state, 0, 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(frame_buffer, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state, 0, 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output frame_buffer as Image
    write_frame_buffer(frame_buffer, image_width, image_height, image_width, image_height);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(frame_buffer));

    cudaDeviceReset();
}

void benchmark_single(int image_height, int image_width, int samples_per_pixel, int num_frames_to_render) {
    int tx = 8;
    int ty = 8;

    std::cerr << "Benchmarking the rendering of " << image_width << "x" << image_height << " images with " << samples_per_pixel << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width*image_height;
    size_t frame_buffer_size = num_pixels*sizeof(vec3);

    // allocate frame_buffer
    vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, frame_buffer_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state, 0, 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    for (int i = 0; i < num_frames_to_render; i++) {
        // Render the current frame and make sure it worked.
        render<<<blocks, threads>>>(frame_buffer, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state, 0, 0);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        // Move the camera to create the next frame.
        move_cam<<<blocks, threads>>>(d_camera);
    }
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds to generate " << num_frames_to_render << " frames.\n";
    std::cerr << "Average FPS: " << (double)num_frames_to_render / timer_seconds << "\n";

    // Output frame_buffer as Image
    write_frame_buffer(frame_buffer, image_width, image_height, image_width, image_height);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(frame_buffer));

    cudaDeviceReset();
}

void benchmark_tiled(int image_height, int image_width, int samples_per_pixel, int num_frames_to_render) {
    // int tx = 8;
    // int ty = 8;

    // std::cerr << "Benchmarking the rendering of " << image_width << "x" << image_height << " images with " << samples_per_pixel << " samples per pixel ";
    // std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // ncclComm_t comms[4];

    // //managing 4 devices
    // int nDev = 4;
    // int devs[4] = { 0, 1, 2, 3 };

    // //allocating and initializing device buffers
    // cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    // int num_pixels = image_width*image_height;
    // size_t frame_buffer_size = num_pixels*sizeof(vec3);

    // int image_width_dev = image_width;
    // int image_height_dev = image_height / nDev;
    // if (image_width_dev * image_height_dev * nDev != image_height * image_width) {
    //     std::cerr << "Not an even split." << std::endl;
    //     exit(1);
    // }
    // int num_pixels_dev = num_pixels / nDev;
    // size_t frame_buffer_size_dev = frame_buffer_size / nDev;

    // vec3** frame_buffer = (vec3**)malloc(nDev * sizeof(vec3*));
    // curandState** d_rand_state = (curandState**)malloc(nDev * sizeof(curandState*));
    // curandState** d_rand_state2 = (curandState**)malloc(nDev * sizeof(curandState*));
    // hitable ***d_list = (hitable***)malloc(nDev * sizeof(hitable**));
    // int num_hitables = 22*22+1+3;
    // hitable ***d_world = (hitable***)malloc(nDev * sizeof(hitable**));
    // camera ***d_camera = (camera***)malloc(nDev * sizeof(camera**));

    // for (int i = 0; i < nDev; ++i) {
    //     checkCudaErrors(cudaSetDevice(i));

    //     // allocate frame_buffer
    //     checkCudaErrors(cudaMallocManaged((void **)&frame_buffer[i], frame_buffer_size_dev));

    //     // allocate random state
    //     checkCudaErrors(cudaMalloc((void **)&d_rand_state[i], num_pixels_dev*sizeof(curandState)));
    //     checkCudaErrors(cudaMalloc((void **)&d_rand_state2[i], 1*sizeof(curandState)));

    //     // we need that 2nd random state to be initialized for the world creation
    //     rand_init<<<1,1>>>(d_rand_state2[i]);
    //     checkCudaErrors(cudaGetLastError());
    //     checkCudaErrors(cudaDeviceSynchronize());

    //     // make our world of hitables & the camera
    //     checkCudaErrors(cudaMalloc((void **)&d_list[i], num_hitables*sizeof(hitable *)));
    //     checkCudaErrors(cudaMalloc((void **)&d_world[i], sizeof(hitable *)));
    //     checkCudaErrors(cudaMalloc((void **)&d_camera[i], sizeof(camera *)));
    //     create_world<<<1,1>>>(d_list[i], d_world[i], d_camera[i], image_width_dev, image_height_dev, d_rand_state2[i]);
    //     checkCudaErrors(cudaGetLastError());
    //     checkCudaErrors(cudaDeviceSynchronize());

    //     checkCudaErrors(cudaStreamCreate(s+i));
    // }

    // //initializing NCCL
    // NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // clock_t start, stop;
    // start = clock();

    // dim3 blocks(image_width_dev/tx+1,image_height_dev/ty+1);
    // dim3 threads(tx,ty);
    // for (int i = 0; i < nDev; ++i) {
    //     checkCudaErrors(cudaSetDevice(i));
    //     // Render our buffer
    //     render_init<<<blocks, threads>>>(image_width_dev, image_height_dev, d_rand_state[i], 0, i*image_height_dev);
    // }

    // //synchronizing on CUDA streams to wait for completion of NCCL operation
    // for (int i = 0; i < nDev; ++i) {
    //     checkCudaErrors(cudaSetDevice(i));
    //     checkCudaErrors(cudaGetLastError());
    //     checkCudaErrors(cudaStreamSynchronize(s[i]));
    // }

    // for (int f = 0; f < num_frames_to_render; f++) {
    //     for (int i = 0; i < nDev; ++i) {
    //         checkCudaErrors(cudaSetDevice(i));
    //         // Render the current frame and make sure it worked.
    //         render<<<blocks, threads>>>(frame_buffer[i], image_width_dev, image_height_dev, samples_per_pixel, d_camera[i], d_world[i], d_rand_state[i], 0, i*image_height_dev);
    //     }

    //     //synchronizing on CUDA streams to wait for completion of NCCL operation
    //     for (int i = 0; i < nDev; ++i) {
    //         checkCudaErrors(cudaSetDevice(i));
    //         checkCudaErrors(cudaGetLastError());
    //         checkCudaErrors(cudaStreamSynchronize(s[i]));
    //     }

    //     for (int i = 0; i < nDev; ++i) {
    //         checkCudaErrors(cudaSetDevice(i));
    //         // Move the camera to create the next frame.
    //         move_cam<<<blocks, threads>>>(d_camera[i]);
    //     }
    // }

    // stop = clock();
    // double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    // std::cerr << "took " << timer_seconds << " seconds to generate " << num_frames_to_render << " frames.\n";
    // std::cerr << "Average FPS: " << (double)num_frames_to_render / timer_seconds << "\n";

    // // Output frame_buffer as Image
    // write_frame_buffer_arr(frame_buffer, image_width, image_height, image_width, image_height);

    // //free device buffers
    // for (int i = 0; i < nDev; ++i) {
    //     checkCudaErrors(cudaSetDevice(i));
    //     // clean up
    //     checkCudaErrors(cudaDeviceSynchronize());
    //     free_world<<<1,1>>>(d_list[i],d_world[i],d_camera[i]);
    //     checkCudaErrors(cudaGetLastError());
    //     checkCudaErrors(cudaFree(d_camera[i]));
    //     checkCudaErrors(cudaFree(d_world[i]));
    //     checkCudaErrors(cudaFree(d_list[i]));
    //     checkCudaErrors(cudaFree(d_rand_state[i]));
    //     checkCudaErrors(cudaFree(d_rand_state2[i]));
    //     checkCudaErrors(cudaFree(frame_buffer[i]));
    // }

    // //finalizing NCCL
    // for(int i = 0; i < nDev; ++i)
    //     ncclCommDestroy(comms[i]);

    // printf("Success \n");

    // cudaDeviceReset();
    (void*)0;
}

MPI_Datatype VEC3;

// Initializes the camera origins data structure.
__global__ void init_origins_for_frames(vec3_t* camera_origins_for_frames, int num_frames_to_render, camera** d_camera) {
    auto current_origin = (*d_camera)->origin;
    auto move_vector = vec3(0,0,-0.1);
    camera_origins_for_frames[0] = {current_origin.x(), current_origin.y(), current_origin.z()};
    for (int i = 1; i < num_frames_to_render; ++i) {
        // Each subsequent frame is offset by the move vector.
        camera_origins_for_frames[i].x = camera_origins_for_frames[i-1].x + move_vector.x();
        camera_origins_for_frames[i].y = camera_origins_for_frames[i-1].y + move_vector.y();
        camera_origins_for_frames[i].z = camera_origins_for_frames[i-1].z + move_vector.z();
    }
}

// Moves camera origin based on input vector.
__global__ void update_camera_origin(vec3_t* new_origin, camera** d_camera) {
    (*d_camera)->origin.e[0] = new_origin->x;
    (*d_camera)->origin.e[1] = new_origin->y;
    (*d_camera)->origin.e[2] = new_origin->z;
}


void benchmark_frame(int argc, char **argv, int image_height, int image_width, int samples_per_pixel, int num_frames_to_render, int network_latency_in_us, bool has_stragglers) {
    std::cerr << "Benchmarking the rendering of " << num_frames_to_render << " " << image_width << "x" << image_height << " images with " << samples_per_pixel << " samples per pixel " << std::endl;

    int num_pixels = image_width*image_height;
    size_t frame_buffer_size = num_pixels*sizeof(vec3_t)*num_frames_to_render;

    // Initialize MPI.
    int num_procs, rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::cout << "Running on " << num_procs << " processes" << std::endl;

    // Create MPI Vec3 Type.
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[7] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Aint offsets[3];
    offsets[0] = offsetof(vec3_t, x);
    offsets[1] = offsetof(vec3_t, y);
    offsets[2] = offsetof(vec3_t, z);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &VEC3);
    MPI_Type_commit(&VEC3);

    // Allocate random state.
    curandState *d_rand_state;
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));

    // Allocate world state.
    int num_hitables = 22*22+1+3;
    hitable **d_list;
    hitable **d_world;
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    // Generate and broadcast random state from rank 0 to all ranks.
    if (rank == 0) {
        // Generate.
        rand_init<<<1,1>>>(d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Broadcast.
    MPI_Bcast((void *)d_rand_state2, sizeof(curandState), MPI_BYTE, 0, MPI_COMM_WORLD);

    // All ranks create world.
    create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // All ranks initialize rendering.
    int tx = 8;
    int ty = 8;
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state, 0, 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();

    // Now we begin rendering pipeline.
    // Goal is to mimic camera moving through the scene at a defined rate.
    if (rank == 0) {
        // Allocate frame buffer.
        vec3_t *frame_buffer;
        checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, frame_buffer_size));

        // Initialize camera origins for each frame.
        vec3_t* camera_origins_for_frames;
        checkCudaErrors(cudaMalloc((void **)&camera_origins_for_frames, num_frames_to_render * sizeof(vec3_t)));
        init_origins_for_frames<<<1,1>>>(camera_origins_for_frames, num_frames_to_render, d_camera);

        // Now we begin our work assignment.
        std::vector<int> free_gpus;
        for (int i = 0; i < num_procs; i++) {
            free_gpus.push_back(i);
        }
        std::vector<int> remaining_frames;
        for (int i = 0; i < num_frames_to_render; i++) {
            remaining_frames.push_back(i);
        }
        std::unordered_map<int, int> work_assignment;
        std::unordered_map<int, MPI_Request*> work_requests;
        while (remaining_frames.size() > 0) {
            // Assignment loop.
            for (auto frame: remaining_frames) {
                if (work_assignment.find(frame) == work_assignment.end()) {
                    // We have not assigned this frame yet.
                    // Pick a gpu and assign it to this frame.
                    if (free_gpus.size() == 0) {
                        // No more free gpus.
                        break;
                    }
                    // Grab the first free gpu and remove it.
                    int gpu = free_gpus.back();
                    work_assignment[frame] = gpu;
                    free_gpus.pop_back();
                    if (gpu != rank) {
                        // This is a remote GPU, send to be completed via MPI.
                        work_requests[frame] = new MPI_Request();
                        MPI_Send(&camera_origins_for_frames[frame], 1, VEC3, gpu, 0, MPI_COMM_WORLD);
                        MPI_Irecv(frame_buffer + frame * num_pixels, num_pixels, VEC3, gpu, 0, MPI_COMM_WORLD, work_requests[frame]);
                    }
                }
            }

            // Check for assignment completions.
            for (auto frame_gpu: work_assignment) {
                auto frame_id = frame_gpu.first;
                auto gpu = frame_gpu.second;
                if (frame_gpu.second == rank) {
                    // This is our work. We can now render it.

                    // Render the current frame and make sure it worked.
                    render_vec3_t<<<blocks, threads>>>(frame_buffer + frame_id * num_pixels, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state, 0, 0);
                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());

                    // Put GPU back in free list.
                    free_gpus.push_back(gpu);
                    // Remove frame from remaining frames.
                    remaining_frames.erase(std::remove(remaining_frames.begin(), remaining_frames.end(), frame_id), remaining_frames.end());
                    // Remove frame from work assignment.
                    work_assignment.erase(frame_id);
                } else {
                    // This is remote work, poll for completion.
                    int completed;
                    MPI_Test(work_requests[frame_id], &completed, MPI_STATUS_IGNORE);
                    if (completed) {
                        // This is done.
                        // Remove frame from remaining frames.
                        remaining_frames.erase(std::remove(remaining_frames.begin(), remaining_frames.end(), frame_id), remaining_frames.end());
                        // Put GPU back in free list.
                        free_gpus.push_back(gpu);
                        // Remove frame from work assignment.
                        work_assignment.erase(frame_id);
                    }
                }
            }

        }

    } else {
        // Allocate frame buffer.
        vec3_t *frame_buffer;
        checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, frame_buffer_size / num_frames_to_render));

        // Stores the camera origins from the root.
        vec3_t camera_origin = {0,0,0};
        while(true) {
            MPI_Status status;
            // Attempt to receive the camera origin for the frame.
            MPI_Recv(&camera_origin, 1, VEC3, 0, 0, MPI_COMM_WORLD, &status);
            // Update camera origin for the frame.
            update_camera_origin<<<1,1>>>(&camera_origin, d_camera);
            // Render the frame.
            render_vec3_t<<<blocks, threads>>>(frame_buffer, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state, 0, 0);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // Send the frame back to the root.
            MPI_Send(frame_buffer, num_pixels, VEC3, 0, 0, MPI_COMM_WORLD);
        }
    }

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds to generate " << num_frames_to_render << " frames.\n";
    std::cerr << "Average FPS: " << (double)num_frames_to_render / timer_seconds << "\n";

    MPI_Abort(MPI_COMM_WORLD, 0);
    MPI_Finalize();

    printf("Success \n");

    cudaDeviceReset();
}

// Benchmarks the throughput of a rendering type.
void benchmark_rendering(std::string rendering_strategy, int image_height, int image_width, int samples_per_pixel, int num_frames_to_render, int network_latency_in_us, bool has_stragglers, int argc, char **argv) {
    if (strcmp(rendering_strategy.c_str(), "singlenode") == 0) {
        benchmark_single(image_height, image_width, samples_per_pixel, num_frames_to_render);
    } else if (strcmp(rendering_strategy.c_str(), "tiled") == 0) {
        benchmark_tiled(image_height, image_width, samples_per_pixel, num_frames_to_render);
    } else if (strcmp(rendering_strategy.c_str(), "frame") == 0) {
        benchmark_frame(argc, argv, image_height, image_width, samples_per_pixel, num_frames_to_render, network_latency_in_us, has_stragglers);
    }
}

int main(int argc, char **argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        //std::cout << "Options:" << std::endl;
        //std::cout << "-h: see this help" << std::endl;
        //std::cout << "-t <int>: type, 0 = test, 1 = benchmark" << std::endl;
        //std::cout << "-r <rendering strategy>: singlenode/tiled/frame" << std::endl;
        //std::cout << "-v <int>: vertical height of image in pixels" << std::endl;
        //std::cout << "-w <int>: width of image in pixels" << std::endl;
        //std::cout << "-s <int>: number of samples per pixel" << std::endl;
        //std::cout << "-f <int>: number of frames to render" << std::endl;
        //std::cout << "-l <int>: emulated network latency" << std::endl;
        //std::cout << "-a <int>: stragglers, 0 = no, 1 = yes" << std::endl;
        return 0;
    }

    int image_height = find_int_arg(argc, argv, "-v", 800);
    int image_width = find_int_arg(argc, argv, "-w", 1200);
    int samples_per_pixel = find_int_arg(argc, argv, "-s", 10);

    int type = find_int_arg(argc, argv, "-t", 0);
    if (type == 0) {
        test_render(image_height, image_width, samples_per_pixel);
        return 0;
    }

    int num_frames_to_render = find_int_arg(argc, argv, "-f", 30);

    std::string rendering_strategy = find_string_option(argc, argv, "-r", std::string("singlenode"));
    int network_latency_in_us = find_int_arg(argc, argv, "0", 0);
    bool has_stragglers = find_int_arg(argc, argv, "-a", 0) == 1; 
    if (strcmp(rendering_strategy.c_str(), "singlenode") != 0  && strcmp(rendering_strategy.c_str(), "tiled") != 0 && strcmp(rendering_strategy.c_str(), "frame") != 0) {
        std::cerr << "Unknown rendering strategy: " << rendering_strategy << std::endl;
        return 1;
    }

    benchmark_rendering(rendering_strategy, image_height, image_width, samples_per_pixel, num_frames_to_render, network_latency_in_us, has_stragglers, argc, argv);
}