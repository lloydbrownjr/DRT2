#include <float.h>
#include <iostream>
#include <map>
#include <string>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <curand_kernel.h>
#include <mpi.h>

#include "../common/options.h"
#include "../common/ray.h"
#include "../common/vec3.h"
#include "camera.h"
#include "cuda_errors.h"
#include "hitable_list.h"
#include "material.h"
#include "sphere.h"

#define num_hitables (22*22 + 1 + 3)

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
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init_tiled(int x_range, int y_range, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= x_range || j >= y_range) {
        return;
    }

    int pixel_index = j * x_range + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render_tiled(vec3 *frame_buffer, int image_width, int image_height, int x_start, int y_start, int x_range, int y_range,
        int number_samples, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= x_range || j >= y_range) {
        return;
    }
    int local_index = j * x_range + i;
    curandState local_rand_state = rand_state[local_index];
    int pixel_x = x_start + i;
    int pixel_y = y_start + j;
    int pixel_index = pixel_y * image_width + pixel_x;
    vec3 col(0, 0, 0);
    for(int sample = 0; sample < number_samples; sample++) {
        float u = float(pixel_x + curand_uniform(&local_rand_state)) / float(image_width);
        float v = float(pixel_y + curand_uniform(&local_rand_state)) / float(image_height);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[local_index] = local_rand_state;
    col /= float(number_samples);
    frame_buffer[pixel_index] = col.getsqrt();
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if (choose_mat < 0.95f) {
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
        *d_world  = new hitable_list(d_list, num_hitables);

        vec3 lookfrom(13, 2, 3);
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

// Moves the camera's origin to create a new scene
__global__ void move_cam(camera **d_camera, int steps = 1) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->origin += steps * vec3(0, 0, -0.1);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < num_hitables; i++) {
            delete ((sphere *)d_list[i])->mat_ptr;
            delete d_list[i];
        }
        delete *d_world;
        delete *d_camera;
    }
}

// Writes the image to a ppm file.
void write_frame_buffer(vec3 *frame_buffer, int nx, int ny, int max_x, int max_y) {
    FILE *f = fopen("output.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", max_x, max_y);
    for(int j = 0; j < max_y; j++) {
        for(int i = 0; i < max_x; i++) {
            vec3 col = frame_buffer[j*max_x + i];
            int ir = int(255.99 * col[0]);
            int ig = int(255.99 * col[1]);
            int ib = int(255.99 * col[2]);
            fprintf(f, "%d %d %d ", ir, ig, ib);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Writes the image to a ppm file.
void write_frame_buffer_filename(vec3 *frame_buffer, int nx, int ny, int max_x, int max_y, std::string filename) {
    FILE *f = fopen(filename.c_str(), "w");
    fprintf(f, "P3\n%d %d\n255\n", max_x, max_y);
    for(int j = 0; j < max_y; j++) {
        for(int i = 0; i < max_x; i++) {
            vec3 col = frame_buffer[j*max_x + i];
            int ir = int(255.99 * col[0]);
            int ig = int(255.99 * col[1]);
            int ib = int(255.99 * col[2]);
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
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    // render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // render<<<blocks, threads>>>(frame_buffer, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
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
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
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
    render_init_tiled<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    for (int i = 0; i < num_frames_to_render; i++) {
        // Render the current frame and make sure it worked.
        render_tiled<<<blocks, threads>>>(frame_buffer, image_width, image_height, 0, 0, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
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

void benchmark_tiled(int image_height, int image_width, int samples_per_pixel, int num_frames_to_render, int num_gpus = -1) {
    int tx = 8;
    int ty = 8;

    std::cerr << "Benchmarking the rendering of " << image_width << "x" << image_height << " images with " << samples_per_pixel << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;

    // allocate frame_buffer
    vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, num_pixels * sizeof(vec3)));

    int available_gpus;
    checkCudaErrors(cudaGetDeviceCount(&available_gpus));
    if (num_gpus > available_gpus) {
        std::cerr << "requeted more than available GPUs, capping." << std::endl;
        num_gpus = available_gpus;
    } else if (num_gpus == -1) {
        num_gpus = available_gpus;
    }

    int num_streams = num_gpus;
    
    int per_gpu_width = image_width;
    int per_gpu_height = image_height / num_gpus;
    int num_pixels_per_gpu = per_gpu_width * per_gpu_height;

    dim3 blocks(per_gpu_width / tx + 1, per_gpu_height / ty + 1);
    dim3 threads(tx, ty);

    // allocate random state
    using std::vector;
    vector<curandState *> d_rand_state(num_gpus);
    vector<curandState *> d_rand_state2(num_gpus);
    vector<cudaStream_t> streams(num_streams);
    vector<hitable **> d_list(num_gpus);
    vector<hitable **> d_world(num_gpus);
    vector<camera **> d_camera(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        checkCudaErrors(cudaSetDevice(gpu_id));
        checkCudaErrors(cudaMalloc((void **) &d_rand_state[gpu_id], num_pixels_per_gpu * sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void **) &d_rand_state2[gpu_id], 1 * sizeof(curandState)));
        checkCudaErrors(cudaStreamCreate(&streams[gpu_id]));
        rand_init<<<1, 1>>>(d_rand_state2[gpu_id]);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMalloc((void **) &d_list[gpu_id], num_hitables * sizeof(hitable *)));
        checkCudaErrors(cudaMalloc((void **) &d_world[gpu_id], 1 * sizeof(hitable *)));
        checkCudaErrors(cudaMalloc((void **) &d_camera[gpu_id], 1 * sizeof(camera *)));
        cudaDeviceSynchronize();
        create_world<<<1, 1>>>(d_list[gpu_id], d_world[gpu_id], d_camera[gpu_id], image_width, image_height, d_rand_state2[gpu_id]);
        checkCudaErrors(cudaGetLastError());
    }

    clock_t start, stop;
    start = clock();
    // Render our buffer
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        checkCudaErrors(cudaSetDevice(gpu_id));
        render_init_tiled<<<blocks, threads, 0, streams[gpu_id]>>>(per_gpu_width, per_gpu_height, d_rand_state[gpu_id]);
        // render_init<<<blocks, threads, 0, streams[gpu_id]>>>(image_width, image_height, d_rand_state[gpu_id], gpu_id, num_gpus);
    }
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
        checkCudaErrors(cudaStreamSynchronize(streams[stream_id]));
    }
    for (int i = 0; i < num_frames_to_render; i++) {
        for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            checkCudaErrors(cudaSetDevice(gpu_id));
            int x_start = 0, y_start = image_height * gpu_id / num_gpus;
            render_tiled<<<blocks, threads, 0, streams[gpu_id]>>>(frame_buffer, image_width, image_height, x_start, y_start, per_gpu_width, per_gpu_height,
                samples_per_pixel, d_camera[gpu_id], d_world[gpu_id], d_rand_state[gpu_id]);
            checkCudaErrors(cudaGetLastError());
            move_cam<<<1, 1, 0, streams[gpu_id]>>>(d_camera[gpu_id]);
            checkCudaErrors(cudaGetLastError());
        }
        for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            checkCudaErrors(cudaStreamSynchronize(streams[gpu_id]));
        }
    }

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds to generate " << num_frames_to_render << " frames.\n";
    std::cerr << "Average FPS: " << (double)num_frames_to_render / timer_seconds << "\n";

    // Output frame_buffer as Image
    write_frame_buffer(frame_buffer, image_width, image_height, image_width, image_height);

    // clean up
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        checkCudaErrors(cudaSetDevice(gpu_id));
        free_world<<<1, 1>>>(d_list[gpu_id], d_world[gpu_id], d_camera[gpu_id]);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_camera[gpu_id]));
        checkCudaErrors(cudaFree(d_world[gpu_id]));
        checkCudaErrors(cudaFree(d_list[gpu_id]));
        checkCudaErrors(cudaStreamDestroy(streams[gpu_id]));
        checkCudaErrors(cudaFree(d_rand_state2[gpu_id]));
        checkCudaErrors(cudaFree(d_rand_state[gpu_id]));
    }
    checkCudaErrors(cudaFree(frame_buffer));

    cudaDeviceReset();
}

__global__ void update_camera_origin(vec3 new_origin, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->origin.e[0] = new_origin.x();
        (*d_camera)->origin.e[1] = new_origin.y();
        (*d_camera)->origin.e[2] = new_origin.z();
    }
}

void init_camera_origins_for_frames(vec3 *camera_origins_for_frames, int num_frames_to_render, vec3& current_origin) {
    auto move_vector = vec3(0, 0, -0.1);
    camera_origins_for_frames[0] = vec3(current_origin);
    for (int i = 1; i < num_frames_to_render; i++) {
        // Each subsequent frame is offset by the move vector.
        camera_origins_for_frames[i].e[0] = camera_origins_for_frames[i-1].x();
        camera_origins_for_frames[i].e[1] = camera_origins_for_frames[i-1].y();
        camera_origins_for_frames[i].e[2] = camera_origins_for_frames[i-1].z();
    }
}

enum message_tag {
    KILL_SIGNAL,
    CAMERA_ORIGIN_INFO,
    FRAME_BUFFER_BACK,
};

void benchmark_frame(int argc, char **argv, int image_height, int image_width, int samples_per_pixel, int num_frames_to_render) {
    int num_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (num_procs < 2) {
        std::cerr << "This program requires at least 2 processes.\n";
        exit(1);
    }

    if (rank == 0) {
        std::cerr << "Benchmarking the rendering of " << num_frames_to_render << " " << image_width << "x" << image_height << " images with " << samples_per_pixel << " samples per pixel " << std::endl;
        std::cerr << "num_gpus: " << num_procs - 1 << std::endl;
    }

    MPI_Datatype MPI_Vec3;
    int nitems = 3;
    int blocklengths[3] = { 1, 1, 1 };
    MPI_Aint offsets[3];
    offsets[0] = offsetof(vec3, e[0]);
    offsets[1] = offsetof(vec3, e[1]);
    offsets[2] = offsetof(vec3, e[2]);
    MPI_Datatype types[3] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT };
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_Vec3);
    MPI_Type_commit(&MPI_Vec3);

    if (rank > 0) {
        checkCudaErrors(cudaSetDevice(rank - 1));
    } else {
        checkCudaErrors(cudaSetDevice(0));
    }

    int num_pixels = image_width * image_height;

    // every machine init random and world
    curandState *d_rand_state;
    curandState *d_rand_state2;
    hitable **d_list;
    hitable **d_world;
    camera **d_camera;
    if (rank > 0) {
        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
        rand_init<<<1, 1>>>(d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
        checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
        checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
        create_world<<<1, 1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
        checkCudaErrors(cudaGetLastError());
    }

    vec3 *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, num_pixels * sizeof(vec3)));

    vec3 *camera_origins_for_frames;
    if (rank == 0) {
        //checkCudaErrors(cudaMallocManaged((void **)&camera_origins_for_frames, num_frames_to_render * sizeof(vec3)));
        camera_origins_for_frames = new vec3[num_frames_to_render];
        auto origin = vec3(13, 2, 3);
        init_camera_origins_for_frames(camera_origins_for_frames, num_frames_to_render, origin);
    }

    clock_t start, stop;
    if (rank == 0) {
        start = clock();
    }

    int tx = 8;
    int ty = 8;

    dim3 blocks(image_width/tx + 1, image_height/ty + 1);
    dim3 threads(tx, ty);
    if (rank > 0) {
        render_init_tiled<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    }
    checkCudaErrors(cudaGetLastError());

    if (rank == 0) {
        std::unordered_set<int> free_gpus;
        std::vector<int> remaining_frames;
        for (int i = 1; i < num_procs; i++) {
            free_gpus.insert(i);
        }
        for (int i = 0; i < num_frames_to_render; i++) {
            remaining_frames.push_back(i);
        }

        std::map<int, int> work_assignment; // gpu_id -> frame_id, ordered on gpu_id.
        // only those in progress are in the work_assignment map.
        // std::unordered_map<int, MPI_Request> work_requests; // gpu_id -> Request
        MPI_Request work_requests[num_procs-1];
        for (int i = 0; i < num_procs-1; i++) {
            work_requests[i] = MPI_REQUEST_NULL;
        }
        MPI_Status status;
        int index;


        vec3 **recv_frame_buffer = new vec3 *[num_procs];
        for (int i = 1; i < num_procs; i++) {
            recv_frame_buffer[i] = new vec3[num_pixels];
        }

        MPI_Request ignore;

        while (true) {
            if (remaining_frames.size() == 0 && free_gpus.size() == num_procs - 1) {
                // no work to be done and no work to be waited so we kill everyone and get out
                int signal = 267;
                for (int other = 1; other < num_procs; other++) {
                    MPI_Isend(&signal, 1, MPI_INT, other, KILL_SIGNAL, MPI_COMM_WORLD, &ignore);
                }
                for (int i = 1; i < num_procs; i++) {
                    delete[] recv_frame_buffer[i];
                }
                delete[] recv_frame_buffer;
                break;
            }

            if (remaining_frames.size() == 0 || free_gpus.size() == 0) {
                MPI_Waitany(num_procs-1, work_requests, &index, &status);
                int gpu_id = index + 1;
                free_gpus.insert(gpu_id);
                // Remove the gpu from the frame to gpu map.
                int frame_id = work_assignment[gpu_id];
                work_assignment.erase(gpu_id);
                work_requests[gpu_id - 1] = MPI_REQUEST_NULL;
                if (remaining_frames.size() == 0) {
                    continue;
                }
            }

            auto frame_iter = remaining_frames.begin();
            int frame = *frame_iter;
            remaining_frames.erase(frame_iter);
            auto gpu_it = free_gpus.begin();
            int gpu = *gpu_it;
            free_gpus.erase(gpu_it);
            work_assignment[gpu] = frame;
            // pop a frame
            // pop a gpu
            vec3 camera_origin = camera_origins_for_frames[frame];
            MPI_Isend(camera_origin.e, 1, MPI_Vec3, gpu, CAMERA_ORIGIN_INFO, MPI_COMM_WORLD, &ignore);
            // send camera to remote gpu
            MPI_Irecv(recv_frame_buffer[gpu], num_pixels, MPI_Vec3, gpu, FRAME_BUFFER_BACK, MPI_COMM_WORLD, &work_requests[gpu-1]);
            // recv into corresponding frame_buffer
        }

    } else {
        vec3 camera_origin;
        int signal;
        MPI_Request requests[2];
        MPI_Irecv(&signal, 1, MPI_INT, 0, KILL_SIGNAL, MPI_COMM_WORLD, &requests[0]);
        int which_request;
        MPI_Status status;
        while (true) {
            MPI_Irecv(camera_origin.e, 1, MPI_Vec3, 0, CAMERA_ORIGIN_INFO, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitany(2, requests, &which_request, &status);
            if (which_request == 0) {
                // signal received
                if (signal == 267) {
                    break;
                }
            }
            // get camera origin info
            update_camera_origin<<<1, 1>>>(camera_origin, d_camera);
            // update d_camera
            render_tiled<<<blocks, threads>>>(frame_buffer, image_width, image_height, 0, 0, image_width, image_height,
                samples_per_pixel, d_camera, d_world, d_rand_state);
            // render
            // send back frame
            checkCudaErrors(cudaDeviceSynchronize());
            MPI_Send(frame_buffer, num_pixels, MPI_Vec3, 0, FRAME_BUFFER_BACK, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds to generate " << num_frames_to_render << " frames.\n";
        std::cerr << "Average FPS: " << (double)num_frames_to_render / timer_seconds << "\n";
    }

    // if (rank == 0) {
    //     checkCudaErrors(cudaFree(camera_origins_for_frames));
    // }
    if (rank > 0) {
        checkCudaErrors(cudaFree(frame_buffer));
        free_world<<<1, 1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaFree(d_camera));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(d_list));
        checkCudaErrors(cudaFree(d_rand_state2));
        checkCudaErrors(cudaFree(d_rand_state));
    }

    MPI_Finalize();

}

// Benchmarks the throughput of a rendering type.
void benchmark_rendering(int argc, char **argv, std::string rendering_strategy, int image_height, int image_width, int samples_per_pixel, int num_frames_to_render, int requested_gpus) {
    if (strcmp(rendering_strategy.c_str(), "singlenode") == 0) {
        benchmark_single(image_height, image_width, samples_per_pixel, num_frames_to_render);
    } else if (strcmp(rendering_strategy.c_str(), "tiled") == 0) {
        benchmark_tiled(image_height, image_width, samples_per_pixel, num_frames_to_render, requested_gpus);
    } else if (strcmp(rendering_strategy.c_str(), "frame") == 0) {
        benchmark_frame(argc, argv, image_height, image_width, samples_per_pixel, num_frames_to_render);
    }
}

int main(int argc, char **argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-t <int>: type, 0 = test, 1 = benchmark" << std::endl;
        std::cout << "-r <rendering strategy>: singlenode/tiled/frame" << std::endl;
        std::cout << "-v <int>: vertical height of image in pixels" << std::endl;
        std::cout << "-w <int>: width of image in pixels" << std::endl;
        std::cout << "-s <int>: number of samples per pixel" << std::endl;
        std::cout << "-f <int>: number of frames to render" << std::endl;
        std::cout << "-g <int>: number of gpus to use" << std::endl;
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
    int requested_gpus = find_int_arg(argc, argv, "-g", -1);

    std::string rendering_strategy = find_string_option(argc, argv, "-r", std::string("singlenode"));
    if (strcmp(rendering_strategy.c_str(), "singlenode") != 0  && strcmp(rendering_strategy.c_str(), "tiled") != 0 && strcmp(rendering_strategy.c_str(), "frame") != 0) {
        std::cerr << "Unknown rendering strategy: " << rendering_strategy << std::endl;
        return 1;
    }

    benchmark_rendering(argc, argv, rendering_strategy, image_height, image_width, samples_per_pixel, num_frames_to_render, requested_gpus);
}