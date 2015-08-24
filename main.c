#include <CL/cl.h>
#include <stdio.h>
#include <string.h>

cl_platform_id platform = NULL;
cl_device_id device = NULL;
cl_context context = NULL;
cl_command_queue cmd_q = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_int err = CL_SUCCESS;

struct {
  const size_t dims;
  size_t global_sz[2];
  size_t local_sz[2];
  cl_mem buf;
  float data[(8 * 8)];
  const uint32_t data_count;
} work = { .dims = 2,
           .global_sz = { 8, 8 },
           .local_sz = { 2, 2 },
           .buf = NULL,
           .data_count = (8 * 8), };

const char *kernel_src =
    "__kernel void basic(__global float* data)\n"
    "{\n"
    "   size_t gid_x = get_global_id(0);\n"
    "   size_t gid_y = get_global_id(1);\n"
    "   size_t lid_x = get_local_id(0);\n"
    "   size_t lid_y = get_local_id(1);\n"
    "   size_t gid = gid_x + (gid_y * get_global_size(0));\n"
    "   size_t lid = lid_x + (lid_y * get_local_size(0));\n"
    "   float gx = gid_x * 10.0f;\n"
    "   float gy = gid_y;\n"
    "   float lx = lid_x/10.0f;\n"
    "   float ly = lid_y/100.0f;\n"
    "   float total = (gx+gy) + (lx+ly);\n"
    "   data[gid] = total;\n"
    "}\n";

void CL_CALLBACK
pfn_notify(const char *msg, const void *data0, size_t sz, void *data1) {
  fprintf(stderr, "an error occurred during opencl execution:\n%s\n", msg);
}

void init(void) {
  cl_uint num_platforms = 0;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (!num_platforms) {
    fprintf(stderr, "error: no opencl platforms found: %d\n", err);
    exit(1);
  }

  printf("found %d platform%s on system\n", num_platforms,
         (num_platforms > 1 ? "s" : ""));

  cl_platform_id *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (err) {
    fprintf(stderr, "error: failed to allocate platforms: %d\n", err);
    exit(0);
  }

  printf("using first platform with GPU\n");

  uint32_t platform_idx = 0, device_idx = 0;
  for (; platform_idx < num_platforms; ++platform_idx) {
    cl_uint platform_gpu_count = 0;
    err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 0, NULL,
                         &platform_gpu_count);
    if (err) {
      fprintf(stderr, "failed to query device ids: %d\n", err);
      exit(1);
    }

    if (!platform_gpu_count)
      continue;

    err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 1,
                         &device, NULL);
    if (err) {
      fprintf(stderr, "failed to allocate device: %d\n", err);
      exit(0);
    }

    break;
  }

  platform = platforms[platform_idx];
  free(platforms);
  platforms = NULL;

  if (device != NULL)
    printf("device found!\n");

  cl_context_properties ctxt_props[] = { CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platform, 0 };
  context = clCreateContext(ctxt_props, 1, &device, pfn_notify, NULL, &err);
  if (context == NULL || err) {
    fprintf(stderr, "failed to create context: %d\n", err);
    exit(1);
  }

  /*
  cl_command_queue_properties q_props[] = {
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE, 0
  };*/
  cmd_q = clCreateCommandQueue(context, device, 0, &err);
  if (cmd_q == NULL || err) {
    fprintf(stderr, "failed to create command queue: %d\n", err);
    exit(1);
  }

  size_t src_len = strlen(kernel_src);
  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src,
                                      &src_len, &err);

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err) {
    size_t len;
    fprintf(stderr, "failed to build program: %d\n", err);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                                &len);
    if (err) {
      fprintf(stderr, "failed to get build log size: %d", err);
      exit(1);
    }
    cl_char *log = (cl_char *)malloc(len);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log,
                                NULL);
    if (err) {
      fprintf(stderr, "failed to get build log: %d", err);
      exit(1);
    }
    fprintf(stderr, "%s\n", log);
    free(log);
  }

  kernel = clCreateKernel(program, "basic", &err);
  if (kernel == NULL || err) {
    fprintf(stderr, "failed to create kernel: %d\n", err);
    exit(1);
  }

  memset(work.data, 0, sizeof(cl_float) * work.data_count);
  work.buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            sizeof(cl_float) * work.data_count, NULL, &err);
  if (work.buf == NULL || err) {
    fprintf(stderr, "failed to create buffer: %d\n", err);
    exit(1);
  }
}

void exec(void) {
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&work.buf);
  if (err) {
    fprintf(stderr, "failed to set kernel argument: %d\n", err);
    exit(1);
  }

  err = clEnqueueNDRangeKernel(cmd_q, kernel, work.dims, NULL, work.global_sz,
                               work.local_sz, 0, NULL, NULL);
  if (err) {
    fprintf(stderr, "failed to enqueue kernel execution: %d\n", err);
    exit(1);
  }

  err = clEnqueueReadBuffer(cmd_q, work.buf, CL_BLOCKING, 0,
                            work.data_count * sizeof(cl_float), work.data, 0,
                            NULL, NULL);
  if (err) {
    fprintf(stderr, "failed to read buffer: %d\n", err);
    exit(0);
  }

  err = clFinish(cmd_q);
  if (err) {
    fprintf(stderr, "failed to submit and wait for all queue commands %d\n",
            err);
    exit(1);
  }

  printf("\nwork-item thread execution layout:\n");
  printf("[xy.xy] = global-id.local-id\n");
  printf("global work size = 8 x 8\n"
         "local work size = 2 x 2\n\n");
  uint32_t id_x, id_y, i;
  for (id_y = 0; id_y < work.global_sz[1]; ++id_y) {
    for (id_x = 0; id_x < work.global_sz[0]; ++id_x) {
      i = id_x + (id_y * work.global_sz[0]);

      if ((i % 2) == 0)
        printf("\t");

      printf("[%s%.2f]", (work.data[i] < 8 ? "0" : ""), work.data[i]);
    }
    printf("\n%s", (id_y & 1) ? "\n" : "");
  }
}

void teardown(void) {
  if (work.buf) {
    err = clReleaseMemObject(work.buf);
    if (err) {
      fprintf(stderr, "failed to release buffer. aborting: %d\n", err);
      abort();
    }
  }

  if (kernel != NULL) {
    err = clReleaseKernel(kernel);
    if (err) {
      fprintf(stderr, "failed to release kernel. aborting: %d\n", err);
      abort();
    }
  }

  if (program != NULL) {
    err = clReleaseProgram(program);
    if (err) {
      fprintf(stderr, "failed to release program. aborting: %d\n", err);
      abort();
    }
  }

  if (cmd_q != NULL) {
    err = clReleaseCommandQueue(cmd_q);
    if (err) {
      fprintf(stderr, "failed to release cmd queue. aborting: %d\n", err);
      abort();
    }
  }

  if (context != NULL) {
    err = clReleaseContext(context);
    if (err) {
      fprintf(stderr, "failed to release context. aborting: %d\n", err);
      abort();
    }
  }

  if (device) {
    err = clReleaseDevice(device);
    if (err) {
      fprintf(stderr, "failed to release device. aborting: %d\n", err);
      abort();
    }
  }
}

int main(int argc, const char **argv) {
  printf("hello ocl-test\n");
  init();
  exec();
  teardown();
  return 0;
}
