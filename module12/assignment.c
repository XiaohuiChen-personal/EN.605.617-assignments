/* Closest-word search.
 * Xiaohui Chen, EN.605.617.81, Fall 2026.
 *
 * The table is one buffer. The search slice is a sub-buffer.
 * The query is constant memory. Each work-group reduces one
 * distance in local memory, then a second kernel keeps the minimum.
 */
#include "cl_check.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_COUNT 4096
#define DEFAULT_START 0
#define DEFAULT_DIM 256
#define DEFAULT_WG 64
#define DEFAULT_VEC 4
#define DEFAULT_TARGET 0
#define DEFAULT_MAP 0
#define MAX_DIM 1024
#define MAX_WORDS 262144
#define MAX_TABLE_BYTES (256u * 1024u * 1024u)
#define ROW_STEP 131u
#define COL_STEP 17u
#define ROW_SPAN 10000u
#define HASH_XOR 2654435761u
#define HASH_MUL 2246822519u
#define QUERY_BUMP 0.05f
#define DIST_TOL 1.0e-5f
#define GAP_MIN 1.0e-3f
#define WARMUP 2
#define TIMED 5
#define BUILD_OPTS "-cl-std=CL1.2 -cl-kernel-arg-info"

struct job {
    int count;
    int start;
    int dim;
    int wg;
    int vec;
    int target;
    int mapped;
};

struct dev {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_ulong const_max;
    cl_ulong local_max;
    size_t align;
    size_t max_wg;
    char name[256];
    char opencl_c[128];
    char platform_name[128];
    char platform_ver[128];
};

struct bufs {
    float *table;
    float *query;
    cl_mem parent;
    cl_mem slice;
    cl_mem query_mem;
    cl_mem dist;
    cl_mem out_index;
    cl_mem out_value;
    size_t offset;
};

struct pair {
    int index;
    float dist;
    float second;
};

struct answer {
    int index;
    float dist;
    double dist_ms;
    double min_ms;
};

static void die(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);

    if (!p)
        die("out of memory");
    return p;
}

static int is_pow2(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

static int vec_ok(int v)
{
    return v == 1 || v == 4 || v == 8;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [count start dim wg vec target map]\n",
        argv0);
    fprintf(stderr, "defaults: %d %d %d %d %d %d %d\n",
        DEFAULT_COUNT, DEFAULT_START, DEFAULT_DIM, DEFAULT_WG,
        DEFAULT_VEC, DEFAULT_TARGET, DEFAULT_MAP);
    fprintf(stderr, "see README.md for what each number means\n");
    exit(EXIT_FAILURE);
}

static long need_long(const char *s, const char *label)
{
    char *end = NULL;
    long v;

    errno = 0;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0) {
        fprintf(stderr, "bad %s: %s\n", label, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static int need_int(const char *s, const char *label,
                    long lo, long hi)
{
    long v = need_long(s, label);

    if (v < lo || v > hi) {
        fprintf(stderr, "%s must be from %ld to %ld\n",
            label, lo, hi);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static void check_job(const struct job *job)
{
    size_t words;
    size_t bytes;

    if (!vec_ok(job->vec))
        die("vec must be 1, 4, or 8");
    if (job->dim % job->vec != 0)
        die("dim must be a multiple of vec");
    if (!is_pow2(job->wg))
        die("wg must be a power of two");
    if (job->target < 0 || job->target >= job->count)
        die("target must be a slice index");
    if (job->mapped != 0 && job->mapped != 1)
        die("map must be 0 or 1");
    words = (size_t)job->start + (size_t)job->count;
    if (words > (size_t)MAX_WORDS)
        die("word count is out of range");
    bytes = words * (size_t)job->dim * sizeof(float);
    if (bytes > (size_t)MAX_TABLE_BYTES)
        die("table is larger than 256 MB");
}

static void parse_job(int argc, char **argv, struct job *job)
{
    job->count = DEFAULT_COUNT;
    job->start = DEFAULT_START;
    job->dim = DEFAULT_DIM;
    job->wg = DEFAULT_WG;
    job->vec = DEFAULT_VEC;
    job->target = DEFAULT_TARGET;
    job->mapped = DEFAULT_MAP;
    if (argc > 8)
        usage(argv[0]);
    if (argc > 1)
        job->count = need_int(argv[1], "count", 1, MAX_WORDS);
    if (argc > 2)
        job->start = need_int(argv[2], "start", 0, MAX_WORDS);
    if (argc > 3)
        job->dim = need_int(argv[3], "dim", 1, MAX_DIM);
    if (argc > 4)
        job->wg = need_int(argv[4], "wg", 1, 1024);
    if (argc > 5)
        job->vec = need_int(argv[5], "vec", 1, 8);
    if (argc > 6)
        job->target = need_int(argv[6], "target", 0, MAX_WORDS);
    if (argc > 7)
        job->mapped = need_int(argv[7], "map", 0, 1);
    check_job(job);
}

static float word_value(int word, int comp)
{
    unsigned u;

    /* Fold the word id in so two rows are not the same list. */
    u = (unsigned)word * ROW_STEP + (unsigned)comp * COL_STEP + 1u;
    u ^= (unsigned)word * HASH_XOR;
    u *= HASH_MUL;
    return (float)(u % ROW_SPAN) / (float)ROW_SPAN - 0.5f;
}

static void fill_table(float *table, int words, int dim)
{
    int w;
    int k;

    for (w = 0; w < words; w++) {
        for (k = 0; k < dim; k++) {
            size_t i = (size_t)w * (size_t)dim + (size_t)k;
            table[i] = word_value(w, k);
        }
    }
}

static void fill_query(float *query, const float *table,
                       const struct job *job)
{
    size_t row;
    int k;
    int abs_word = job->start + job->target;

    row = (size_t)abs_word * (size_t)job->dim;
    for (k = 0; k < job->dim; k++)
        query[k] = table[row + (size_t)k];
    query[0] += QUERY_BUMP;
}

static float row_dist(const float *row, const float *query, int dim)
{
    int k;
    float acc = 0.0f;

    for (k = 0; k < dim; k++) {
        float diff = row[k] - query[k];
        acc += diff * diff;
    }
    return acc;
}

static void cpu_closest(const struct job *job, const float *table,
                        const float *query, struct pair *out)
{
    int i;

    out->index = -1;
    out->dist = INFINITY;
    out->second = INFINITY;
    for (i = 0; i < job->count; i++) {
        size_t row = (size_t)(job->start + i) * (size_t)job->dim;
        float d = row_dist(table + row, query, job->dim);
        if (d < out->dist) {
            out->second = out->dist;
            out->dist = d;
            out->index = i;
        } else if (d < out->second) {
            out->second = d;
        }
    }
}

static void trim_end(char *s)
{
    size_t n = strlen(s);

    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\n'
                     || s[n - 1] == '\r')) {
        s[n - 1] = '\0';
        n--;
    }
}

static void query_device_str(cl_device_id dev, cl_device_info key,
                             char *buf, size_t n)
{
    if (n == 0)
        die("empty string buffer");
    buf[0] = '\0';
    CL_CHECK(clGetDeviceInfo(dev, key, n, buf, NULL));
    buf[n - 1] = '\0';
}

static void query_platform_str(cl_platform_id plat,
                               cl_platform_info key,
                               char *buf, size_t n)
{
    if (n == 0)
        die("empty string buffer");
    buf[0] = '\0';
    CL_CHECK(clGetPlatformInfo(plat, key, n, buf, NULL));
    buf[n - 1] = '\0';
}

static cl_platform_id *all_platforms(cl_uint *nout)
{
    cl_uint n = 0;
    cl_platform_id *ps;

    CL_CHECK(clGetPlatformIDs(0, NULL, &n));
    if (n == 0)
        die("no OpenCL platform");
    ps = calloc(n, sizeof *ps);
    if (!ps)
        die("out of memory");
    CL_CHECK(clGetPlatformIDs(n, ps, NULL));
    *nout = n;
    return ps;
}

static void read_device(struct dev *dev)
{
    cl_uint bits = 0;

    query_platform_str(dev->platform, CL_PLATFORM_NAME,
        dev->platform_name, sizeof dev->platform_name);
    query_platform_str(dev->platform, CL_PLATFORM_VERSION,
        dev->platform_ver, sizeof dev->platform_ver);
    query_device_str(dev->device, CL_DEVICE_NAME,
        dev->name, sizeof dev->name);
    query_device_str(dev->device, CL_DEVICE_OPENCL_C_VERSION,
        dev->opencl_c, sizeof dev->opencl_c);
    trim_end(dev->platform_name);
    trim_end(dev->platform_ver);
    trim_end(dev->name);
    trim_end(dev->opencl_c);
    CL_CHECK(clGetDeviceInfo(dev->device,
        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
        sizeof dev->const_max, &dev->const_max, NULL));
    CL_CHECK(clGetDeviceInfo(dev->device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof dev->local_max, &dev->local_max, NULL));
    CL_CHECK(clGetDeviceInfo(dev->device,
        CL_DEVICE_MEM_BASE_ADDR_ALIGN,
        sizeof bits, &bits, NULL));
    CL_CHECK(clGetDeviceInfo(dev->device,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof dev->max_wg, &dev->max_wg, NULL));
    dev->align = (size_t)(bits / 8);
    if (dev->align == 0)
        die("device reported zero alignment");
}

static void make_runtime(struct dev *dev)
{
    cl_int err;
    cl_context_properties cprops[3];
    cl_queue_properties qprops[3];

    cprops[0] = CL_CONTEXT_PLATFORM;
    cprops[1] = (cl_context_properties)dev->platform;
    cprops[2] = 0;
    dev->context = clCreateContext(
        cprops, 1, &dev->device, NULL, NULL, &err);
    CL_CHECK_ERR(err, "clCreateContext");
    qprops[0] = CL_QUEUE_PROPERTIES;
    qprops[1] = CL_QUEUE_PROFILING_ENABLE;
    qprops[2] = 0;
    dev->queue = clCreateCommandQueueWithProperties(
        dev->context, dev->device, qprops, &err);
    CL_CHECK_ERR(err, "clCreateCommandQueueWithProperties");
}

static void open_gpu(struct dev *dev)
{
    cl_platform_id *ps;
    cl_uint n = 0;
    cl_uint i;
    int found = 0;

    ps = all_platforms(&n);
    for (i = 0; i < n; i++) {
        cl_int err = clGetDeviceIDs(
            ps[i], CL_DEVICE_TYPE_GPU, 1, &dev->device, NULL);
        if (err == CL_SUCCESS) {
            dev->platform = ps[i];
            found = 1;
            break;
        }
    }
    free(ps);
    if (!found)
        die("no GPU device");
    read_device(dev);
    make_runtime(dev);
}

static void print_device(const struct dev *dev)
{
    printf("platform: %s\n", dev->platform_name);
    printf("platform version: %s\n", dev->platform_ver);
    printf("device: %s\n", dev->name);
    printf("opencl c: %s\n", dev->opencl_c);
    printf("align: %zu bytes\n", dev->align);
    printf("constant max: %llu bytes\n",
        (unsigned long long)dev->const_max);
    printf("local max: %llu bytes\n",
        (unsigned long long)dev->local_max);
    printf("device work-group max: %zu\n", dev->max_wg);
}

static void check_device_fit(const struct dev *dev,
                             const struct job *job)
{
    size_t origin;
    size_t qbytes;
    size_t local_need;

    origin = (size_t)job->start * (size_t)job->dim * sizeof(float);
    if (origin % dev->align != 0) {
        fprintf(stderr,
            "origin %zu is not a multiple of align %zu\n",
            origin, dev->align);
        exit(EXIT_FAILURE);
    }
    qbytes = (size_t)job->dim * sizeof(float);
    if ((cl_ulong)qbytes > dev->const_max)
        die("query does not fit in constant memory");
    local_need = (size_t)job->wg * (sizeof(float) + sizeof(cl_int));
    if ((cl_ulong)local_need > dev->local_max)
        die("work-group does not fit in local memory");
    if ((size_t)job->wg > dev->max_wg) {
        fprintf(stderr, "wg %d exceeds device max %zu\n",
            job->wg, dev->max_wg);
        exit(EXIT_FAILURE);
    }
}

static int beside_exe(const char *argv0, char *out, size_t n)
{
    const char *slash;
    size_t dir;
    const char *leaf = "words.cl";

    slash = strrchr(argv0, '/');
    if (!slash)
        return 0;
    dir = (size_t)(slash - argv0);
    if (dir + 1 + strlen(leaf) + 1 > n)
        return 0;
    memcpy(out, argv0, dir);
    out[dir] = '/';
    memcpy(out + dir + 1, leaf, strlen(leaf) + 1);
    return 1;
}

static char *slurp(const char *path, size_t *nout)
{
    FILE *fp;
    char *buf;
    long n;

    fp = fopen(path, "rb");
    if (!fp)
        return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    n = ftell(fp);
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(fp);
        die("out of memory");
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fclose(fp);
        free(buf);
        return NULL;
    }
    fclose(fp);
    buf[n] = '\0';
    *nout = (size_t)n;
    return buf;
}

static char *load_source(const char *argv0, size_t *nout)
{
    char path[512];
    char *text;

    text = slurp("words.cl", nout);
    if (text) {
        printf("kernel file: words.cl\n");
        return text;
    }
    if (!beside_exe(argv0, path, sizeof path))
        die("cannot open words.cl");
    text = slurp(path, nout);
    if (!text)
        die("cannot open words.cl");
    printf("kernel file: %s\n", path);
    return text;
}

static int blank_log(const char *log)
{
    size_t i;

    for (i = 0; log[i] != '\0'; i++) {
        if (log[i] != ' ' && log[i] != '\n' && log[i] != '\t'
            && log[i] != '\r')
            return 0;
    }
    return 1;
}

static void print_build_log(cl_program prog, cl_device_id device)
{
    size_t n = 0;
    char *log;

    CL_CHECK(clGetProgramBuildInfo(
        prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n));
    if (n == 0) {
        printf("build log: (empty)\n");
        return;
    }
    log = xmalloc(n + 1);
    CL_CHECK(clGetProgramBuildInfo(
        prog, device, CL_PROGRAM_BUILD_LOG, n, log, NULL));
    log[n] = '\0';
    if (blank_log(log))
        printf("build log: (empty)\n");
    else
        printf("build log:\n%s\n", log);
    free(log);
}

static cl_program build_source(struct dev *dev, const char *src,
                               size_t n)
{
    cl_int err;
    cl_program prog;

    prog = clCreateProgramWithSource(
        dev->context, 1, &src, &n, &err);
    CL_CHECK_ERR(err, "clCreateProgramWithSource");
    err = clBuildProgram(
        prog, 1, &dev->device, BUILD_OPTS, NULL, NULL);
    print_build_log(prog, dev->device);
    CL_CHECK_ERR(err, "clBuildProgram");
    printf("build options: %s\n", BUILD_OPTS);
    return prog;
}

static void print_kernels(cl_program prog)
{
    cl_uint n = 0;
    cl_uint i;
    cl_kernel *ks;

    CL_CHECK(clCreateKernelsInProgram(prog, 0, NULL, &n));
    if (n == 0)
        die("program has no kernels");
    ks = calloc(n, sizeof *ks);
    if (!ks)
        die("out of memory");
    CL_CHECK(clCreateKernelsInProgram(prog, n, ks, NULL));
    printf("kernels: %u\n", n);
    for (i = 0; i < n; i++) {
        char name[64];

        memset(name, 0, sizeof name);
        CL_CHECK(clGetKernelInfo(ks[i], CL_KERNEL_FUNCTION_NAME,
            sizeof name, name, NULL));
        printf("  %s\n", name);
        CL_CHECK(clReleaseKernel(ks[i]));
    }
    free(ks);
}

static const char *kernel_for(const struct job *job)
{
    if (job->vec == 8)
        return "distance_v8";
    if (job->vec == 4)
        return "distance_v4";
    return "distance_v1";
}

static cl_kernel make_kernel(cl_program prog, const char *name)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(prog, name, &err);

    CL_CHECK_ERR(err, name);
    return kernel;
}

static size_t kernel_limit(struct dev *dev, cl_kernel kernel)
{
    size_t limit = 0;

    CL_CHECK(clGetKernelWorkGroupInfo(
        kernel, dev->device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof limit, &limit, NULL));
    return limit;
}

static void print_kernel_facts(struct dev *dev, cl_kernel dist,
                               cl_kernel amin, int wg)
{
    size_t a = kernel_limit(dev, dist);
    size_t b = kernel_limit(dev, amin);
    size_t limit = a < b ? a : b;
    cl_uint nargs = 0;
    char name[64];

    printf("work-group limit: distance %zu, min %zu\n", a, b);
    if ((size_t)wg > limit) {
        fprintf(stderr, "wg %d exceeds limit %zu\n", wg, limit);
        exit(EXIT_FAILURE);
    }
    CL_CHECK(clGetKernelInfo(dist, CL_KERNEL_NUM_ARGS,
        sizeof nargs, &nargs, NULL));
    memset(name, 0, sizeof name);
    CL_CHECK(clGetKernelArgInfo(dist, 0, CL_KERNEL_ARG_NAME,
        sizeof name, name, NULL));
    printf("distance args: %u, first %s\n", nargs, name);
}

static cl_mem buf_from(cl_context ctx, cl_mem_flags flags,
                       size_t nbytes, void *host)
{
    cl_int err;
    cl_mem mem = clCreateBuffer(ctx, flags, nbytes, host, &err);

    CL_CHECK_ERR(err, "clCreateBuffer");
    return mem;
}

static cl_mem make_slice(cl_mem parent, size_t origin, size_t bytes,
                         size_t *offset_out)
{
    cl_int err;
    cl_buffer_region region;
    cl_mem slice;
    size_t offset = 0;

    region.origin = origin;
    region.size = bytes;
    slice = clCreateSubBuffer(parent, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK_ERR(err, "clCreateSubBuffer");
    CL_CHECK(clGetMemObjectInfo(slice, CL_MEM_OFFSET,
        sizeof offset, &offset, NULL));
    if (offset != origin)
        die("sub-buffer offset does not match the origin");
    *offset_out = offset;
    return slice;
}

static void make_buffers(struct dev *dev, const struct job *job,
                         struct bufs *b)
{
    int words = job->start + job->count;
    size_t dim = (size_t)job->dim;
    size_t table_n = (size_t)words * dim * sizeof(float);
    size_t slice_n = (size_t)job->count * dim * sizeof(float);
    size_t origin = (size_t)job->start * dim * sizeof(float);
    size_t query_n = dim * sizeof(float);
    cl_context ctx = dev->context;

    memset(b, 0, sizeof *b);
    b->table = xmalloc(table_n);
    b->query = xmalloc(query_n);
    fill_table(b->table, words, job->dim);
    fill_query(b->query, b->table, job);
    b->parent = buf_from(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, table_n, b->table);
    b->slice = make_slice(b->parent, origin, slice_n, &b->offset);
    b->query_mem = buf_from(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, query_n, b->query);
    b->dist = buf_from(ctx, CL_MEM_READ_WRITE,
        (size_t)job->count * sizeof(float), NULL);
    b->out_index = buf_from(
        ctx, CL_MEM_READ_WRITE, sizeof(cl_int), NULL);
    b->out_value = buf_from(
        ctx, CL_MEM_READ_WRITE, sizeof(float), NULL);
}

static void print_search(const struct job *job, size_t offset)
{
    printf("search: start %d, count %d, dim %d, vec %d, wg %d\n",
        job->start, job->count, job->dim, job->vec, job->wg);
    printf("distance kernel: %s\n", kernel_for(job));
    printf("query: word %d, component 0 + %.2f\n",
        job->start + job->target, QUERY_BUMP);
    printf("sub-buffer offset: %zu bytes\n", offset);
    printf("readback: %s\n", job->mapped ? "map" : "copy");
    printf("groups: %d, work-items per group: %d\n",
        job->count, job->wg);
}

static void set_distance_args(cl_kernel kernel, const struct job *job,
                              struct bufs *b)
{
    cl_int dim = job->dim;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->slice));
    CL_CHECK(clSetKernelArg(
        kernel, 1, sizeof(cl_mem), &b->query_mem));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->dist));
    CL_CHECK(clSetKernelArg(
        kernel, 3, (size_t)job->wg * sizeof(float), NULL));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof dim, &dim));
}

static void set_min_args(cl_kernel kernel, const struct job *job,
                         struct bufs *b)
{
    cl_int count = job->count;
    size_t floats = (size_t)job->wg * sizeof(float);
    size_t ints = (size_t)job->wg * sizeof(cl_int);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b->dist));
    CL_CHECK(clSetKernelArg(
        kernel, 1, sizeof(cl_mem), &b->out_index));
    CL_CHECK(clSetKernelArg(
        kernel, 2, sizeof(cl_mem), &b->out_value));
    CL_CHECK(clSetKernelArg(kernel, 3, floats, NULL));
    CL_CHECK(clSetKernelArg(kernel, 4, ints, NULL));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof count, &count));
}

static void enqueue_distance(struct dev *dev, const struct job *job,
                             cl_kernel kernel, cl_event *ev)
{
    size_t local = (size_t)job->wg;
    size_t global = local * (size_t)job->count;

    CL_CHECK(clEnqueueNDRangeKernel(
        dev->queue, kernel, 1, NULL, &global, &local,
        0, NULL, ev));
}

static double elapsed_ms(cl_event ev)
{
    cl_ulong start = 0;
    cl_ulong end = 0;

    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
        sizeof start, &start, NULL));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
        sizeof end, &end, NULL));
    return (double)(end - start) / 1.0e6;
}

static double median_of(double *v, int n)
{
    int i;

    for (i = 1; i < n; i++) {
        double x = v[i];
        int j = i;

        while (j > 0 && v[j - 1] > x) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
    return v[n / 2];
}

static double median_distance(struct dev *dev, const struct job *job,
                              cl_kernel kernel)
{
    double samples[TIMED];
    int i;

    for (i = 0; i < WARMUP; i++)
        enqueue_distance(dev, job, kernel, NULL);
    CL_CHECK(clFinish(dev->queue));
    for (i = 0; i < TIMED; i++) {
        cl_event ev;

        enqueue_distance(dev, job, kernel, &ev);
        CL_CHECK(clWaitForEvents(1, &ev));
        samples[i] = elapsed_ms(ev);
        CL_CHECK(clReleaseEvent(ev));
    }
    return median_of(samples, TIMED);
}

static double run_min(struct dev *dev, const struct job *job,
                      cl_kernel kernel)
{
    size_t local = (size_t)job->wg;
    cl_event ev;
    double ms;

    CL_CHECK(clEnqueueNDRangeKernel(
        dev->queue, kernel, 1, NULL, &local, &local,
        0, NULL, &ev));
    CL_CHECK(clWaitForEvents(1, &ev));
    ms = elapsed_ms(ev);
    CL_CHECK(clReleaseEvent(ev));
    return ms;
}

static void read_copy(struct dev *dev, struct bufs *b,
                      struct answer *got)
{
    cl_int index = -1;
    float value = -1.0f;

    CL_CHECK(clEnqueueReadBuffer(
        dev->queue, b->out_index, CL_TRUE, 0,
        sizeof index, &index, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(
        dev->queue, b->out_value, CL_TRUE, 0,
        sizeof value, &value, 0, NULL, NULL));
    got->index = (int)index;
    got->dist = value;
}

static void read_map(struct dev *dev, struct bufs *b,
                     struct answer *got)
{
    cl_int err;
    cl_int *ip;
    float *fp;

    ip = clEnqueueMapBuffer(
        dev->queue, b->out_index, CL_TRUE, CL_MAP_READ,
        0, sizeof(cl_int), 0, NULL, NULL, &err);
    CL_CHECK_ERR(err, "map index");
    fp = clEnqueueMapBuffer(
        dev->queue, b->out_value, CL_TRUE, CL_MAP_READ,
        0, sizeof(float), 0, NULL, NULL, &err);
    CL_CHECK_ERR(err, "map value");
    got->index = (int)(*ip);
    got->dist = *fp;
    CL_CHECK(clEnqueueUnmapMemObject(
        dev->queue, b->out_index, ip, 0, NULL, NULL));
    CL_CHECK(clEnqueueUnmapMemObject(
        dev->queue, b->out_value, fp, 0, NULL, NULL));
    CL_CHECK(clFinish(dev->queue));
}

static void read_answer(struct dev *dev, const struct job *job,
                        struct bufs *b, struct answer *got)
{
    if (job->mapped)
        read_map(dev, b, got);
    else
        read_copy(dev, b, got);
}

static void run_search(struct dev *dev, const struct job *job,
                       cl_kernel dist, cl_kernel amin,
                       struct bufs *b, struct answer *got)
{
    set_distance_args(dist, job, b);
    got->dist_ms = median_distance(dev, job, dist);
    set_min_args(amin, job, b);
    got->min_ms = run_min(dev, job, amin);
    read_answer(dev, job, b, got);
}

static int report(const struct job *job, const struct pair *cpu,
                  const struct answer *got)
{
    float err = fabsf(got->dist - cpu->dist);
    float gap = cpu->second - cpu->dist;
    int index_ok = got->index == cpu->index;
    int expect_ok = cpu->index == job->target;
    int dist_ok = err <= DIST_TOL;
    int unique = gap > GAP_MIN;
    int pass = index_ok && expect_ok && dist_ok && unique;

    printf("closest word: %d (slice index %d)\n",
        job->start + got->index, got->index);
    printf("distance: %.6e\n", got->dist);
    printf("cpu word: %d (slice index %d)\n",
        job->start + cpu->index, cpu->index);
    printf("cpu distance: %.6e\n", cpu->dist);
    if (job->count < 2)
        printf("second distance: n/a\n");
    else
        printf("second distance: %.6e\n", cpu->second);
    printf("abs err: %.3e\n", err);
    printf("distance median: %.4f ms\n", got->dist_ms);
    printf("min kernel: %.4f ms\n", got->min_ms);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static void release_mem(cl_mem mem)
{
    if (mem)
        CL_CHECK(clReleaseMemObject(mem));
}

static void release_all(struct dev *dev, cl_program prog,
                        cl_kernel dist, cl_kernel amin,
                        struct bufs *b)
{
    CL_CHECK(clReleaseKernel(dist));
    CL_CHECK(clReleaseKernel(amin));
    CL_CHECK(clReleaseProgram(prog));
    release_mem(b->slice);
    release_mem(b->parent);
    release_mem(b->query_mem);
    release_mem(b->dist);
    release_mem(b->out_index);
    release_mem(b->out_value);
    CL_CHECK(clReleaseCommandQueue(dev->queue));
    CL_CHECK(clReleaseContext(dev->context));
    free(b->table);
    free(b->query);
}

int main(int argc, char **argv)
{
    struct job job;
    struct dev dev;
    struct bufs bufs;
    struct pair cpu;
    struct answer got;
    char *src;
    size_t nsrc = 0;
    cl_program prog;
    cl_kernel dist;
    cl_kernel amin;
    int pass;

    setvbuf(stdout, NULL, _IOLBF, 0);
    parse_job(argc, argv, &job);
    memset(&dev, 0, sizeof dev);
    open_gpu(&dev);
    print_device(&dev);
    check_device_fit(&dev, &job);
    src = load_source(argv[0], &nsrc);
    prog = build_source(&dev, src, nsrc);
    free(src);
    print_kernels(prog);
    dist = make_kernel(prog, kernel_for(&job));
    amin = make_kernel(prog, "find_min");
    print_kernel_facts(&dev, dist, amin, job.wg);
    make_buffers(&dev, &job, &bufs);
    print_search(&job, bufs.offset);
    cpu_closest(&job, bufs.table, bufs.query, &cpu);
    run_search(&dev, &job, dist, amin, &bufs, &got);
    pass = report(&job, &cpu, &got);
    release_all(&dev, prog, dist, amin, &bufs);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
