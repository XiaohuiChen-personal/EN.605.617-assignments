/* One work-group computes one word. The host makes dim a multiple
 * of the load width, and the work-group size a power of two.
 * Every work-item in the group calls block_sum, because it barriers.
 */

float block_sum(__local float *scratch, float acc)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t step;

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (step = lsize >> 1; step > 0; step >>= 1) {
        if (lid < step)
            scratch[lid] += scratch[lid + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return scratch[0];
}

float sumsq8(float8 d)
{
    float4 lo = d.lo;
    float4 hi = d.hi;

    return dot(lo, lo) + dot(hi, hi);
}

void scan_min(__global const float *dist, int count,
              __private float *best, __private int *best_i)
{
    size_t lid = get_local_id(0);
    int step = (int)get_local_size(0);
    int i;

    for (i = (int)lid; i < count; i += step) {
        float v = dist[i];
        if (v < *best || (v == *best && i < *best_i)) {
            *best = v;
            *best_i = i;
        }
    }
}

__kernel void distance_v1(
    __global const float *table,
    __constant float *query,
    __global float *dist,
    __local float *scratch,
    int dim)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t word = get_group_id(0);
    size_t base = word * (size_t)dim;
    size_t k;
    float acc = 0.0f;

    for (k = lid; k < (size_t)dim; k += lsize) {
        float diff = table[base + k] - query[k];
        acc += diff * diff;
    }
    acc = block_sum(scratch, acc);
    if (lid == 0)
        dist[word] = acc;
}

__kernel void distance_v4(
    __global const float *table,
    __constant float *query,
    __global float *dist,
    __local float *scratch,
    int dim)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t word = get_group_id(0);
    size_t base = word * (size_t)dim;
    size_t nvec = (size_t)dim / 4;
    size_t k;
    float acc = 0.0f;

    for (k = lid; k < nvec; k += lsize) {
        float4 t = vload4(k, table + base);
        float4 q = vload4(k, query);
        float4 d = t - q;
        acc += dot(d, d);
    }
    acc = block_sum(scratch, acc);
    if (lid == 0)
        dist[word] = acc;
}

__kernel void distance_v8(
    __global const float *table,
    __constant float *query,
    __global float *dist,
    __local float *scratch,
    int dim)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t word = get_group_id(0);
    size_t base = word * (size_t)dim;
    size_t nvec = (size_t)dim / 8;
    size_t k;
    float acc = 0.0f;

    for (k = lid; k < nvec; k += lsize) {
        float8 t = vload8(k, table + base);
        float8 q = vload8(k, query);
        acc += sumsq8(t - q);
    }
    acc = block_sum(scratch, acc);
    if (lid == 0)
        dist[word] = acc;
}

__kernel void find_min(
    __global const float *dist,
    __global int *out_index,
    __global float *out_value,
    __local float *vals,
    __local int *idxs,
    int count)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t step;
    float best = INFINITY;
    int best_i = -1;

    scan_min(dist, count, &best, &best_i);
    vals[lid] = best;
    idxs[lid] = best_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (step = lsize >> 1; step > 0; step >>= 1) {
        if (lid < step) {
            float v = vals[lid + step];
            int ix = idxs[lid + step];
            if (v < vals[lid] ||
                (v == vals[lid] && ix < idxs[lid])) {
                vals[lid] = v;
                idxs[lid] = ix;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        out_index[0] = idxs[0];
        out_value[0] = vals[0];
    }
}
