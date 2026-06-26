// DS4 Metal matvec kernels used by generation.

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};

struct ds4_metal_args_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

struct ds4_metal_args_mul_mv_ext {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Decode-time Q8_0 matrix-vector multiply. DS4 uses this for Q8_0 dense
// projections such as shared experts and output-side small matvecs.
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Decode shared-expert gate/up projections followed by SwiGLU:
//
//     mid = silu(min(gate, limit)) * clamp(up, -limit, limit)
//
// DS4's shared expert uses two Q8_0 matrices with the same input row.  This
// kernel preserves the exact Q8_0 dot-product reduction shape for both
// projections, still writes gate/up for diagnostics, and derives `mid` in the
// same lane that owns the reduced output row.  The point is not to fuse two
// independent weight streams into one matmul; it is to remove the separate
// activation pass and its reread of the two 2048-wide rows.
[[host_name("kernel_dsv4_shared_gate_up_swiglu_q8_0")]]
kernel void kernel_dsv4_shared_gate_up_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
                                 (i12 / args.r2) * args.nb02 +
                                 (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device const char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device const char *)src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;

            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }

            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *up_f32 = (device float *)dst_up +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u;
        }
    }
}

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    device const T0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

// Decode-time dense F32/F16 matrix-vector multiply. The instantiated kernels
// handle unquantized DS4 weights and activations that are already float rows.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

// Host-visible dense matvec variants used by the graph for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

// Vectorized dense matvec using float4/half4 loads. DS4 uses this where the
// inner dimension and alignment make vector loads cheaper than scalar lanes.
template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

// Host-visible vectorized dense matvec variants for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;

// DS4 compressor projections always compute two same-shaped F16 matvecs from
// the same normalized activation: one for projected KV and one for pooling
// scores.  This paired variant keeps the exact dense F16 row-reduction shape
// for each matrix, but shares one dispatch and one activation stream.
template<short NR0, typename args_t>
void kernel_mul_mv_f16_f32_pair_4_impl(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);

    device const half  * ax_a [NR0];
    device const half4 * ax4_a[NR0];
    device const half  * ax_b [NR0];
    device const half4 * ax4_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax_a [row] = (device const half  *) ((device char *) src0_a + offset0);
        ax4_a[row] = (device const half4 *) ((device char *) src0_a + offset0);
        ax_b [row] = (device const half  *) ((device char *) src0_b + offset0);
        ax4_b[row] = (device const half4 *) ((device char *) src0_b + offset0);
    }

    float sum_a[NR0] = { 0.f };
    float sum_b[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4_a = ax4_a[row] + (ib*NB + il*NF)/4;
            device const half4 * xb4_b = ax4_b[row] + (ib*NB + il*NF)/4;

            float suma = 0.f;
            float sumb = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                const float4 yv = float4(yl4[i]);
                suma += dot(float4(xb4_a[i]), yv);
                sumb += dot(float4(xb4_b[i]), yv);
            }

            sum_a[row] += suma;
            sum_b[row] += sumb;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            const float yi = y[i];
            sum_a[row] += ax_a[row][i] * yi;
            sum_b[row] += ax_b[row][i] * yi;
        }
    }

    device float * dst_a_f32 = (device float *) dst_a + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_b_f32 = (device float *) dst_b + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_a_f32, sum_a, r0, args.ne01, tiisg, sgitg, shmem);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    helper_mv_reduce_and_write<NR0>(dst_b_f32, sum_b, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename args_t>
void kernel_mul_mv_f16_f32_pair_4_disp(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_f16_f32_pair_4_impl<2>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_f16_f32_pair_4_impl<4>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
    }
}

kernel void kernel_mul_mv_f16_f32_pair_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_f16_f32_pair_4_disp<constant ds4_metal_args_mul_mv &>(
            args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

// Scalar fallback for short rows. It trades parallelism for lower dispatch and
// reduction overhead when DS4 asks for tiny dense matvecs.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ds4_metal_args_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

// Host-visible short-row dense matvec variants.
template [[host_name("kernel_mul_mv_f32_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

// DS4 small-batch mat-vec kernel used for 2..8 prompt tokens.
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// Small-batch prompt matvec for 2..5 tokens. It bridges decode-style matvec and
// full matmul when DS4 prefill chunks are too small to amortize matrix tiles.
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;

// Host-visible small-batch variants. DS4 currently needs F16 and Q8_0 weights
// for r1=2..5 during the prompt path.
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,      4,  dequantize_f16_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0, 32, dequantize_q8_0_t4>;

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];

#ifdef DS4_METAL_HAS_TENSOR
template<
    short NR0, short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    threadgroup SA *sb = sa + NR0*NK;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NK, NR1));

    device const T1 *ptrB = (device const T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (!FC_mul_mm_bc_out || r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }
        for (int work = tiitg; work < NK*NR1; work += NUM_THREADS) {
            const int col = work/NK;
            const int k = work%NK;
            if ((!FC_mul_mm_bc_out && !FC_mul_mm_bc_inp) ||
                (r1 + col < N && loop_k + k < K)) {
                sb[col*NK + k] = (SA)ptrB[(uint64_t)(r1 + col)*strideB + loop_k + k];
            } else {
                sb[col*NK + k] = (SA)0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(0, 0);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    if (!FC_mul_mm_bc_out) {
        device float *dst_tile = dst_batch + r0 + (uint64_t)r1*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_mul_mm_mpp<64, 32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp")]]  kernel mul_mm_mpp_t kernel_mul_mm_mpp<64, 32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;

// Retained Metal4/TensorOps dense prefill kernel.  The legacy MPP prototype
// staged both operands in threadgroup memory; this version stages only the
// model weight tile and lets MPP read the dense RHS activation matrix directly
// from device memory.  That direct-RHS shape was the clear win for DS4's large
// aligned F16/Q8_0 prompt matmuls.  The host selects the widest token tile that
// evenly divides the batch, with 128-token tiles retained after the 64-token
// retest was neutral or slower.
template<
    short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device T1 *ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

typedef decltype(kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_direct_rhs_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n128")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
#endif

// Tiled matrix-matrix kernel used for prompt batches larger than 8. DS4 uses
// this to turn prefill into large simdgroup matrix operations; each block_q
// contains 16*nl weights.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // Pointer-form store avoids a slower address-lowering path in
                // current Apple Metal compilers for this dequantized tile write.
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

kernel void kernel_mul_mm_f16_f32_pair(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_a = (threadgroup half *)(shmem);
    threadgroup half * sa_b = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0;

    device const half4x4 * xa = (device const half4x4 *)(src0_a + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const half4x4 * xb = (device const half4x4 *)(src0_b + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_a[8];
    simdgroup_float8x8 mc_b[8];

    for (short i = 0; i < 8; i++) {
        mc_a[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_b[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_a;
        half4x4 temp_b;
        dequantize_f16(xa, il, temp_a);
        dequantize_f16(xb, il, temp_b);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;

            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;

            const short ib = 8*sx + sy;

            *(sa_a + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            *(sa_b + 64*ib + 8*ly + lx) = temp_b[i/4][i%4];
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (half) *((device float *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < 1) ? il + 2 : il % 2;
        xa = (il < 2) ? xa + 2 : xa;
        xb = (il < 2) ? xb + 2 : xb;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_a = (sa_a + 4*64*(sgitg%2));
        threadgroup const half * lsma_b = (sa_b + 4*64*(sgitg%2));
        threadgroup const half * lsmb   = (sb   + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_a + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_a[i], mb[i/4], ma[i%4], mc_a[i]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_b + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_b[i], mb[i/4], ma[i%4], mc_b[i]);
            }

            lsma_a += 8*64;
            lsma_b += 8*64;
            lsmb   += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C_a = (device float *) dst_a +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;
        device float * C_b = (device float *) dst_b +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i], C_a + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
            simdgroup_store(mc_b[i], C_b + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = (threadgroup float *) shmem;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_a + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_b[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_b + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

// Host-visible prefill matmul variants for F16 and Q8_0 weights.
template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, half4x4, 1, dequantize_f16,  half,  half4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;

// ---- Dense NEOX RoPE (Fase 3.5) --------------------------------------------
// Qwen2/Llama convention: rotate dim i with i + n_rot/2, per head. One thread
// per (head, i) pair, i in [0, n_rot/2). In-place on an f32 [n_head*head_dim].
struct ds4_dense_rope_args {
    uint  n_head;
    uint  head_dim;
    uint  n_rot;
    uint  pos;
    float freq_base;
};

kernel void kernel_dense_rope_neox_f32(
        constant ds4_dense_rope_args & a [[buffer(0)]],
        device float * x [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const uint rot_half = a.n_rot / 2u;
    const uint total = a.n_head * rot_half;
    if (gid >= total) return;
    const uint h = gid / rot_half;
    const uint i = gid % rot_half;
    device float * head = x + h * a.head_dim;
    const float freq  = pow(a.freq_base, -2.0f * (float)i / (float)a.n_rot);
    const float theta = (float)a.pos * freq;
    const float c = cos(theta);
    const float s = sin(theta);
    const float x0 = head[i];
    const float x1 = head[i + rot_half];
    head[i]          = x0 * c - x1 * s;
    head[i + rot_half] = x0 * s + x1 * c;
}

// ---- Dense FFN building blocks (Fase 3.5) ----------------------------------
// SwiGLU activation: out[i] = silu(gate[i]) * up[i]. One thread per element.
kernel void kernel_dense_swiglu_f32(
        constant uint & n [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * up   [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// RMSNorm with learned weight: out[i] = x[i] / rms(x) * w[i]. Single-thread
// reference reduction (correctness first; optimize later). One thread total.
struct ds4_dense_rmsnorm_args { uint n; float eps; };
kernel void kernel_dense_rms_norm_f32(
        constant ds4_dense_rmsnorm_args & a [[buffer(0)]],
        device const float * x   [[buffer(1)]],
        device const float * w   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    float ss = 0.0f;
    for (uint i = 0; i < a.n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / sqrt(ss / (float)a.n + a.eps);
    for (uint i = 0; i < a.n; i++) out[i] = x[i] * scale * w[i];
}

// Parallel RMSNorm (Fase opt). One threadgroup of 128 threads; sum-of-squares
// via per-simdgroup simd_sum + threadgroup reduction, then a parallel write.
// Replaces the single-thread kernel above. Dispatch: grid=(1,1,1) tpg=(128,1,1).
kernel void kernel_dense_rms_norm_f32_sg(
        constant ds4_dense_rmsnorm_args & a [[buffer(0)]],
        device const float * x   [[buffer(1)]],
        device const float * w   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint   tid   [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint nt = 128u;          // threads/threadgroup (must match dispatch)
    threadgroup float sdata[4];    // nt/32 = 4 simdgroups
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += nt) ss += x[i]*x[i];
    ss = simd_sum(ss);
    if (tiisg == 0) sdata[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (uint s = 0; s < nt/32u; s++) tot += sdata[s];
        sdata[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(sdata[0] / (float)a.n + a.eps);
    for (uint i = tid; i < a.n; i += nt) out[i] = x[i]*scale*w[i];
}

// Element-wise add (residual): a[i] += b[i]. One thread per element.
kernel void kernel_dense_add_f32(
        constant uint & n [[buffer(0)]],
        device       float * a [[buffer(1)]],
        device const float * b [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    a[gid] += b[gid];
}

// ---- Dense GQA attention (decode) (Fase 3.5) -------------------------------
// One query token at position pos attends to cached positions 0..n_ctx-1.
// GQA: query head hq reads kv head hq/(n_head/n_kv). K/V caches are laid out
// [n_ctx][n_kv*head_dim]. One thread per query head. Online softmax.
struct ds4_dense_attn_args {
    uint  n_head;
    uint  n_kv;
    uint  head_dim;
    uint  n_ctx;
    float scale;
};
kernel void kernel_dense_attn_decode_f32(
        constant ds4_dense_attn_args & a [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const float * kcache [[buffer(2)]],
        device const float * vcache [[buffer(3)]],
        device       float * out    [[buffer(4)]],
        uint hq [[thread_position_in_grid]]) {
    if (hq >= a.n_head) return;
    const uint group = a.n_head / a.n_kv;
    const uint hkv = hq / group;
    const uint hd = a.head_dim;
    const uint kvdim = a.n_kv * hd;
    device const float * qh = q + hq * hd;
    device       float * oh = out + hq * hd;

    float maxs = -1e30f;
    for (uint t = 0; t < a.n_ctx; t++) {
        device const float * kt = kcache + t * kvdim + hkv * hd;
        float dot = 0.0f;
        for (uint d = 0; d < hd; d++) dot += qh[d] * kt[d];
        dot *= a.scale;
        if (dot > maxs) maxs = dot;
    }
    for (uint d = 0; d < hd; d++) oh[d] = 0.0f;
    float sum = 0.0f;
    for (uint t = 0; t < a.n_ctx; t++) {
        device const float * kt = kcache + t * kvdim + hkv * hd;
        device const float * vt = vcache + t * kvdim + hkv * hd;
        float dot = 0.0f;
        for (uint d = 0; d < hd; d++) dot += qh[d] * kt[d];
        const float w = exp(dot * a.scale - maxs);
        sum += w;
        for (uint d = 0; d < hd; d++) oh[d] += w * vt[d];
    }
    const float inv = 1.0f / sum;
    for (uint d = 0; d < hd; d++) oh[d] *= inv;
}

// Optimized attention decode (Fase opt step 3). One simdgroup (32 lanes) per
// query head, single online-softmax pass over the KV cache. Lane L owns output
// dims {L, L+32, ...}; the q·k score is reduced across the 32 lanes via
// simd_sum. Replaces the 28-thread, two-pass scalar kernel above.
// Dispatch: threadgroups=(n_head,1,1), threadsPerThreadgroup=(32,1,1).
// head_dim must be <= 256 (acc/qreg sized for ceil(256/32)=8).
kernel void kernel_dense_attn_decode_f32_sg(
        constant ds4_dense_attn_args & a [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const float * kcache [[buffer(2)]],
        device const float * vcache [[buffer(3)]],
        device       float * out    [[buffer(4)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {
    const uint hq = tgpig.x;
    if (hq >= a.n_head) return;
    const uint group = a.n_head / a.n_kv;
    const uint hkv = hq / group;
    const uint hd = a.head_dim;
    const uint kvdim = a.n_kv * hd;
    device const float * qh = q + hq * hd;
    device       float * oh = out + hq * hd;

    const uint ndl = (hd + 31u) / 32u;   // dims per lane (<= 8)
    float qreg[8];
    float acc[8];
    for (uint j = 0; j < ndl; j++) {
        const uint d = lane + 32u*j;
        qreg[j] = (d < hd) ? qh[d] : 0.0f;
        acc[j]  = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (uint t = 0; t < a.n_ctx; t++) {
        device const half * kt = (device const half *)kcache + (ulong)t*kvdim + hkv*hd;
        float p = 0.0f;
        for (uint j = 0; j < ndl; j++) {
            const uint d = lane + 32u*j;
            if (d < hd) p += qreg[j] * kt[d];
        }
        const float s = simd_sum(p) * a.scale;
        const float m_new = max(m, s);
        const float corr  = exp(m - m_new);     // first iter: exp(-inf)=0
        const float pe    = exp(s - m_new);
        l = l * corr + pe;
        device const half * vt = (device const half *)vcache + (ulong)t*kvdim + hkv*hd;
        for (uint j = 0; j < ndl; j++) {
            const uint d = lane + 32u*j;
            if (d < hd) acc[j] = acc[j]*corr + pe * vt[d];
        }
        m = m_new;
    }
    const float inv = 1.0f / l;
    for (uint j = 0; j < ndl; j++) {
        const uint d = lane + 32u*j;
        if (d < hd) oh[d] = acc[j] * inv;
    }
}

// ===========================================================================
// Batched-prefill kernels (process M tokens at once). The matmuls use
// kernel_mul_mm (q4_K/q6_K); these cover the elementwise + attention parts.
// ===========================================================================

// Batched RMSNorm over M rows. Dispatch grid=(M,1,1) tpg=(128,1,1).
kernel void kernel_dense_rms_norm_f32_batch(
        constant ds4_dense_rmsnorm_args & a [[buffer(0)]],
        device const float * x   [[buffer(1)]],
        device const float * w   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint   tgig  [[threadgroup_position_in_grid]],
        uint   tid   [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint nt = 128u;
    threadgroup float sdata[4];
    device const float * xr   = x   + (ulong)tgig * a.n;
    device       float * outr = out + (ulong)tgig * a.n;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += nt) ss += xr[i]*xr[i];
    ss = simd_sum(ss);
    if (tiisg == 0) sdata[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0.0f; for (uint s = 0; s < nt/32u; s++) tot += sdata[s]; sdata[0] = tot; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(sdata[0] / (float)a.n + a.eps);
    for (uint i = tid; i < a.n; i += nt) outr[i] = xr[i]*scale*w[i];
}

// Batched NEOX RoPE for M tokens. Token tok sits at position a.pos+tok; q is
// [M, n_head*head_dim]. Dispatch n_tok*n_head*(n_rot/2) threads.
kernel void kernel_dense_rope_neox_f32_batch(
        constant ds4_dense_rope_args & a [[buffer(0)]],
        device float * x [[buffer(1)]],
        constant uint & n_tok [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint rot_half = a.n_rot / 2u;
    const uint per_tok = a.n_head * rot_half;
    if (gid >= n_tok * per_tok) return;
    const uint tok = gid / per_tok;
    const uint rem = gid % per_tok;
    const uint h = rem / rot_half;
    const uint i = rem % rot_half;
    const uint qd = a.n_head * a.head_dim;
    device float * head = x + (ulong)tok*qd + h*a.head_dim;
    const float freq  = pow(a.freq_base, -2.0f * (float)i / (float)a.n_rot);
    const float theta = (float)(a.pos + tok) * freq;
    const float c = cos(theta), s = sin(theta);
    const float x0 = head[i], x1 = head[i + rot_half];
    head[i]            = x0 * c - x1 * s;
    head[i + rot_half] = x0 * s + x1 * c;
}

// Batched causal attention over M query tokens. Query token m (at absolute
// position start_pos+m) attends keys [0, start_pos+m]. One simdgroup per
// (head, token). Dispatch grid=(n_head, M, 1) tpg=(32,1,1). head_dim<=256.
kernel void kernel_dense_attn_prefill_f32(
        constant ds4_dense_attn_args & a [[buffer(0)]],
        device const float * q      [[buffer(1)]],   // [M, n_head*head_dim]
        device const float * kcache [[buffer(2)]],
        device const float * vcache [[buffer(3)]],
        device       float * out    [[buffer(4)]],   // [M, n_head*head_dim]
        constant uint & start_pos   [[buffer(5)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {
    const uint hq = tgpig.x;
    const uint m  = tgpig.y;
    if (hq >= a.n_head) return;
    const uint group = a.n_head / a.n_kv;
    const uint hkv = hq / group;
    const uint hd  = a.head_dim;
    const uint kvdim = a.n_kv * hd;
    const uint qd  = a.n_head * hd;
    const uint n_causal = start_pos + m + 1u;
    device const float * qh = q + (ulong)m*qd + hq*hd;
    device       float * oh = out + (ulong)m*qd + hq*hd;

    const uint ndl = (hd + 31u) / 32u;
    float qreg[8], acc[8];
    for (uint j = 0; j < ndl; j++) {
        const uint d = lane + 32u*j;
        qreg[j] = (d < hd) ? qh[d] : 0.0f;
        acc[j]  = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (uint t = 0; t < n_causal; t++) {
        device const half * kt = (device const half *)kcache + (ulong)t*kvdim + hkv*hd;
        float p = 0.0f;
        for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; if (d < hd) p += qreg[j]*kt[d]; }
        const float s = simd_sum(p) * a.scale;
        const float m_new = max(mx, s);
        const float corr = exp(mx - m_new);
        const float pe   = exp(s - m_new);
        l = l*corr + pe;
        device const half * vt = (device const half *)vcache + (ulong)t*kvdim + hkv*hd;
        for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; if (d < hd) acc[j] = acc[j]*corr + pe*vt[d]; }
        mx = m_new;
    }
    const float inv = 1.0f / l;
    for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; if (d < hd) oh[d] = acc[j]*inv; }
}

// Tiled flash-attention prefill (FlashAttention-2, simdgroup matrices). One
// threadgroup = one simdgroup (32 lanes) computes 8 query rows for one head. The
// key sequence is streamed in tiles of 8; S=Q.K^T and O+=P.V use the matrix units,
// with a per-row online softmax. f16 KV cache, causal mask, GQA. head_dim<=128.
// Grid=(n_head, M/8, 1) tpg=(32,1,1) — only FULL 8-row blocks; the host runs the
// scalar kernel for the M%8 tail. O lives in threadgroup memory so it can be
// rescaled per row across key tiles. Reference kernel above stays the default.
kernel void kernel_dense_flash_prefill_f32(
        constant ds4_dense_attn_args & a [[buffer(0)]],
        device const float * q      [[buffer(1)]],   // [M, n_head*head_dim] f32
        device const half  * kcache [[buffer(2)]],   // [n_ctx, n_kv*head_dim] f16
        device const half  * vcache [[buffer(3)]],
        device       float * out    [[buffer(4)]],   // [M, n_head*head_dim] f32
        constant uint & start_pos   [[buffer(5)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {
    const uint hq = tgpig.x;
    const uint qb = tgpig.y;                 // query-block (8 rows)
    if (hq >= a.n_head) return;
    const uint hd  = a.head_dim;             // <= 128
    const uint group = a.n_head / a.n_kv;
    const uint hkv = hq / group;
    const uint kvdim = a.n_kv * hd;
    const uint qd  = a.n_head * hd;
    const uint ndt = hd / 8u;                // depth tiles (<=16)
    const uint q0  = qb * 8u;                // first query row of this block

    threadgroup float Ksh[8*128];
    threadgroup float Vsh[8*128];
    threadgroup float Ssh[8*8];
    threadgroup float Dsh[8*8];      // diagonal matrix (per-row rescale / 1/l)
    threadgroup float mrow[8];
    threadgroup float lrow[8];

    for (uint i = lane; i < 8u; i += 32u) { mrow[i] = -INFINITY; lrow[i] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Q tile and O accumulator stay in registers (ndt matrices of 8 rows x 8 cols)
    simdgroup_float8x8 Qm[16];
    simdgroup_float8x8 Om[16];
    device const float * qbase = q + (ulong)q0*qd + hq*hd;
    for (uint dt = 0; dt < ndt; dt++) {
        simdgroup_load(Qm[dt], qbase + dt*8u, qd, 0, false);
        Om[dt] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const uint qpos_max = start_pos + q0 + 7u;   // causal bound of the last row
    const uint n_keys   = qpos_max + 1u;         // keys [0, qpos_max]

    for (uint kt = 0; kt < n_keys; kt += 8u) {
        // cooperative f16 -> f32 load of this 8-key tile (K and V)
        for (uint i = lane; i < 8u*hd; i += 32u) {
            const uint kk = i / hd, dd = i % hd;
            const uint key = kt + kk;
            float kk_v = 0.0f, vv_v = 0.0f;
            if (key < n_keys) {
                kk_v = (float)kcache[(ulong)key*kvdim + hkv*hd + dd];
                vv_v = (float)vcache[(ulong)key*kvdim + hkv*hd + dd];
            }
            Ksh[i] = kk_v; Vsh[i] = vv_v;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // S[8 rows, 8 keys] = Q . K^T  (K loaded transposed -> [depth, key])
        simdgroup_float8x8 Sm = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (uint dt = 0; dt < ndt; dt++) {
            simdgroup_float8x8 Ktm;
            simdgroup_load(Ktm, Ksh + dt*8u, hd, 0, true);
            simdgroup_multiply_accumulate(Sm, Qm[dt], Ktm, Sm);
        }
        simdgroup_store(Sm, Ssh, 8, 0, false);
        for (uint i = lane; i < 64u; i += 32u) Dsh[i] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // per-row online softmax (lanes 0..7 own rows 0..7). Writes P into Ssh and
        // the per-row correction factor onto the diagonal of Dsh.
        if (lane < 8u) {
            const uint r = lane;
            const uint qg = start_pos + q0 + r;       // this row's absolute position
            float sc[8], rmax = -INFINITY;
            for (uint j = 0; j < 8u; j++) {
                const uint key = kt + j;
                float s = Ssh[r*8u + j] * a.scale;
                if (key > qg) s = -INFINITY;          // causal
                sc[j] = s; rmax = max(rmax, s);
            }
            const float m_old = mrow[r];
            const float m_new = max(m_old, rmax);
            float corr = exp(m_old - m_new);
            if (!isfinite(corr)) corr = 0.0f;
            float lsum = 0.0f;
            for (uint j = 0; j < 8u; j++) {
                const float p = (sc[j] == -INFINITY) ? 0.0f : exp(sc[j] - m_new);
                Ssh[r*8u + j] = p;
                lsum += p;
            }
            lrow[r] = lrow[r]*corr + lsum;
            mrow[r] = m_new;
            Dsh[r*8u + r] = corr;                      // diagonal rescale factor
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(corr) . O   (per-row rescale, in registers) ; then O += P . V
        simdgroup_float8x8 Dm, Pm;
        simdgroup_load(Dm, Dsh, 8, 0, false);
        simdgroup_load(Pm, Ssh, 8, 0, false);
        for (uint dt = 0; dt < ndt; dt++) {
            simdgroup_float8x8 Vm, scaled;
            simdgroup_multiply(scaled, Dm, Om[dt]);       // rescale rows of O
            simdgroup_load(Vm, Vsh + dt*8u, hd, 0, false);
            simdgroup_multiply_accumulate(scaled, Pm, Vm, scaled);
            Om[dt] = scaled;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // finalize: O = diag(1/l) . O, then store rows to out
    for (uint i = lane; i < 64u; i += 32u) Dsh[i] = 0.0f;
    if (lane < 8u) Dsh[lane*8u + lane] = (lrow[lane] > 0.0f) ? 1.0f/lrow[lane] : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Dinv;
    simdgroup_load(Dinv, Dsh, 8, 0, false);
    device float * obase = out + (ulong)q0*qd + hq*hd;
    for (uint dt = 0; dt < ndt; dt++) {
        simdgroup_float8x8 Of;
        simdgroup_multiply(Of, Dinv, Om[dt]);
        simdgroup_store(Of, obase + dt*8u, qd, 0, false);
    }
}

// f32 -> f16 narrowing copy. Used to store the KV cache in half precision: K/V are
// computed/biased/roped in f32 scratch, then converted into the f16 cache. Halves the
// KV memory footprint and the attention read traffic. Dispatch n threads.
kernel void kernel_dense_cvt_f32_to_f16(
        constant uint & n [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device       half  * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = (half)src[gid];
}

// Broadcast bias add over M rows: x[m,i] += bias[i]. args = {row_width, M*row_width}.
kernel void kernel_dense_add_bias_batch(
        constant uint2 & a [[buffer(0)]],
        device       float * x    [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.y) return;
    x[gid] += bias[gid % a.x];
}

// ===========================================================================
// Long-context attention: split-KV / flash-decoding. The key sequence is split
// into n_split chunks; one simdgroup per (head, split) computes a partial online
// softmax over its chunk (n_head*n_split simdgroups instead of n_head), then a
// combine pass merges the partials. Restores parallelism and cuts the serial
// per-token work at large context. head_dim <= 256.
// ===========================================================================
struct ds4_dense_attn_split_args {
    uint  n_head, n_kv, head_dim, n_ctx, n_split, chunk;
    float scale;
};

kernel void kernel_dense_attn_decode_split_f32(
        constant ds4_dense_attn_split_args & a [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const float * kcache [[buffer(2)]],
        device const float * vcache [[buffer(3)]],
        device       float * pm     [[buffer(4)]],   // [n_head, n_split]
        device       float * pl     [[buffer(5)]],   // [n_head, n_split]
        device       float * pacc   [[buffer(6)]],   // [n_head, n_split, head_dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {
    const uint hq = tgpig.x;
    const uint sp = tgpig.y;
    if (hq >= a.n_head || sp >= a.n_split) return;
    const uint group = a.n_head / a.n_kv;
    const uint hkv = hq / group;
    const uint hd = a.head_dim;
    const uint kvdim = a.n_kv * hd;
    device const float * qh = q + hq*hd;
    const uint t0 = sp * a.chunk;
    uint t1 = t0 + a.chunk; if (t1 > a.n_ctx) t1 = a.n_ctx;

    const uint ndl = (hd + 31u) / 32u;
    float qreg[8], acc[8];
    for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; qreg[j] = (d < hd) ? qh[d] : 0.0f; acc[j] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (uint t = t0; t < t1; t++) {
        device const half * kt = (device const half *)kcache + (ulong)t*kvdim + hkv*hd;
        float p = 0.0f;
        for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; if (d < hd) p += qreg[j]*kt[d]; }
        const float s = simd_sum(p) * a.scale;
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), pe = exp(s - m_new);
        l = l*corr + pe;
        device const half * vt = (device const half *)vcache + (ulong)t*kvdim + hkv*hd;
        for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; if (d < hd) acc[j] = acc[j]*corr + pe*vt[d]; }
        m = m_new;
    }
    const uint base = hq*a.n_split + sp;
    if (lane == 0) { pm[base] = m; pl[base] = l; }
    device float * po = pacc + (ulong)base*hd;
    for (uint j = 0; j < ndl; j++) { const uint d = lane + 32u*j; if (d < hd) po[d] = acc[j]; }
}

kernel void kernel_dense_attn_decode_combine_f32(
        constant ds4_dense_attn_split_args & a [[buffer(0)]],
        device const float * pm   [[buffer(1)]],
        device const float * pl   [[buffer(2)]],
        device const float * pacc [[buffer(3)]],
        device       float * out  [[buffer(4)]],   // [n_head, head_dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {
    const uint hq = tgpig.x;
    if (hq >= a.n_head) return;
    const uint hd = a.head_dim, S = a.n_split;
    float M = -INFINITY;
    for (uint s = 0; s < S; s++) M = max(M, pm[hq*S + s]);
    float L = 0.0f;
    for (uint s = 0; s < S; s++) L += exp(pm[hq*S + s] - M) * pl[hq*S + s];
    const float inv = 1.0f / L;
    device float * oh = out + hq*hd;
    const uint ndl = (hd + 31u) / 32u;
    for (uint j = 0; j < ndl; j++) {
        const uint d = lane + 32u*j;
        if (d < hd) {
            float acc = 0.0f;
            for (uint s = 0; s < S; s++) acc += exp(pm[hq*S + s] - M) * pacc[(ulong)(hq*S + s)*hd + d];
            oh[d] = acc * inv;
        }
    }
}

// ---- Dense Q6_K matvec (Fase: Q6_K support) --------------------------------
// out[row] = dot(dequant(W[row]), x). Canonical GGML Q6_K dequant inline. One
// thread per output row (reference; optimize later). Block = 210 bytes:
// ql[128] qh[64] scales[16](int8) d(f16). in_dim must be a multiple of 256.
struct ds4_dense_mvq_args { uint in_dim; uint out_dim; };
kernel void kernel_dense_mul_mv_q6_K_f32(
        constant ds4_dense_mvq_args & a [[buffer(0)]],
        device const char  * W   [[buffer(1)]],
        device const float * x   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= a.out_dim) return;
    const uint nblk = a.in_dim / 256u;
    const uint BLK = 210u;
    float acc = 0.0f;
    for (uint bi = 0; bi < nblk; bi++) {
        device const char * blk = W + (ulong)(row * nblk + bi) * BLK;
        device const uchar * ql = (device const uchar *)(blk);
        device const uchar * qh = (device const uchar *)(blk + 128);
        device const char  * sc = (device const char  *)(blk + 192);
        const float d = (float)(*(device const half *)(blk + 208));
        device const float * xb = x + (ulong)bi * 256u;
        for (uint n = 0; n < 256u; n += 128u) {
            for (uint l = 0; l < 32u; ++l) {
                const uint is = l / 16u;
                const int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += d * (float)sc[is + 0] * (float)q1 * xb[n + l +  0];
                acc += d * (float)sc[is + 2] * (float)q2 * xb[n + l + 32];
                acc += d * (float)sc[is + 4] * (float)q3 * xb[n + l + 64];
                acc += d * (float)sc[is + 6] * (float)q4 * xb[n + l + 96];
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    out[row] = acc;
}

// ---- Dense Q4_K matvec (Fase: Q4_K dense support) --------------------------
// Canonical GGML Q4_K dequant inline + matvec. Block = 144 bytes:
// d(f16) dmin(f16) scales[12] qs[128]. in_dim must be a multiple of 256.
static void ds4_get_scale_min_k4(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}
kernel void kernel_dense_mul_mv_q4_K_f32(
        constant ds4_dense_mvq_args & a [[buffer(0)]],
        device const char  * W   [[buffer(1)]],
        device const float * x   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= a.out_dim) return;
    const uint nblk = a.in_dim / 256u;
    const uint BLK = 144u;
    float acc = 0.0f;
    for (uint bi = 0; bi < nblk; bi++) {
        device const char * blk = W + (ulong)(row * nblk + bi) * BLK;
        const float d   = (float)(*(device const half *)(blk + 0));
        const float dmn = (float)(*(device const half *)(blk + 2));
        device const uchar * scales = (device const uchar *)(blk + 4);
        device const uchar * q = (device const uchar *)(blk + 16);
        device const float * xb = x + (ulong)bi * 256u;
        uint xi = 0; int is = 0;
        for (uint j = 0; j < 256u; j += 64u) {
            uchar sc, m;
            ds4_get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            ds4_get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            for (uint l = 0; l < 32u; ++l) acc += (d1 * (float)(q[l] & 0xF) - m1) * xb[xi++];
            for (uint l = 0; l < 32u; ++l) acc += (d2 * (float)(q[l] >>  4) - m2) * xb[xi++];
            q += 32; is += 2;
        }
    }
    out[row] = acc;
}

// ---- Dense Q5_K matvec ------------------------------------------------------
// Block = 176 bytes: d(f16) dmin(f16) scales[12] qh[32] qs[128]. Reuses
// ds4_get_scale_min_k4; high bit comes from qh. in_dim multiple of 256.
kernel void kernel_dense_mul_mv_q5_K_f32(
        constant ds4_dense_mvq_args & a [[buffer(0)]],
        device const char  * W   [[buffer(1)]],
        device const float * x   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= a.out_dim) return;
    const uint nblk = a.in_dim / 256u;
    const uint BLK = 176u;
    float acc = 0.0f;
    for (uint bi = 0; bi < nblk; bi++) {
        device const char * blk = W + (ulong)(row * nblk + bi) * BLK;
        const float d   = (float)(*(device const half *)(blk + 0));
        const float dmn = (float)(*(device const half *)(blk + 2));
        device const uchar * scales = (device const uchar *)(blk + 4);
        device const uchar * qh = (device const uchar *)(blk + 16);
        device const uchar * ql = (device const uchar *)(blk + 48);
        device const float * xb = x + (ulong)bi * 256u;
        uint xi = 0; int is = 0; uchar u1 = 1, u2 = 2;
        for (uint j = 0; j < 256u; j += 64u) {
            uchar sc, m;
            ds4_get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * (float)sc, m1 = dmn * (float)m;
            ds4_get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * (float)sc, m2 = dmn * (float)m;
            for (uint l = 0; l < 32u; ++l) acc += (d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1) * xb[xi++];
            for (uint l = 0; l < 32u; ++l) acc += (d2 * (float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2) * xb[xi++];
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
    out[row] = acc;
}

// ---- Dense Q3_K matvec ------------------------------------------------------
// Block = 110 bytes: hmask[32] qs[64] scales[12] d(f16). Canonical GGML Q3_K
// dequant with the packed 6-bit scale unpack. in_dim multiple of 256.
kernel void kernel_dense_mul_mv_q3_K_f32(
        constant ds4_dense_mvq_args & a [[buffer(0)]],
        device const char  * W   [[buffer(1)]],
        device const float * x   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= a.out_dim) return;
    const uint nblk = a.in_dim / 256u;
    const uint BLK = 110u;
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    float acc = 0.0f;
    for (uint bi = 0; bi < nblk; bi++) {
        device const char * blk = W + (ulong)(row * nblk + bi) * BLK;
        device const uchar * hm = (device const uchar *)(blk + 0);
        device const uchar * qbase = (device const uchar *)(blk + 32);
        device const uchar * sb = (device const uchar *)(blk + 96);
        const float d_all = (float)(*(device const half *)(blk + 108));
        device const float * xb = x + (ulong)bi * 256u;

        /* unpack 12 scale bytes -> 16 signed 6-bit scales */
        uint a0 = (uint)sb[0] | ((uint)sb[1]<<8) | ((uint)sb[2]<<16) | ((uint)sb[3]<<24);
        uint a1 = (uint)sb[4] | ((uint)sb[5]<<8) | ((uint)sb[6]<<16) | ((uint)sb[7]<<24);
        uint a2 = (uint)sb[8] | ((uint)sb[9]<<8) | ((uint)sb[10]<<16) | ((uint)sb[11]<<24);
        uint tmp = a2;
        uint A2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        uint A3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        uint A0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        uint A1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        char sc8[16];
        uint av[4] = { A0, A1, A2, A3 };
        for (int k = 0; k < 16; k++) sc8[k] = (char)((av[k>>2] >> (8*(k&3))) & 0xFF);

        uint xi = 0; int is = 0; uchar m = 1;
        for (uint n = 0; n < 256u; n += 128u) {
            device const uchar * q = qbase + (n/128u)*32u;
            uint shift = 0;
            for (int jj = 0; jj < 4; ++jj) {
                float dl = d_all * (float)((int)sc8[is++] - 32);
                for (uint l = 0; l < 16u; ++l)
                    acc += dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)) * xb[xi++];
                dl = d_all * (float)((int)sc8[is++] - 32);
                for (uint l = 0; l < 16u; ++l)
                    acc += dl * (float)((int)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4)) * xb[xi++];
                shift += 2; m <<= 1;
            }
        }
    }
    out[row] = acc;
}

// ---- Dense Q2_K matvec ------------------------------------------------------
// Block = 84 bytes: scales[16] qs[64] d(f16) dmin(f16). Canonical GGML Q2_K
// dequant inline. in_dim multiple of 256.
kernel void kernel_dense_mul_mv_q2_K_f32(
        constant ds4_dense_mvq_args & a [[buffer(0)]],
        device const char  * W   [[buffer(1)]],
        device const float * x   [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= a.out_dim) return;
    const uint nblk = a.in_dim / 256u;
    const uint BLK = 84u;
    float acc = 0.0f;
    for (uint bi = 0; bi < nblk; bi++) {
        device const char * blk = W + (ulong)(row * nblk + bi) * BLK;
        device const uchar * scales = (device const uchar *)(blk + 0);
        device const uchar * qbase = (device const uchar *)(blk + 16);
        const float d   = (float)(*(device const half *)(blk + 80));
        const float dmn = (float)(*(device const half *)(blk + 82));
        device const float * xb = x + (ulong)bi * 256u;
        uint xi = 0; int is = 0;
        for (uint n = 0; n < 256u; n += 128u) {
            device const uchar * q = qbase + (n/128u)*32u;
            uint shift = 0;
            for (int jj = 0; jj < 4; ++jj) {
                uchar sc = scales[is++];
                float dl = d * (float)(sc & 0xF), ml = dmn * (float)(sc >> 4);
                for (uint l = 0; l < 16u; ++l) acc += (dl * (float)((q[l] >> shift) & 3) - ml) * xb[xi++];
                sc = scales[is++];
                dl = d * (float)(sc & 0xF); ml = dmn * (float)(sc >> 4);
                for (uint l = 0; l < 16u; ++l) acc += (dl * (float)((q[l+16] >> shift) & 3) - ml) * xb[xi++];
                shift += 2;
            }
        }
    }
    out[row] = acc;
}

// ===========================================================================
// Optimized simdgroup K-quant matvec kernels (Fase opt step 1).
// Faithful ports of llama.cpp/ggml kernel_mul_mv_q4_K_f32 and _q6_K_f32
// (ggml-metal.metal), adapted to ds4's contiguous per-row block layout and
// the {in_dim,out_dim} arg struct. nsg=2 simdgroups/threadgroup, nr0=2
// output rows/simdgroup; reduction via simd_sum. Dispatch with
// threadsPerThreadgroup=(32,2,1), threadgroups=ceil(out_dim/4).
// Block structs use dense_-prefixed names (dense.metal precedes moe.metal in
// the concatenated source, which defines its own block_q4_K/block_q6_K).
// ===========================================================================
struct dense_block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct dense_block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };

kernel void kernel_dense_mul_mv_q4_K_f32_sg(
        constant ds4_dense_mvq_args & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0..3
    const short it = tiisg%8;  // 0..7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0..3

    const int  nb   = (int)(args.in_dim/256u);
    const uint nb01 = args.in_dim/256u * 144u;   // bytes/row

    const int first_row = ((int)tgpig.x * NSG + (int)sgitg) * nr0;

    device const dense_block_q4_K * x =
        (device const dense_block_q4_K *)(src0 + (uint64_t)first_row * nb01);
    device const float * y = src1;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.f, 0.f};

    device const float * y4 = y + ix*256 + 64*iq + 8*ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16*iq + 4*ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * (1.f/16.f) +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * (1.f/16.f)) -
                         dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);

            q1 += nb01/2;
            sc += nb01/2;
            dh += nb01/2;
        }
        y4 += 4 * 256;
    }

    for (int row = 0; row < nr0 && first_row + row < (int)args.out_dim; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = s;
    }
}

struct dense_block_q2_K { uchar scales[16]; uchar qs[64]; half d; half dmin; };
struct dense_block_q3_K { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; };
struct dense_block_q5_K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; };

// Port of ggml kernel_mul_mv_q2_K_f32 (nsg=2, nr0=4).
kernel void kernel_dense_mul_mv_q2_K_f32_sg(
        constant ds4_dense_mvq_args & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 4;
    const int  nb   = (int)(args.in_dim/256u);
    const uint nb01 = args.in_dim/256u * 84u;
    const int  first_row = ((int)tgpig.x * NSG + (int)sgitg) * nr0;
    device const dense_block_q2_K * x =
        (device const dense_block_q2_K *)(src0 + (uint64_t)first_row * nb01);
    device const float * y = src1;

    float yl[32];
    float sumf[4] = {0.f,0.f,0.f,0.f};
    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;
    device const float * y4 = y + ix*256 + 128*iq + 8*ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f,0.f,0.f,0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }
        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16*iq + 4*ir;
        device const half     * dh = &x[ib].d;
        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * (1.f/16.f);
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * (1.f/ 1.f) +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * (1.f/ 4.f) +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * (1.f/16.f) +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * (1.f/64.f)) -
                         dmin * (sumy[0]*(sc[0]&0xF0) + sumy[1]*(sc[2]&0xF0) + sumy[2]*(sc[4]&0xF0) + sumy[3]*(sc[6]&0xF0));
            qs += nb01/2;
            sc += nb01;
            dh += nb01/2;
        }
        y4 += 4 * 256;
    }
    for (int row = 0; row < nr0 && first_row + row < (int)args.out_dim; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = s;
    }
}

// Port of ggml kernel_mul_mv_q3_K_f32 (nsg=2, nr0=2).
kernel void kernel_dense_mul_mv_q3_K_f32_sg(
        constant ds4_dense_mvq_args & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const int  nb   = (int)(args.in_dim/256u);
    const uint nb01 = args.in_dim/256u * 110u;
    const int  first_row = ((int)tgpig.x * NSG + (int)sgitg) * nr0;
    device const dense_block_q3_K * x =
        (device const dense_block_q3_K *)(src0 + (uint64_t)first_row * nb01);
    device const float * yy = src1;

    float yl[32];
    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;
    const short il  = 2*((tid%4)/2);
    const short ir  = tid%2;
    const short l0  = 8*ir;
    const ushort4 mm[4] = {{0x0001,0x0100,0x0002,0x0200},
                           {0x0004,0x0400,0x0008,0x0800},
                           {0x0010,0x1000,0x0020,0x2000},
                           {0x0040,0x4000,0x0080,0x8000}};
    const int4 qm[2] = {{0x0003,0x0300,0x000c,0x0c00},{0x0030,0x3000,0x00c0,0xc000}};
    const ushort4 hm = mm[2*ip + il/2];
    const short shift = 2*il;
    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;
    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;
    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;
    device const float * y1 = yy + ix*256 + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f,0.f};
    float sumf2[2] = {0.f,0.f};
    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }
        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;
        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];
            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
            float s1=0,s2=0,s3=0,s4=0,s5=0,s6=0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qv = q[l/2];
                s1 += yl[l+0] * (qv & qm[il/2][0]);
                s2 += yl[l+1] * (qv & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qv & qm[il/2][2]);
                s5 += yl[l+17] * (qv & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);
            s1=s2=s3=s4=s5=s6=0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qv = q[l/2+8];
                s1 += yl[l+8] * (qv & qm[il/2][0]);
                s2 += yl[l+9] * (qv & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qv & qm[il/2][2]);
                s5 += yl[l+25] * (qv & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);
            q  += nb01/2;
            h  += nb01/2;
            a  += nb01/2;
            dh += nb01/2;
        }
        y1 += 4 * 256;
    }
    for (int row = 0; row < nr0; ++row) {
        const float sf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sf);
    }
    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < (int)args.out_dim; ++row)
            dst[first_row + row] = sumf1[row];
    }
}

// Port of ggml kernel_mul_mv_q5_K_f32 (nsg=2, nr0=1).
kernel void kernel_dense_mul_mv_q5_K_f32_sg(
        constant ds4_dense_mvq_args & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 1;
    const int  nb   = (int)(args.in_dim/256u);
    const uint nb01 = args.in_dim/256u * 176u;
    const int  first_row = ((int)tgpig.x * NSG + (int)sgitg) * nr0;
    device const dense_block_q5_K * x =
        (device const dense_block_q5_K *)(src0 + (uint64_t)first_row * nb01);
    device const float * yy = src1;

    float sumf[1] = {0.f};
    float yl[16], yh[16];
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;
    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;
    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;
    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
    device const float * y1 = yy + ix*256 + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f,0.f,0.f,0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }
        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f,0.f,0.f,0.f};
            float4 acc2 = {0.f,0.f,0.f,0.f};
            for (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            sumf[row] += dh[0] * (sc8[0] * (acc1[0]       + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]*(1.f/16.f) + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]       + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]*(1.f/16.f) + 16.f*acc2[3])) -
                         dh[1] * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
            q1 += nb01;
            qh += nb01;
            dh += nb01/2;
            a  += nb01/2;
        }
        y1 += 4 * 256;
    }
    for (int row = 0; row < nr0 && first_row + row < (int)args.out_dim; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}

kernel void kernel_dense_mul_mv_q6_K_f32_sg(
        constant ds4_dense_mvq_args & args [[buffer(0)]],
        device const char  * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;

    const int  nb   = (int)(args.in_dim/256u);
    const uint nb01 = args.in_dim/256u * 210u;   // bytes/row

    const int first_row = ((int)tgpig.x * NSG + (int)sgitg) * nr0;

    device const dense_block_q6_K * x =
        (device const dense_block_q6_K *)(src0 + (uint64_t)first_row * nb01);
    device const float * yy = src1;

    float sumf[2] = {0.f, 0.f};
    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;     // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const char    * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i*256 + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4))) - 32);
                sums[1] += yl[4*l + 1] * ((int)((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2))) - 32);
                sums[2] += yl[4*l + 2] * ((int)((char)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0))) - 32);
                sums[3] += yl[4*l + 3] * ((int)((char)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2))) - 32);
            }
            sumf[row] += dh[0] * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);

            q1 += nb01;
            q2 += nb01;
            qh += nb01;
            sc += nb01;
            dh += nb01/2;
        }
    }

    for (int row = 0; row < nr0 && first_row + row < (int)args.out_dim; ++row) {
        const float s = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = s;
    }
}

/* ---- qwen3_next building blocks ------------------------------------------- *
 * Causal depthwise 1-D convolution (ggml_ssm_conv semantics). The Gated-DeltaNet
 * layer of qwen3_next runs the projected q/k/v through this short conv before the
 * delta-rule. Per channel c and output token t:
 *     out[t, c] = sum_{k=0..K-1} sx[t+k, c] * w[k, c]
 * sx carries the (K-1)-token conv history prepended, so it has T+K-1 rows; the
 * window is therefore causal. Row-major: sx[(T+K-1), C], w[K, C], out[T, C].
 * Pure conv only — the SiLU that follows in the graph is a separate kernel. */
struct dense_ssm_conv_args { uint C; uint K; uint T; };
kernel void kernel_dense_ssm_conv_f32(
        constant dense_ssm_conv_args & a [[buffer(0)]],
        device const float * sx  [[buffer(1)]],   /* [(T+K-1), C] row-major */
        device const float * w   [[buffer(2)]],   /* [K, C]       row-major */
        device       float * out [[buffer(3)]],   /* [T, C]       row-major */
        uint gid [[thread_position_in_grid]]) {
    const uint total = a.C * a.T;
    if (gid >= total) return;
    const uint t = gid / a.C;
    const uint c = gid % a.C;
    float s = 0.0f;
    for (uint k = 0; k < a.K; ++k) {
        s += sx[(t + k) * a.C + c] * w[k * a.C + c];
    }
    out[t * a.C + c] = s;
}

/* Inclusive prefix-sum along the contiguous axis (ggml_cumsum semantics). The
 * chunkwise gated delta rule cumsum-s the (log-)decay along each chunk. Layout is
 * row-major [R, N]: R independent rows of length N, out[r,i] = Σ_{j<=i} in[r,i].
 * One thread per row (N = chunk length, small) — reference, correctness-first. */
struct dense_cumsum_args { uint N; uint R; };
kernel void kernel_dense_cumsum_f32(
        constant dense_cumsum_args & a [[buffer(0)]],
        device const float * in  [[buffer(1)]],   /* [R, N] row-major */
        device       float * out [[buffer(2)]],   /* [R, N] row-major */
        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.R) return;
    device const float * ir = in  + (size_t)gid * a.N;
    device       float * orr = out + (size_t)gid * a.N;
    float acc = 0.0f;
    for (uint i = 0; i < a.N; ++i) { acc += ir[i]; orr[i] = acc; }
}

/* Per-head RMSNorm with a shared weight (qwen3_next full-attn q/k norm). Input is
 * [n_head, head_dim] row-major; each head is normalized independently over head_dim
 * with weight w[head_dim]. One thread per head — reference, correctness-first. */
struct dense_head_rms_args { uint n_head; uint head_dim; float eps; };
kernel void kernel_dense_head_rms_norm_f32(
        constant dense_head_rms_args & a [[buffer(0)]],
        device const float * x   [[buffer(1)]],   /* [n_head, head_dim] */
        device const float * w   [[buffer(2)]],   /* [head_dim] */
        device       float * out [[buffer(3)]],   /* [n_head, head_dim] */
        uint h [[thread_position_in_grid]]) {
    if (h >= a.n_head) return;
    device const float * xh = x   + (size_t)h * a.head_dim;
    device       float * oh = out + (size_t)h * a.head_dim;
    float ss = 0.0f;
    for (uint i = 0; i < a.head_dim; ++i) ss += xh[i] * xh[i];
    const float scale = 1.0f / sqrt(ss / (float)a.head_dim + a.eps);
    for (uint i = 0; i < a.head_dim; ++i) oh[i] = xh[i] * scale * w[i];
}

/* Sigmoid output gate: out[i] = x[i] * sigmoid(gate[i]). qwen3_next gates both the
 * full-attn output and (via SiLU, separate kernel) the DeltaNet output. */
struct dense_sigmoid_gate_args { uint n; };
kernel void kernel_dense_sigmoid_gate_f32(
        constant dense_sigmoid_gate_args & a [[buffer(0)]],
        device const float * x    [[buffer(1)]],
        device const float * gate [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n) return;
    out[gid] = x[gid] * (1.0f / (1.0f + exp(-gate[gid])));
}
