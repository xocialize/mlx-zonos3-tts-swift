// Copyright © 2025 Apple Inc.
// SPDX-License-Identifier: MIT
//
// The scalar-gate, unmasked gated-delta Metal kernel of mlx-lm (mlx_lm/models/gated_delta.py,
// _make_gated_delta_kernel(has_mask=False, vectorized=False)), as adapted for the ZONOS3 GDN layers in Zyphra's
// zonos3_mlx/models/gdn_kernel.py (MIT) and carried over here unchanged: q, k [B, T, Hk, Dk], v [B, T, Hv, Dv],
// g / beta [B, T, Hv], state [B, Hv, Dv, Dk], all float32. q arrives pre-scaled by Dk**-0.5.
// The kernel runs multi-row calls (prefill); one row runs the ops step in Zonos3LM.swift.
import Foundation
import MLX
import MLXFast

enum GatedDeltaKernel {
    static let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "zonos3_gated_delta_step", inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"], source: source)

    /// q, k, v [T, H, D]; g, beta [T, H]; state [H, Dv, Dk] -> (y [T, H, Dv], state).
    static func run(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ g: MLXArray, _ beta: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) {
        let T = k.shape[0], Hk = k.shape[1], Dk = k.shape[2], Hv = v.shape[1], Dv = v.shape[2]
        let out = kernel(
            [q.expandedDimensions(axis: 0), k.expandedDimensions(axis: 0), v.expandedDimensions(axis: 0),
             g.expandedDimensions(axis: 0), beta.expandedDimensions(axis: 0), state.expandedDimensions(axis: 0), T],
            template: [("InT", DType.float32), ("StT", DType.float32), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid: (32, Dv, Hv), threadGroup: (32, 4, 1),
            outputShapes: [[1, T, Hv, Dv], [1, Hv, Dv, Dk]], outputDTypes: [.float32, .float32])
        return (out[0].squeezed(axis: 0), out[1].squeezed(axis: 0))
    }
}
