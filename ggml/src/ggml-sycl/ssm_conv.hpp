#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ssm_conv_silu(ggml_backend_sycl_context & ctx, ggml_tensor * conv_dst, ggml_tensor * dst);
