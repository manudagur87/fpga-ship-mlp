/*
----------------------------------------------------------------------------------
--  MLP Inference Core — Approach A, Packed (signed-weight)
--  Version 2 — Optimized with pipeline/unroll pragmas
--  Weights/LUT via DMA (AXI-Stream), 4 values packed per 32-bit word for features
--  Network: 7 inputs -> 2 hidden (sigmoid) -> 1 output (linear, binary threshold)
--  Fixed-point: 0.8 unsigned for features/activations, signed int8 for weights
--
--  Protocol:
--    First word = mode:
--      0x00 = weight-load phase:
--             next 19 words = weights (unpacked, one per word — only 19 values)
--             next 256 words = LUT entries (unpacked, one per word)
--      0x01 = inference phase:
--             next word = batch_size N
--             next N*2 words = packed features (4 per word, 2 words per sample)
--
--  Output packing: 4 results per 32-bit word
----------------------------------------------------------------------------------
*/

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"

#define N_INPUTS    7
#define N_HIDDEN    2
#define N_OUTPUTS   1
#define LUT_SIZE    256

#define HID_SHIFT   5
#define OUT_SHIFT   7

typedef ap_axis<32,0,0,0> AXIS;

void mlp_A_packed_signed_v2(hls::stream<AXIS>& S_AXIS, hls::stream<AXIS>& M_AXIS) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS

    static int w_hid[N_INPUTS + 1][N_HIDDEN];
    static int w_out[N_HIDDEN + 1];
    static int sigmoid_lut[LUT_SIZE];

    // Partition all weight/LUT arrays into registers for parallel access
#pragma HLS ARRAY_PARTITION variable=w_hid complete dim=0
#pragma HLS ARRAY_PARTITION variable=w_out complete
#pragma HLS ARRAY_PARTITION variable=sigmoid_lut complete

    AXIS read_input, write_output;

    // Read mode word
    read_input = S_AXIS.read();
    int mode = (int)read_input.data;

    if (mode == 0x00) {
        // Weight-load phase (unpacked — weights are only 19+256 values, no need to pack)

        mlp_load_w_hid_row: for (int r = 0; r < N_INPUTS + 1; r++) {
            mlp_load_w_hid_col: for (int c = 0; c < N_HIDDEN; c++) {
            #pragma HLS PIPELINE II=1
                read_input = S_AXIS.read();
                w_hid[r][c] = (int)read_input.data;
            }
        }

        mlp_load_w_out: for (int i = 0; i < N_HIDDEN + 1; i++) {
        #pragma HLS PIPELINE II=1
            read_input = S_AXIS.read();
            w_out[i] = (int)read_input.data;
        }

        mlp_load_lut: for (int i = 0; i < LUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
            read_input = S_AXIS.read();
            sigmoid_lut[i] = (int)read_input.data;
        }

        // Send ack
        write_output.data = 1;
        write_output.keep = 0xF;
        write_output.strb = 0xF;
        write_output.last = 1;
        M_AXIS.write(write_output);

    } else {
        // Inference phase — packed features

        read_input = S_AXIS.read();
        int batch_size = (int)read_input.data;

        int x[N_INPUTS];
        int h[N_HIDDEN];

        int out_count = 0;
        ap_uint<32> out_word = 0;
        int total_out_words = (batch_size + 3) / 4;
        int out_word_idx = 0;

        mlp_sample_loop: for (int s = 0; s < batch_size; s++) {
        #pragma HLS PIPELINE II=1

            // Read 2 packed words -> 7 features
            read_input = S_AXIS.read();
            ap_uint<32> w0 = read_input.data;
            x[0] = (w0 >>  0) & 0xFF;
            x[1] = (w0 >>  8) & 0xFF;
            x[2] = (w0 >> 16) & 0xFF;
            x[3] = (w0 >> 24) & 0xFF;

            read_input = S_AXIS.read();
            ap_uint<32> w1 = read_input.data;
            x[4] = (w1 >>  0) & 0xFF;
            x[5] = (w1 >>  8) & 0xFF;
            x[6] = (w1 >> 16) & 0xFF;

            // Hidden layer
            mlp_hidden: for (int n = 0; n < N_HIDDEN; n++) {
            #pragma HLS UNROLL
                int acc = w_hid[0][n] * 255;  // bias (row 0) * 1.0 in 0.8 format
                mlp_hidden_mac: for (int i = 0; i < N_INPUTS; i++) {
                #pragma HLS UNROLL
                    acc += x[i] * w_hid[i + 1][n];  // rows 1-7 = weights
                }
                int sig_idx = (acc >> HID_SHIFT) + 128;
                if (sig_idx < 0) sig_idx = 0;
                if (sig_idx > 255) sig_idx = 255;
                h[n] = sigmoid_lut[sig_idx];
            }

            // Output layer (linear, binary threshold)
            int acc_out = w_out[0] * 255;  // bias (row 0) * 1.0 in 0.8 format
            mlp_output_mac: for (int i = 0; i < N_HIDDEN; i++) {
            #pragma HLS UNROLL
                acc_out += h[i] * w_out[i + 1];  // rows 1-2 = weights
            }
            int y = (acc_out > 0) ? 1 : 0;

            // Pack output
            int slot = out_count % 4;
            out_word |= ((ap_uint<32>)y) << (slot * 8);
            out_count++;

            if (slot == 3 || s == batch_size - 1) {
                write_output.data = out_word;
                write_output.keep = 0xF;
                write_output.strb = 0xF;
                out_word_idx++;
                write_output.last = (out_word_idx == total_out_words) ? 1 : 0;
                M_AXIS.write(write_output);
                out_word = 0;
            }
        }
    }
}
