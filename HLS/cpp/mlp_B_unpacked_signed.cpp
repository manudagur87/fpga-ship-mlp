/*
----------------------------------------------------------------------------------
--  MLP Inference Core — Approach B, Unpacked (signed-weight)
--  Weights/LUT via AXI-Lite registers, one value per 32-bit AXI-Stream word
--  Network: 7 inputs -> 2 hidden (sigmoid) -> 1 output (linear, binary threshold)
--  Fixed-point: 0.8 unsigned for features/activations, signed int8 for weights
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

void mlp_B_unpacked_signed(
    hls::stream<AXIS>& S_AXIS,
    hls::stream<AXIS>& M_AXIS,
    int batch_size,
    int w_hid[N_INPUTS + 1][N_HIDDEN],   // 8 rows x 2 cols (row 0 = bias, rows 1-7 = weights)
    int w_out[N_HIDDEN + 1],              // 3 values (row 0 = bias, rows 1-2 = weights)
    int sigmoid_lut[LUT_SIZE]
) {
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS
#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=w_hid
#pragma HLS INTERFACE s_axilite port=w_out
#pragma HLS INTERFACE s_axilite port=sigmoid_lut
#pragma HLS INTERFACE s_axilite port=return

    AXIS read_input, write_output;
    int x[N_INPUTS];
    int h[N_HIDDEN];

    mlp_sample_loop: for (int s = 0; s < batch_size; s++) {

        // Read 7 features from stream (one per 32-bit word)
        mlp_read_features: for (int i = 0; i < N_INPUTS; i++) {
            read_input = S_AXIS.read();
            x[i] = (int)read_input.data;
        }

        // Hidden layer: 2 neurons
        mlp_hidden: for (int n = 0; n < N_HIDDEN; n++) {
            int acc = w_hid[0][n] * 255;  // bias (row 0) * 1.0 in 0.8 format
            mlp_hidden_mac: for (int i = 0; i < N_INPUTS; i++) {
                acc += x[i] * w_hid[i + 1][n];  // rows 1-7 = weights
            }
            int sig_idx = (acc >> HID_SHIFT) + 128;
            if (sig_idx < 0) sig_idx = 0;
            if (sig_idx > 255) sig_idx = 255;
            h[n] = sigmoid_lut[sig_idx];
        }

        // Output layer: 1 neuron (linear, binary threshold)
        int acc_out = w_out[0] * 255;  // bias (row 0) * 1.0 in 0.8 format
        mlp_output_mac: for (int i = 0; i < N_HIDDEN; i++) {
            acc_out += h[i] * w_out[i + 1];  // rows 1-2 = weights
        }
        int y = (acc_out > 0) ? 1 : 0;

        // Write result to output stream
        write_output.data = y;
        write_output.keep = 0xF;
        write_output.strb = 0xF;
        write_output.last = (s == batch_size - 1) ? 1 : 0;
        M_AXIS.write(write_output);
    }
}
