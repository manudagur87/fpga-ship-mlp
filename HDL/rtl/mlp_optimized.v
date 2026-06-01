`timescale 1ns / 1ps

// MLP Compute Module — Optimized (12 cycles/sample)
//
// Optimizations vs mlp_parallel.v (16 cy):
//   1. Merge HIDDEN_BIAS into last HIDDEN_MAC (dedicated bias ports)
//   2. Parallel sigmoid lookup (dual LUT ports, both neurons in 1 cy)
//   3. Merge OUTPUT_BIAS into last OUTPUT_MAC (dedicated bias port)
//   4. Accept start in OUTPUT_DONE (skip IDLE between samples)
//   5. Bias*255 via shift-subtract (no DSP for constant multiply)
//
// Network: 7 inputs -> 2 hidden (sigmoid) -> 1 output (binary threshold)
//
// Cycle breakdown per sample (steady-state):
//   HIDDEN_ADDR:   1 cy   (pipeline fill)
//   HIDDEN_MAC:    7 cy   (both neurons + bias folded into last)
//   HIDDEN_ACT:    1 cy   (parallel sigmoid both neurons)
//   OUTPUT_MAC:    2 cy   (h[0]*w, h[1]*w + bias folded into last)
//   OUTPUT_DONE:   1 cy   (threshold + done, overlaps with next start)
//   Total: 12 cycles/sample

module mlp_optimized (
    input wire clk,
    input wire rst_n,
    input wire start,
    output wire done,
    output wire ready,    // high when can accept start (IDLE or OUTPUT_DONE)

    // Data RAM read port (1-cycle read latency)
    output wire       data_read_en,
    output reg  [8:0] data_read_address,
    input  wire [7:0] data_read_data_out,

    // Sample base address in circular buffer
    input wire [8:0] sample_base_addr,

    // Hidden weight lookup — neuron 0
    output wire [3:0] w_hid_idx_0,
    input  wire [7:0] w_hid_val_0,

    // Hidden weight lookup — neuron 1
    output wire [3:0] w_hid_idx_1,
    input  wire [7:0] w_hid_val_1,

    // Hidden biases (dedicated ports, bypass weight index mux)
    input wire [7:0] bias_hid_0,   // = w_hid_regs[0]
    input wire [7:0] bias_hid_1,   // = w_hid_regs[1]

    // Output weight lookup
    output wire [1:0] w_out_idx,
    input  wire [7:0] w_out_val,

    // Output bias (dedicated port)
    input wire [7:0] bias_out,     // = w_out_regs[0]

    // Sigmoid LUT — dual read ports (parallel lookup)
    output wire [7:0] sig_lut_idx_0,
    input  wire [7:0] sig_lut_val_0,
    output wire [7:0] sig_lut_idx_1,
    input  wire [7:0] sig_lut_val_1,

    // Result (combinational, valid when done asserted)
    output wire [7:0] result
);

    // ----------------------------------------------------------------
    // FSM states
    // ----------------------------------------------------------------
    localparam IDLE        = 3'd0;
    localparam HIDDEN_ADDR = 3'd1;
    localparam HIDDEN_MAC  = 3'd2;
    localparam HIDDEN_ACT  = 3'd3;
    localparam OUTPUT_MAC  = 3'd4;
    localparam OUTPUT_DONE = 3'd5;

    (* fsm_encoding = "none" *) reg [2:0] state;

    // ----------------------------------------------------------------
    // Counters & storage
    // ----------------------------------------------------------------
    reg [2:0] feat_idx;
    reg       out_idx;
    (* DONT_TOUCH = "true" *) reg signed [19:0] acc0;
    (* DONT_TOUCH = "true" *) reg signed [19:0] acc1;
    (* DONT_TOUCH = "true" *) reg signed [19:0] acc_out;
    reg [7:0] h [0:1];

    // ----------------------------------------------------------------
    // Signed weight wires
    // ----------------------------------------------------------------
    wire signed [7:0] w_hid_s0 = $signed(w_hid_val_0);
    wire signed [7:0] w_hid_s1 = $signed(w_hid_val_1);
    wire signed [7:0] w_out_s  = $signed(w_out_val);

    // ----------------------------------------------------------------
    // Bias * 255 via shift-subtract: (bias << 8) - bias
    // Avoids DSP usage for constant multiply
    // ----------------------------------------------------------------
    wire signed [19:0] b0_ext       = {{12{bias_hid_0[7]}}, bias_hid_0};
    wire signed [19:0] bias0_x255   = (b0_ext <<< 8) - b0_ext;

    wire signed [19:0] b1_ext       = {{12{bias_hid_1[7]}}, bias_hid_1};
    wire signed [19:0] bias1_x255   = (b1_ext <<< 8) - b1_ext;

    wire signed [19:0] bout_ext     = {{12{bias_out[7]}}, bias_out};
    wire signed [19:0] bias_out_x255 = (bout_ext <<< 8) - bout_ext;

    wire last_hidden_mac = (feat_idx == 3'd6);

    // ----------------------------------------------------------------
    // Weight index generation
    // ----------------------------------------------------------------
    // Hidden weights: neuron 0 = w_hid[(feat+1)*2], neuron 1 = w_hid[(feat+1)*2+1]
    reg [3:0] w_hid_idx_0_c, w_hid_idx_1_c;
    always @(*) begin
        case (state)
            HIDDEN_MAC: begin
                w_hid_idx_0_c = {feat_idx + 3'd1, 1'b0};
                w_hid_idx_1_c = {feat_idx + 3'd1, 1'b1};
            end
            default: begin
                w_hid_idx_0_c = 4'd0;
                w_hid_idx_1_c = 4'd1;
            end
        endcase
    end
    assign w_hid_idx_0 = w_hid_idx_0_c;
    assign w_hid_idx_1 = w_hid_idx_1_c;

    // Output weights: w_out[out_idx + 1]
    reg [1:0] w_out_idx_c;
    always @(*) begin
        case (state)
            OUTPUT_MAC:  w_out_idx_c = {1'b0, out_idx} + 2'd1;
            default:     w_out_idx_c = 2'd0;
        endcase
    end
    assign w_out_idx = w_out_idx_c;

    // ----------------------------------------------------------------
    // Dual sigmoid index — both computed combinationally from accumulators
    // sig_idx = clamp((acc >> 5) + 128, 0, 255)
    // ----------------------------------------------------------------
    wire signed [19:0] sig_shifted_0 = acc0 >>> 5;
    wire signed [19:0] sig_raw_0     = sig_shifted_0 + 20'sd128;
    assign sig_lut_idx_0 = (sig_raw_0 < 20'sd0)   ? 8'd0   :
                           (sig_raw_0 > 20'sd255)  ? 8'd255 :
                           sig_raw_0[7:0];

    wire signed [19:0] sig_shifted_1 = acc1 >>> 5;
    wire signed [19:0] sig_raw_1     = sig_shifted_1 + 20'sd128;
    assign sig_lut_idx_1 = (sig_raw_1 < 20'sd0)   ? 8'd0   :
                           (sig_raw_1 > 20'sd255)  ? 8'd255 :
                           sig_raw_1[7:0];

    // ----------------------------------------------------------------
    // RAM read enable — active during address setup and MAC phases
    // ----------------------------------------------------------------
    assign data_read_en = ((state == IDLE || state == OUTPUT_DONE) && start) ||
                          (state == HIDDEN_ADDR) ||
                          (state == HIDDEN_MAC);

    // ----------------------------------------------------------------
    // Control outputs
    // ----------------------------------------------------------------
    assign done  = (state == OUTPUT_DONE);
    assign ready = (state == IDLE) || (state == OUTPUT_DONE);

    // Combinational binary threshold (valid when done asserted)
    assign result = ($signed(acc_out) > 20'sd0) ? 8'd1 : 8'd0;

    // ----------------------------------------------------------------
    // Main FSM
    // ----------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state             <= IDLE;
            acc0              <= 20'sd0;
            acc1              <= 20'sd0;
            acc_out           <= 20'sd0;
            feat_idx          <= 3'd0;
            out_idx           <= 1'b0;
            h[0]              <= 8'd0;
            h[1]              <= 8'd0;
            data_read_address <= 9'd0;
        end else begin
            case (state)

                // ---- Wait for start ----
                IDLE: begin
                    if (start) begin
                        feat_idx          <= 3'd0;
                        acc0              <= 20'sd0;
                        acc1              <= 20'sd0;
                        data_read_address <= sample_base_addr;
                        state             <= HIDDEN_ADDR;
                    end
                end

                // ---- Pipeline fill: RAM latency ----
                HIDDEN_ADDR: begin
                    data_read_address <= sample_base_addr + 9'd1;
                    state             <= HIDDEN_MAC;
                end

                // ---- Hidden layer MAC (7 iterations, bias folded into last) ----
                HIDDEN_MAC: begin
                    // Both neurons accumulate same feature, different weights
                    // On last iteration (feat_idx==6), also add bias*255
                    acc0 <= acc0 + w_hid_s0 * $signed({1'b0, data_read_data_out})
                                 + (last_hidden_mac ? bias0_x255 : 20'sd0);
                    acc1 <= acc1 + w_hid_s1 * $signed({1'b0, data_read_data_out})
                                 + (last_hidden_mac ? bias1_x255 : 20'sd0);

                    if (last_hidden_mac) begin
                        state <= HIDDEN_ACT;
                    end else begin
                        feat_idx          <= feat_idx + 3'd1;
                        data_read_address <= sample_base_addr + {6'd0, feat_idx} + 9'd2;
                    end
                end

                // ---- Parallel sigmoid activation (both neurons, 1 cycle) ----
                HIDDEN_ACT: begin
                    h[0]    <= sig_lut_val_0;   // port 0: sigmoid(acc0)
                    h[1]    <= sig_lut_val_1;   // port 1: sigmoid(acc1)
                    acc_out <= 20'sd0;
                    out_idx <= 1'b0;
                    state   <= OUTPUT_MAC;
                end

                // ---- Output layer MAC (2 iterations, bias folded into last) ----
                OUTPUT_MAC: begin
                    acc_out <= acc_out + w_out_s * $signed({1'b0, h[out_idx]})
                             + ((out_idx == 1'b1) ? bias_out_x255 : 20'sd0);

                    if (out_idx == 1'b1)
                        state <= OUTPUT_DONE;
                    else
                        out_idx <= 1'b1;
                end

                // ---- Done: result valid, can accept next start immediately ----
                OUTPUT_DONE: begin
                    if (start) begin
                        // Start next sample without returning to IDLE
                        feat_idx          <= 3'd0;
                        acc0              <= 20'sd0;
                        acc1              <= 20'sd0;
                        data_read_address <= sample_base_addr;
                        state             <= HIDDEN_ADDR;
                    end else begin
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
