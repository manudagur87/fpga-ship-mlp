`timescale 1ns / 1ps

// MLP Compute Module — Parallel Hidden Neurons (used by V2, V4)
// Both hidden neurons process the SAME feature simultaneously with
// two multipliers and two accumulators. Single DATA_RAM read port.
//
// Network: 7 inputs -> 2 hidden (sigmoid) -> 1 output (binary threshold)
//
// Cycle breakdown per sample:
//   HIDDEN_ADDR:  1 cy  (pipeline fill)
//   HIDDEN_MAC:   7 cy  (both neurons in parallel)
//   HIDDEN_BIAS:  1 cy  (both biases in parallel)
//   HIDDEN_ACT0:  1 cy  (sigmoid neuron 0)
//   HIDDEN_ACT1:  1 cy  (sigmoid neuron 1)
//   OUTPUT_MAC:   2 cy  (h[0]*w_out[1], h[1]*w_out[2])
//   OUTPUT_BIAS:  1 cy
//   OUTPUT_ACT:   1 cy
//   DONE:         1 cy
//   Total: 16 cycles

module mlp_parallel (
    input wire clk,
    input wire rst_n,
    input wire start,
    output wire done,

    // Data RAM read port (1-cycle read latency)
    output wire       data_read_en,
    output reg  [8:0] data_read_address,
    input  wire [7:0] data_read_data_out,

    // Sample base address
    input wire [8:0] sample_base_addr,

    // Hidden weight lookup — neuron 0
    output wire [3:0] w_hid_idx_0,
    input  wire [7:0] w_hid_val_0,

    // Hidden weight lookup — neuron 1
    output wire [3:0] w_hid_idx_1,
    input  wire [7:0] w_hid_val_1,

    // Output weight lookup
    output wire [1:0] w_out_idx,
    input  wire [7:0] w_out_val,

    // Sigmoid LUT lookup
    output wire [7:0] sig_lut_idx,
    input  wire [7:0] sig_lut_val,

    // Result (valid when done is asserted)
    output reg [7:0] result
);

    // ----------------------------------------------------------------
    // FSM states
    // ----------------------------------------------------------------
    localparam IDLE        = 4'd0;
    localparam HIDDEN_ADDR = 4'd1;
    localparam HIDDEN_MAC  = 4'd2;   // both neurons accumulate in parallel
    localparam HIDDEN_BIAS = 4'd3;   // both biases in parallel
    localparam HIDDEN_ACT0 = 4'd4;   // sigmoid for neuron 0
    localparam HIDDEN_ACT1 = 4'd5;   // sigmoid for neuron 1
    localparam OUTPUT_MAC  = 4'd6;
    localparam OUTPUT_BIAS = 4'd7;
    localparam OUTPUT_ACT  = 4'd8;
    localparam DONE_STATE  = 4'd9;

    (* fsm_encoding = "none" *) reg [3:0] state;

    // ----------------------------------------------------------------
    // Counters & storage
    // ----------------------------------------------------------------
    reg [2:0] feat_idx;
    reg       out_idx;
    (* DONT_TOUCH = "true" *) reg signed [19:0] acc0;  // neuron 0 accumulator
    (* DONT_TOUCH = "true" *) reg signed [19:0] acc1;  // neuron 1 accumulator
    (* DONT_TOUCH = "true" *) reg signed [19:0] acc_out; // output accumulator
    (* DONT_TOUCH = "true" *) reg [7:0] h [0:1];

    // ----------------------------------------------------------------
    // Combinational weight-index generation
    // ----------------------------------------------------------------
    //   Neuron 0: w_hid[(feat_idx+1)*2 + 0] = {feat_idx+1, 1'b0}
    //   Neuron 1: w_hid[(feat_idx+1)*2 + 1] = {feat_idx+1, 1'b1}
    //   Bias 0:   w_hid[0]
    //   Bias 1:   w_hid[1]
    reg [3:0] w_hid_idx_0_c, w_hid_idx_1_c;
    always @(*) begin
        case (state)
            HIDDEN_MAC: begin
                w_hid_idx_0_c = {feat_idx + 3'd1, 1'b0};
                w_hid_idx_1_c = {feat_idx + 3'd1, 1'b1};
            end
            HIDDEN_BIAS: begin
                w_hid_idx_0_c = 4'd0;
                w_hid_idx_1_c = 4'd1;
            end
            default: begin
                w_hid_idx_0_c = 4'd0;
                w_hid_idx_1_c = 4'd1;
            end
        endcase
    end
    assign w_hid_idx_0 = w_hid_idx_0_c;
    assign w_hid_idx_1 = w_hid_idx_1_c;

    reg [1:0] w_out_idx_c;
    always @(*) begin
        case (state)
            OUTPUT_MAC:  w_out_idx_c = {1'b0, out_idx} + 2'd1;
            OUTPUT_BIAS: w_out_idx_c = 2'd0;
            default:     w_out_idx_c = 2'd0;
        endcase
    end
    assign w_out_idx = w_out_idx_c;

    // ----------------------------------------------------------------
    // Sigmoid index — uses acc0 during HIDDEN_ACT0, acc1 during HIDDEN_ACT1
    // ----------------------------------------------------------------
    wire signed [19:0] sig_acc = (state == HIDDEN_ACT1) ? acc1 : acc0;
    wire signed [19:0] sig_shifted = sig_acc >>> 5;
    wire signed [19:0] sig_raw    = sig_shifted + 20'sd128;
    assign sig_lut_idx = (sig_raw < 20'sd0)   ? 8'd0   :
                         (sig_raw > 20'sd255)  ? 8'd255 :
                         sig_raw[7:0];

    // ----------------------------------------------------------------
    // RAM read enable
    // ----------------------------------------------------------------
    assign data_read_en = (state == IDLE && start) ||
                          (state == HIDDEN_ADDR)   ||
                          (state == HIDDEN_MAC);

    // ----------------------------------------------------------------
    // Done
    // ----------------------------------------------------------------
    assign done = (state == DONE_STATE);

    // ----------------------------------------------------------------
    // Signed weight wires
    // ----------------------------------------------------------------
    wire signed [7:0] w_hid_s0 = $signed(w_hid_val_0);
    wire signed [7:0] w_hid_s1 = $signed(w_hid_val_1);
    wire signed [7:0] w_out_s  = $signed(w_out_val);

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
            result            <= 8'd0;
            h[0]              <= 8'd0;
            h[1]              <= 8'd0;
            data_read_address <= 9'd0;
        end else begin
            case (state)

                IDLE: begin
                    if (start) begin
                        feat_idx          <= 3'd0;
                        acc0              <= 20'sd0;
                        acc1              <= 20'sd0;
                        data_read_address <= sample_base_addr;
                        state             <= HIDDEN_ADDR;
                    end
                end

                // Pipeline fill
                HIDDEN_ADDR: begin
                    data_read_address <= sample_base_addr + 9'd1;
                    state             <= HIDDEN_MAC;
                end

                // Both neurons accumulate same feature, different weights
                HIDDEN_MAC: begin
                    acc0 <= acc0 + w_hid_s0 * $signed({1'b0, data_read_data_out});
                    acc1 <= acc1 + w_hid_s1 * $signed({1'b0, data_read_data_out});

                    if (feat_idx == 3'd6) begin
                        state <= HIDDEN_BIAS;
                    end else begin
                        feat_idx          <= feat_idx + 3'd1;
                        data_read_address <= sample_base_addr + {6'd0, feat_idx} + 9'd2;
                    end
                end

                // Both biases in parallel
                HIDDEN_BIAS: begin
                    acc0  <= acc0 + w_hid_s0 * $signed(9'd255);
                    acc1  <= acc1 + w_hid_s1 * $signed(9'd255);
                    state <= HIDDEN_ACT0;
                end

                // Sigmoid neuron 0 (sig_acc = acc0)
                HIDDEN_ACT0: begin
                    h[0]  <= sig_lut_val;
                    state <= HIDDEN_ACT1;
                end

                // Sigmoid neuron 1 (sig_acc = acc1)
                HIDDEN_ACT1: begin
                    h[1]    <= sig_lut_val;
                    acc_out <= 20'sd0;
                    out_idx <= 1'b0;
                    state   <= OUTPUT_MAC;
                end

                // Output layer MAC
                OUTPUT_MAC: begin
                    acc_out <= acc_out + w_out_s * $signed({1'b0, h[out_idx]});
                    if (out_idx == 1'b1) begin
                        state <= OUTPUT_BIAS;
                    end else begin
                        out_idx <= 1'b1;
                    end
                end

                // Output bias
                OUTPUT_BIAS: begin
                    acc_out <= acc_out + w_out_s * $signed(9'd255);
                    state   <= OUTPUT_ACT;
                end

                // Binary threshold
                OUTPUT_ACT: begin
                    result <= ($signed(acc_out) > 20'sd0) ? 8'd1 : 8'd0;
                    state  <= DONE_STATE;
                end

                DONE_STATE: begin
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
