`timescale 1ns / 1ps

// Version 4 Wrapper — DMA Weights + Packed Data, Single MLP (parallel neurons)
//
// DMA Input (single transfer):
//   Words  0..15  : w_hid[0..15]
//   Words 16..18  : w_out[0..2]
//   Words 19..274 : sigmoid_lut[0..255]
//   Words 275+    : packed features (2 words/sample), TLAST on last
//
// DMA Output (packed): 4 results per word, ceil(N/4) words, TLAST on last
//
// Uses mlp_parallel: 16 cycles/sample, single DATA_RAM.

module myip_v1_0_v4
    (
        input  wire        ACLK,
        input  wire        ARESETN,
        output reg         S_AXIS_TREADY,
        input  wire [31:0] S_AXIS_TDATA,
        input  wire        S_AXIS_TLAST,
        input  wire        S_AXIS_TVALID,
        output reg         M_AXIS_TVALID,
        output reg  [31:0] M_AXIS_TDATA,
        output reg         M_AXIS_TLAST,
        input  wire        M_AXIS_TREADY
    );

    localparam DATA_DEPTH_BITS = 9;
    localparam RES_DEPTH_BITS  = 12;
    localparam WIDTH           = 8;
    localparam NUM_WEIGHTS     = 275;

    (* fsm_encoding = "none" *) reg [2:0] w_state;
    localparam W_IDLE         = 3'd0;
    localparam W_READ_WEIGHTS = 3'd1;
    localparam W_RECV_PACKED  = 3'd2;
    localparam W_UNPACK       = 3'd3;
    localparam W_FINISH_PROC  = 3'd4;
    localparam W_WRITE_PACKED = 3'd5;

    // ----------------------------------------------------------------
    // Weight / LUT registers
    // ----------------------------------------------------------------
    reg [7:0] w_hid_regs   [0:15];
    reg [7:0] w_out_regs   [0:2];
    reg [7:0] sig_lut_regs [0:255];

    // ----------------------------------------------------------------
    // DMA / circular buffer
    // ----------------------------------------------------------------
    reg [8:0]  weight_cnt;
    reg [DATA_DEPTH_BITS-1:0] data_wr_ptr;
    reg [11:0] samples_received, total_samples, sample_idx;
    reg        word_in_sample;
    reg        recv_done, got_tlast;
    reg [DATA_DEPTH_BITS-1:0] sample_base_addr, buf_tail;

    // Unpack
    reg [31:0] captured_word;
    reg [1:0]  unpack_cnt;
    reg [2:0]  bytes_to_write;

    wire [DATA_DEPTH_BITS-1:0] buf_used = data_wr_ptr - buf_tail;
    wire buffer_has_room = (buf_used < ((1 << DATA_DEPTH_BITS) - 7));

    // DATA_RAM write (from unpack)
    reg        data_wr_en_r;
    reg [DATA_DEPTH_BITS-1:0] data_wr_addr_r;
    reg [7:0]  data_wr_data_r;

    wire process_active = (w_state==W_RECV_PACKED)||(w_state==W_UNPACK)||(w_state==W_FINISH_PROC);

    // MLP
    wire        mlp_done;
    wire [7:0]  mlp_result;
    wire        mlp_data_rd_en;
    wire [DATA_DEPTH_BITS-1:0] mlp_data_rd_addr;
    wire [7:0]  data_rd_data;
    wire [3:0]  mlp_w_hid_idx_0, mlp_w_hid_idx_1;
    wire [1:0]  mlp_w_out_idx;
    wire [7:0]  mlp_sig_idx;
    wire        mlp_start = process_active && (sample_idx < samples_received);

    // RES_RAM
    wire       res_wr_en   = process_active && mlp_done;
    wire [RES_DEPTH_BITS-1:0] res_wr_addr = sample_idx[RES_DEPTH_BITS-1:0];
    wire [7:0] res_wr_data = mlp_result;

    // Output packing
    reg [RES_DEPTH_BITS-1:0] out_rd_idx;
    reg [31:0] out_pack_word;
    reg [1:0]  pack_cnt;
    reg        pack_rd_pending;
    reg        res_rd_en_r;
    reg [RES_DEPTH_BITS-1:0] res_rd_addr_r;
    wire [7:0] res_rd_data;

    always @(posedge ACLK) begin
        if (!ARESETN) begin
            w_state<=W_IDLE; S_AXIS_TREADY<=0; M_AXIS_TVALID<=0; M_AXIS_TLAST<=0;
            weight_cnt<=0; data_wr_en_r<=0; data_wr_ptr<=0;
            samples_received<=0; total_samples<=0; sample_idx<=0;
            word_in_sample<=0; recv_done<=0; got_tlast<=0; unpack_cnt<=0;
            sample_base_addr<=0; buf_tail<=0;
            res_rd_en_r<=0; pack_cnt<=0; pack_rd_pending<=0; out_rd_idx<=0;
        end else begin
            data_wr_en_r <= 1'b0;

            // MLP result capture + buffer tail advance
            if (process_active && mlp_done) begin
                sample_idx       <= sample_idx + 12'd1;
                sample_base_addr <= sample_base_addr + {{(DATA_DEPTH_BITS-3){1'b0}},3'd7};
                buf_tail         <= buf_tail + {{(DATA_DEPTH_BITS-3){1'b0}},3'd7};
            end

            case (w_state)

                W_IDLE: begin
                    M_AXIS_TVALID<=0; M_AXIS_TLAST<=0; weight_cnt<=0;
                    if (S_AXIS_TVALID) begin S_AXIS_TREADY<=1; w_state<=W_READ_WEIGHTS; end
                end

                W_READ_WEIGHTS: begin
                    S_AXIS_TREADY<=1;
                    if (S_AXIS_TVALID && S_AXIS_TREADY) begin
                        if (weight_cnt<16) w_hid_regs[weight_cnt[3:0]]<=S_AXIS_TDATA[7:0];
                        else if (weight_cnt<19) w_out_regs[weight_cnt-16]<=S_AXIS_TDATA[7:0];
                        else sig_lut_regs[weight_cnt-19]<=S_AXIS_TDATA[7:0];
                        if (weight_cnt==NUM_WEIGHTS-1) begin
                            data_wr_ptr<=0; buf_tail<=0; sample_base_addr<=0;
                            samples_received<=0; sample_idx<=0; word_in_sample<=0;
                            recv_done<=0; got_tlast<=0;
                            S_AXIS_TREADY<=buffer_has_room;
                            w_state<=W_RECV_PACKED;
                        end else weight_cnt<=weight_cnt+1;
                    end
                end

                W_RECV_PACKED: begin
                    if (!recv_done) S_AXIS_TREADY<=buffer_has_room; else S_AXIS_TREADY<=0;
                    if (S_AXIS_TVALID && S_AXIS_TREADY && !recv_done) begin
                        captured_word<=S_AXIS_TDATA; S_AXIS_TREADY<=0; unpack_cnt<=0;
                        bytes_to_write <= (word_in_sample==0) ? 3'd4 : 3'd3;
                        got_tlast<=S_AXIS_TLAST;
                        w_state<=W_UNPACK;
                    end
                    // Completion check
                    if (recv_done && sample_idx==total_samples && total_samples!=0) begin
                        out_rd_idx<=0; res_rd_addr_r<=0; res_rd_en_r<=1;
                        pack_cnt<=0; pack_rd_pending<=1; out_pack_word<=0;
                        w_state<=W_WRITE_PACKED;
                    end
                end

                W_UNPACK: begin
                    data_wr_en_r<=1; data_wr_addr_r<=data_wr_ptr;
                    case(unpack_cnt)
                        0: data_wr_data_r<=captured_word[7:0];
                        1: data_wr_data_r<=captured_word[15:8];
                        2: data_wr_data_r<=captured_word[23:16];
                        3: data_wr_data_r<=captured_word[31:24];
                    endcase
                    data_wr_ptr<=data_wr_ptr+{{(DATA_DEPTH_BITS-1){1'b0}},1'b1};
                    if ({1'b0,unpack_cnt}==bytes_to_write-1) begin
                        if (word_in_sample==0) word_in_sample<=1;
                        else begin word_in_sample<=0; samples_received<=samples_received+1; end
                        if (got_tlast) begin
                            recv_done<=1;
                            total_samples<=(word_in_sample==1)?samples_received+1:samples_received;
                            w_state<=W_FINISH_PROC;
                        end else w_state<=W_RECV_PACKED;
                    end else unpack_cnt<=unpack_cnt+1;
                end

                W_FINISH_PROC: begin
                    S_AXIS_TREADY<=0;
                    if (sample_idx==total_samples && total_samples!=0) begin
                        out_rd_idx<=0; res_rd_addr_r<=0; res_rd_en_r<=1;
                        pack_cnt<=0; pack_rd_pending<=1; out_pack_word<=0;
                        w_state<=W_WRITE_PACKED;
                    end
                end

                W_WRITE_PACKED: begin
                    S_AXIS_TREADY<=0; res_rd_en_r<=1;
                    if (total_samples==0) w_state<=W_IDLE;
                    else if (!M_AXIS_TVALID) begin
                        if (pack_rd_pending) pack_rd_pending<=0;
                        else begin
                            case(pack_cnt)
                                0: out_pack_word[7:0]  <=res_rd_data;
                                1: out_pack_word[15:8] <=res_rd_data;
                                2: out_pack_word[23:16]<=res_rd_data;
                                3: out_pack_word[31:24]<=res_rd_data;
                            endcase
                            if (pack_cnt==3 || out_rd_idx==total_samples[RES_DEPTH_BITS-1:0]-1) begin
                                M_AXIS_TVALID<=1;
                                case(pack_cnt)
                                    0: M_AXIS_TDATA<={24'd0,res_rd_data};
                                    1: M_AXIS_TDATA<={16'd0,res_rd_data,out_pack_word[7:0]};
                                    2: M_AXIS_TDATA<={8'd0,res_rd_data,out_pack_word[15:0]};
                                    3: M_AXIS_TDATA<={res_rd_data,out_pack_word[23:0]};
                                endcase
                                M_AXIS_TLAST<=(out_rd_idx==total_samples[RES_DEPTH_BITS-1:0]-1)?1:0;
                            end else begin
                                pack_cnt<=pack_cnt+1; out_rd_idx<=out_rd_idx+1;
                                res_rd_addr_r<=res_rd_addr_r+1; pack_rd_pending<=1;
                            end
                        end
                    end else if (M_AXIS_TREADY) begin
                        M_AXIS_TVALID<=0;
                        if (out_rd_idx==total_samples[RES_DEPTH_BITS-1:0]-1) w_state<=W_IDLE;
                        else begin
                            out_rd_idx<=out_rd_idx+1; res_rd_addr_r<=res_rd_addr_r+1;
                            pack_cnt<=0; out_pack_word<=0; pack_rd_pending<=1;
                        end
                    end
                end

                default: w_state<=W_IDLE;
            endcase
        end
    end

    // ================================================================
    // Sub-modules
    // ================================================================
    memory_RAM #(.width(WIDTH),.depth_bits(DATA_DEPTH_BITS)) DATA_RAM (
        .clk(ACLK), .write_en(data_wr_en_r), .write_address(data_wr_addr_r), .write_data_in(data_wr_data_r),
        .read_en(mlp_data_rd_en), .read_address(mlp_data_rd_addr), .read_data_out(data_rd_data));

    memory_RAM #(.width(WIDTH),.depth_bits(RES_DEPTH_BITS)) RES_RAM (
        .clk(ACLK), .write_en(res_wr_en), .write_address(res_wr_addr), .write_data_in(res_wr_data),
        .read_en(res_rd_en_r), .read_address(res_rd_addr_r), .read_data_out(res_rd_data));

    mlp_parallel mlp_inst (
        .clk(ACLK), .rst_n(ARESETN),
        .start(mlp_start), .done(mlp_done),
        .data_read_en(mlp_data_rd_en), .data_read_address(mlp_data_rd_addr),
        .data_read_data_out(data_rd_data),
        .sample_base_addr(sample_base_addr),
        .w_hid_idx_0(mlp_w_hid_idx_0), .w_hid_val_0(w_hid_regs[mlp_w_hid_idx_0]),
        .w_hid_idx_1(mlp_w_hid_idx_1), .w_hid_val_1(w_hid_regs[mlp_w_hid_idx_1]),
        .w_out_idx(mlp_w_out_idx),      .w_out_val(w_out_regs[mlp_w_out_idx]),
        .sig_lut_idx(mlp_sig_idx),      .sig_lut_val(sig_lut_regs[mlp_sig_idx]),
        .result(mlp_result));

endmodule
