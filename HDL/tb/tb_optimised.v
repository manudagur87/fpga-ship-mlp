`timescale 1ns / 1ps

module tb_mlp_v4_opt;

    parameter N_SAMPLES   = 100;
    parameter NUM_WEIGHTS = 275;

    reg                ACLK = 0;
    reg                ARESETN;
    wire               S_AXIS_TREADY;
    reg  [31:0]        S_AXIS_TDATA;
    reg                S_AXIS_TLAST;
    reg                S_AXIS_TVALID;
    wire               M_AXIS_TVALID;
    wire [31:0]        M_AXIS_TDATA;
    wire               M_AXIS_TLAST;
    reg                M_AXIS_TREADY;

    myip_v1_0_v4_opt UUT (
        .ACLK(ACLK), .ARESETN(ARESETN),
        .S_AXIS_TREADY(S_AXIS_TREADY), .S_AXIS_TDATA(S_AXIS_TDATA),
        .S_AXIS_TLAST(S_AXIS_TLAST),   .S_AXIS_TVALID(S_AXIS_TVALID),
        .M_AXIS_TVALID(M_AXIS_TVALID), .M_AXIS_TDATA(M_AXIS_TDATA),
        .M_AXIS_TLAST(M_AXIS_TLAST),   .M_AXIS_TREADY(M_AXIS_TREADY)
    );

    reg [7:0] weights_mem  [0:NUM_WEIGHTS-1];
    reg [7:0] features_mem [0:N_SAMPLES*7-1];
    reg [7:0] expected_mem [0:N_SAMPLES-1];
    reg [7:0] results_mem  [0:N_SAMPLES-1];

    integer word_cnt, sample_idx, i;
    integer cycle_start, cycle_end;
    reg success;
    reg M_AXIS_TLAST_prev = 0;

    always @(posedge ACLK) M_AXIS_TLAST_prev <= M_AXIS_TLAST;
    always #50 ACLK = ~ACLK;  // 10 MHz (100ns period)

    initial begin
        $display("[V4-OPT TB] Loading test data...");
        $readmemh("test_weights.mem", weights_mem);
        $readmemh("test_input.mem", features_mem);
        $readmemh("test_result_expected.mem", expected_mem);

        #25;
        ARESETN       = 0;
        S_AXIS_TVALID = 0;
        S_AXIS_TLAST  = 0;
        M_AXIS_TREADY = 0;
        #200;
        ARESETN = 1;
        #100;

        // === Send weights ===
        $display("[V4-OPT TB] Sending weights + %0d packed samples...", N_SAMPLES);
        word_cnt = 0;
        S_AXIS_TVALID = 1;
        while (word_cnt < NUM_WEIGHTS) begin
            if (S_AXIS_TREADY) begin
                S_AXIS_TDATA = {24'd0, weights_mem[word_cnt]};
                S_AXIS_TLAST = 0;
                word_cnt = word_cnt + 1;
            end
            #100;
        end

        // === Send packed features (2 words per sample) ===
        // Record cycle count at start of feature transmission
        cycle_start = $time;
        sample_idx = 0;
        word_cnt = 0;
        while (sample_idx < N_SAMPLES) begin
            if (S_AXIS_TREADY) begin
                if (word_cnt == 0) begin
                    S_AXIS_TDATA = {features_mem[sample_idx*7+3], features_mem[sample_idx*7+2],
                                    features_mem[sample_idx*7+1], features_mem[sample_idx*7+0]};
                    S_AXIS_TLAST = 0;
                    word_cnt = 1;
                end else begin
                    S_AXIS_TDATA = {8'd0, features_mem[sample_idx*7+6],
                                    features_mem[sample_idx*7+5], features_mem[sample_idx*7+4]};
                    S_AXIS_TLAST = (sample_idx == N_SAMPLES - 1);
                    word_cnt = 0;
                    sample_idx = sample_idx + 1;
                end
            end
            #100;
        end
        S_AXIS_TVALID = 0;
        S_AXIS_TLAST  = 0;

        // === Receive packed results ===
        $display("[V4-OPT TB] Receiving packed results...");
        word_cnt = 0;
        M_AXIS_TREADY = 1;
        while (M_AXIS_TLAST | ~M_AXIS_TLAST_prev) begin
            if (M_AXIS_TVALID) begin
                if (word_cnt * 4 + 0 < N_SAMPLES) results_mem[word_cnt*4+0] = M_AXIS_TDATA[ 7: 0];
                if (word_cnt * 4 + 1 < N_SAMPLES) results_mem[word_cnt*4+1] = M_AXIS_TDATA[15: 8];
                if (word_cnt * 4 + 2 < N_SAMPLES) results_mem[word_cnt*4+2] = M_AXIS_TDATA[23:16];
                if (word_cnt * 4 + 3 < N_SAMPLES) results_mem[word_cnt*4+3] = M_AXIS_TDATA[31:24];
                word_cnt = word_cnt + 1;
            end
            #100;
        end
        M_AXIS_TREADY = 0;
        cycle_end = $time;

        // === Performance ===
        $display("[V4-OPT TB] Processing time (features sent to last result): %0d ns (%0d cycles)",
                 cycle_end - cycle_start, (cycle_end - cycle_start) / 100);
        $display("[V4-OPT TB] Effective throughput: %0d ns/sample (%0.1f cy/sample)",
                 (cycle_end - cycle_start) / N_SAMPLES,
                 (cycle_end - cycle_start) / 100.0 / N_SAMPLES);

        // === Compare ===
        success = 1;
        for (i = 0; i < N_SAMPLES; i = i + 1) begin
            if (results_mem[i] !== expected_mem[i]) begin
                $display("  MISMATCH sample %0d: got %02h, expected %02h", i, results_mem[i], expected_mem[i]);
                success = 0;
            end else
                $display("  sample %0d: %02h OK", i, results_mem[i]);
        end

        if (success) $display("[V4-OPT TB] TEST PASSED (%0d samples).", N_SAMPLES);
        else         $display("[V4-OPT TB] TEST FAILED.");
        #200; $finish;
    end

endmodule
