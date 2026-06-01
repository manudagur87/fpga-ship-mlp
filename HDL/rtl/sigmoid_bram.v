`timescale 1ns / 1ps

// Dual-read sigmoid LUT stored in BRAM.
// Write: synchronous (1 port). Reads: combinational (2 ports).
// Vivado infers distributed RAM or BRAM with combinational read.

module sigmoid_bram (
    input  wire        clk,
    // Write port
    input  wire        wr_en,
    input  wire [7:0]  wr_addr,
    input  wire [7:0]  wr_data,
    // Read port A (combinational)
    input  wire [7:0]  rd_addr_a,
    output wire [7:0]  rd_data_a,
    // Read port B (combinational)
    input  wire [7:0]  rd_addr_b,
    output wire [7:0]  rd_data_b
);

    reg [7:0] lut [0:255];

    always @(posedge clk) begin
        if (wr_en)
            lut[wr_addr] <= wr_data;
    end

    // Combinational reads — no latency, same as old LUT mux
    assign rd_data_a = lut[rd_addr_a];
    assign rd_data_b = lut[rd_addr_b];

endmodule
