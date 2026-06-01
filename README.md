# fpga-ship-mlp
Hardware-accelerated ship detection system on the Xilinx Kria KV260. Uses a sliding window approach with an MLP neural network to detect ships in satellite imagery. Features three inference modes: pure software (ARM), HDL (Verilog), and HLS accelerators.

## Run on the Kria KV260

1. Install Ubuntu + PYNQ on the Kria KV260.
2. From this repo, copy the PYNQ/ folder to the board:
   ```bash
   scp -r PYNQ/ ubuntu@<board-ip>:~/finale/
   ```
3. SSH into the board and run the host:
   ```bash
   cd ~/finale && sudo python3 ship_detector_finale.py
   ```

The Tkinter GUI loads the bitstream, lets you select a coprocessor mode, and runs inference on a chosen scene.
