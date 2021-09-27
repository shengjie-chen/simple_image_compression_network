############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project image_compression_model
add_files conv_nonsquare_top.cpp -cflags "-std=c++0x"
add_files -tb conv3_nonsquare_tb.cpp -cflags "-std=c++0x"
open_solution "solution1"
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
#source "./image_compression_model/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
