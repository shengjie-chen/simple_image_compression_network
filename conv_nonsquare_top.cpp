/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *
 *  \file conv_top.cpp
 *
 *  HLS Top function with a single convolutional layer for unit testing
 *
 *****************************************************************************/
#define AP_INT_MAX_W 16384
//#define DEBUG
#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"

#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "conv.hpp"
#include "memdata_nonsquare.h"
#include "config_nonsquare.h"
#include "streamtools.h"


template<
	unsigned IFM_Channels1,
	unsigned INPUT_PRECISION,
	unsigned PADDING,
	unsigned IFMDim1_x,
	unsigned IFMDim1_y
>
void padding2(stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > & in, stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > & out){

	FMPadding_nonsquare<IFMDim1_x+2*PADDING, IFMDim1_y +2*PADDING, 2*PADDING, 2*PADDING, IFM_Channels1, IFM_Channels1, ap_uint<INPUT_PRECISION>>(in, out);
}



template<
unsigned KERNEL_DIM_X ,
unsigned KERNEL_DIM_Y , 
unsigned SIMD1 , 
unsigned PE1 , 
unsigned WIDTH , 
unsigned IFM_Channels1 , 
unsigned OFM_Channels1 , 
unsigned IFMDim1_x , 
unsigned IFMDim1_y , 
unsigned OFMDim1_x , 
unsigned OFMDim1_y , 
unsigned STRIDE_x , 
unsigned STRIDE_y , 
unsigned PADDING , 
unsigned INPUT_PRECISION , 
unsigned TILE1 , 
unsigned ACTIVATION_PRECISION >
void conv2d(
	FixedPointWeights<SIMD1, ap_int<WIDTH>, PE1, TILE1> const &weights,
	FixedPointWeights<1, ap_int<8>, 1, OFM_Channels1> const &bias, 
	stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > & in, 
	stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION> > & out, 
	unsigned int numReps){
#pragma HLS DATAFLOW
//	ConvLayer_Batch<KERNEL_DIM_X, IFM_Channels1, IFMDim1_x, OFM_Channels1, OFMDim1_x, SIMD1, PE1, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_int<16> >, Identity >(in, out, PARAM::weights, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_dsp());
  unsigned const MatrixW = KERNEL_DIM_X * KERNEL_DIM_Y * IFM_Channels1;
  unsigned const MatrixH = OFM_Channels1;
  unsigned const InpPerImage = (IFMDim1_x+2*PADDING)*(IFMDim1_y+2*PADDING);
  hls::stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > in_pad("in_pad");
  hls::stream<ap_uint<SIMD1*INPUT_PRECISION> > wa_in("wa_in");
  hls::stream<ap_uint<SIMD1*INPUT_PRECISION> > convInp("convInp");
  hls::stream<ap_uint<SIMD1*INPUT_PRECISION> > convInp_2("convInp_2");
  hls::stream<ap_uint<PE1*ACTIVATION_PRECISION> > mvOut("mvOut");
  hls::stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION> > out_wo_bias;

  padding2<IFM_Channels1, INPUT_PRECISION, PADDING, IFMDim1_x,IFMDim1_y>(in, in_pad);

  StreamingDataWidthConverter_Batch<IFM_Channels1*INPUT_PRECISION, SIMD1*INPUT_PRECISION, InpPerImage>(in_pad, wa_in, numReps);

  // TODO 18*12 is error
  ConvolutionInputGenerator_NonSquare<KERNEL_DIM_X, KERNEL_DIM_Y, IFM_Channels1, INPUT_PRECISION, IFMDim1_x+2*PADDING, IFMDim1_y+2*PADDING,
  	  IFMDim1_x+2*PADDING - KERNEL_DIM_X + 1, IFMDim1_y+2*PADDING - KERNEL_DIM_Y + 1, SIMD1,1, 1>(wa_in, convInp, numReps, ap_resource_dflt());

  // get real windows
  unsigned win_cnt = 0;
  unsigned win_num = 0;
  ap_uint<SIMD1*INPUT_PRECISION> tmp;
  for(int row=0; row < IFMDim1_y+2*PADDING - KERNEL_DIM_Y + 1; row++){
	  for(int col=0; col < IFMDim1_x+2*PADDING - KERNEL_DIM_X + 1; col++){
		  for(int f_row=0; f_row < KERNEL_DIM_Y; f_row++){
			  for(int f_col=0; f_col < KERNEL_DIM_X; f_col++){
				  for(int channel=0; channel<IFM_Channels1/SIMD1;channel++){
					  tmp = convInp.read();
					  if((row % STRIDE_y == 0) & (col % STRIDE_x == 0)){
						  convInp_2.write(tmp);
					  }
				  }
			  }
		  }
	  }
  }

  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD1, PE1, 1, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_int<ACTIVATION_PRECISION> >, Identity>
	(static_cast<hls::stream<ap_uint<SIMD1*INPUT_PRECISION>>&>(convInp_2),
	 static_cast<hls::stream<ap_uint<PE1*ACTIVATION_PRECISION>>&>  (mvOut),
	 weights, PassThroughActivation<ap_uint<ACTIVATION_PRECISION>>(), numReps* OFMDim1_x * OFMDim1_y, ap_resource_dsp());
  StreamingDataWidthConverter_Batch<PE1*ACTIVATION_PRECISION, OFM_Channels1*ACTIVATION_PRECISION, OFMDim1_x * OFMDim1_y * (OFM_Channels1 / PE1)>(mvOut, out_wo_bias, numReps);

  // add bias and relu
  for(unsigned i=0; i < OFMDim1_x * OFMDim1_y; i++){
	  ap_uint<OFM_Channels1*ACTIVATION_PRECISION> tmp = out_wo_bias.read();
	  for(unsigned j=0; j < OFM_Channels1; j++){
#pragma HLS UNROLL
		  tmp((j+1)*ACTIVATION_PRECISION - 1, j*ACTIVATION_PRECISION) = tmp((j+1)*ACTIVATION_PRECISION - 1, j*ACTIVATION_PRECISION) + bias.weights(j)[0][0];
		  if (tmp((j+1)*ACTIVATION_PRECISION - 1, (j+1)*ACTIVATION_PRECISION - 1) == 1) {
			  tmp((j+1)*ACTIVATION_PRECISION - 1, j*ACTIVATION_PRECISION) = 0;
		  }
	  }
	  out.write(tmp);
  }

}

void conv2d_layer0(stream<ap_uint<CONV_0_IFM_CH* CONV_0_IN_BIT> > & in, stream<ap_uint<CONV_0_OFM_CH* CONV_0_OUT_BIT> > & out, unsigned int numReps){
	conv2d< CONV_0_K, CONV_0_K, CONV_0_SIMD, CONV_0_PE,
		CONV_0_W_BIT, CONV_0_IFM_CH, CONV_0_OFM_CH, CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_OFM_ROW, CONV_0_OFM_COL,
		CONV_0_S, CONV_0_S, CONV_0_P, CONV_0_IN_BIT, CONV_0_W_TILES, CONV_0_OUT_BIT>(PARAM::weights_layer0, PARAM::bias_layer0, in, out, numReps);
}




void encoder_net(stream<ap_uint<CONV_0_IFM_CH* CONV_0_IN_BIT> > & in, stream<ap_uint<CONV_3_OFM_CH* CONV_3_OUT_BIT> > & out, const unsigned int numReps) {
#pragma HLS DATAFLOW

	hls::stream<ap_uint<CONV_0_OUT_BIT* CONV_0_OFM_CH>>  conv_0_out("conv_0_out");
	conv2d< CONV_0_K, CONV_0_K, CONV_0_SIMD, CONV_0_PE,
		CONV_0_W_BIT, CONV_0_IFM_CH, CONV_0_OFM_CH, CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_OFM_ROW, CONV_0_OFM_COL,
		CONV_0_S, CONV_0_S, CONV_0_P, CONV_0_IN_BIT, CONV_0_W_TILES, CONV_0_OUT_BIT>(PARAM::weights_layer0, PARAM::bias_layer0, in, conv_0_out, numReps);
#ifdef DEBUG
	cout << "layer0 calculation completes. conv_0_out size: " << conv_0_out.size() << endl;
#endif

	hls::stream<ap_uint<CONV_1_OUT_BIT* CONV_1_OFM_CH>>  conv_1_out("conv_1_out");
	conv2d< CONV_1_K, CONV_1_K, CONV_1_SIMD, CONV_1_PE,
		CONV_1_W_BIT, CONV_1_IFM_CH, CONV_1_OFM_CH, CONV_1_IFM_ROW, CONV_1_IFM_COL, CONV_1_OFM_ROW, CONV_1_OFM_COL,
		CONV_1_S, CONV_1_S, CONV_1_P, CONV_1_IN_BIT, CONV_1_W_TILES, CONV_1_OUT_BIT>(PARAM::weights_layer1, PARAM::bias_layer1, conv_0_out, conv_1_out, numReps);
#ifdef DEBUG
	cout << "layer1 calculation completes. conv_1_out size: " << conv_1_out.size() << endl;
#endif

	hls::stream<ap_uint<CONV_2_OUT_BIT* CONV_2_OFM_CH>>  conv_2_out("conv_2_out");
	conv2d< CONV_2_K, CONV_2_K, CONV_2_SIMD, CONV_2_PE,
		CONV_2_W_BIT, CONV_2_IFM_CH, CONV_2_OFM_CH, CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_OFM_ROW, CONV_2_OFM_COL,
		CONV_2_S, CONV_2_S, CONV_2_P, CONV_2_IN_BIT, CONV_2_W_TILES, CONV_2_OUT_BIT>(PARAM::weights_layer2, PARAM::bias_layer2, conv_1_out, conv_2_out, numReps);
#ifdef DEBUG
	cout << "layer2 calculation completes. conv_2_out size: " << conv_2_out.size() << endl;
#endif

	hls::stream<ap_uint<CONV_3_OUT_BIT* CONV_3_OFM_CH>>  conv_3_out("conv_3_out");
	conv2d< CONV_3_K, CONV_3_K, CONV_3_SIMD, CONV_3_PE,
		CONV_3_W_BIT, CONV_3_IFM_CH, CONV_3_OFM_CH, CONV_3_IFM_ROW, CONV_3_IFM_COL, CONV_3_OFM_ROW, CONV_3_OFM_COL,
		CONV_3_S, CONV_3_S, CONV_3_P, CONV_3_IN_BIT, CONV_3_W_TILES, CONV_3_OUT_BIT>(PARAM::weights_layer3, PARAM::bias_layer3, conv_2_out, out, numReps);
#ifdef DEBUG
	cout << "layer3 calculation completes. conv_3_out size: " << conv_3_out.size() << endl;
#endif


}



