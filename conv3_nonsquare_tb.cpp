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
 *  \file conv3_tb.cpp
 *
 *  Testbench for the convolution HLS block
 *
 *****************************************************************************/
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#define AP_INT_MAX_W 16384
#include "ap_int.h"
#include "weights.hpp"
#include "bnn-library.h"
#include "memdata_nonsquare.h"
#include "config_nonsquare.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "conv.hpp"
using namespace hls;
using namespace std;




#define MAX_IMAGES 1
//void Testbench_conv_nonsquare(stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > & in, stream<ap_uint<CONV_0_OFM_CH*CONV_0_OUT_BIT> > & out, unsigned int numReps);

void conv2d_layer1(stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > & in, stream<ap_uint<CONV_0_OFM_CH*CONV_0_OUT_BIT> > & out, unsigned int numReps);

void CONV_0_P2(stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > & in, stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > & out);

void deconv2d_layer4(stream<ap_uint<CONV_4_IFM_CH* CONV_4_IN_BIT> >& in, stream<ap_uint<CONV_4_OFM_CH* CONV_4_OUT_BIT> >& out, unsigned int numReps);
//void deconv522_layer1(stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > & in, stream<ap_uint<CONV_0_OFM_CH*CONV_0_OUT_BIT> > & out, unsigned int numReps);

//int test_CONV_0_P(){
//	static	ap_uint<CONV_0_IN_BIT> IMAGE[MAX_IMAGES][CONV_0_IFM_ROW][CONV_0_IFM_COL][CONV_0_IFM_CH];
//	stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > input_stream("input_stream");
//	stream<ap_uint<CONV_0_OFM_CH*CONV_0_OUT_BIT> > output_stream("output_stream");
//	unsigned int counter = 0;
//	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
//		for (unsigned int oy = 0; oy < CONV_0_IFM_COL; oy++) {
//			for (unsigned int ox = 0; ox < CONV_0_IFM_ROW; ox++) {
//				ap_uint<CONV_0_IN_BIT*CONV_0_IFM_CH> input_channel = 0;
//				for(unsigned int channel = 0; channel < CONV_0_IFM_CH; channel++)
//				{
//					ap_uint<CONV_0_IN_BIT> input = (ap_uint<CONV_0_IN_BIT>)(1);
//					if ((oy == 3) & (ox == 0)){
//						input = (ap_uint<CONV_0_IN_BIT>)(0);
//					}
//
//					IMAGE[n_image][ox][oy][channel]= input;
//					input_channel = input_channel >> CONV_0_IN_BIT;
//					input_channel(CONV_0_IFM_CH*CONV_0_IN_BIT-1,(CONV_0_IFM_CH-1)*CONV_0_IN_BIT)=input;
//
//					counter++;
//				}
//				input_stream.write(input_channel);
//			}
//		}
//	}
//	CONV_0_P2(input_stream, output_stream);
//
//	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
//			for (unsigned int oy = 0; oy < CONV_0_IFM_COL+4; oy++) {
//				for (unsigned int ox = 0; ox < CONV_0_IFM_ROW+4; ox++) {
//					auto EXP = output_stream.read();
//					std::cout << EXP << std::endl;
//				}
//			}
//		}
//	return 0;
//}

int test_conv2d_layer0()
{
	std::cout << "Input image size is " << CONV_0_IFM_ROW << " X " << CONV_0_IFM_COL << " X " << CONV_0_IFM_CH << std::endl;

	static	ap_uint<CONV_0_IN_BIT> IMAGE[MAX_IMAGES][CONV_0_IFM_ROW+2*CONV_0_P][CONV_0_IFM_COL+2*CONV_0_P][CONV_0_IFM_CH];
	static	ap_int<CONV_0_OUT_BIT> TEST[MAX_IMAGES][CONV_0_OFM_ROW][CONV_0_OFM_COL][CONV_0_OFM_CH];
	stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > input_stream("input_stream");
	stream<ap_uint<CONV_0_IFM_CH*CONV_0_IN_BIT> > input_stream_pad("input_stream_pad");
	stream<ap_uint<CONV_0_OFM_CH*CONV_0_OUT_BIT> > output_stream("output_stream");
	//unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < CONV_0_IFM_COL+2*CONV_0_P; oy++) {
			for (unsigned int ox = 0; ox < CONV_0_IFM_ROW+2*CONV_0_P; ox++) {
				ap_uint<CONV_0_IN_BIT*CONV_0_IFM_CH> input_channel = 0;
				if (((ox < CONV_0_P) | (ox >= CONV_0_IFM_ROW+CONV_0_P)) |((oy < CONV_0_P) | (oy >= CONV_0_IFM_COL+CONV_0_P))){
					input_channel = 0;
					for(unsigned int channel = 0; channel < CONV_0_IFM_CH; channel++){
						IMAGE[n_image][ox][oy][channel]= 0;
					}
				}else{
					for(unsigned int channel = 0; channel < CONV_0_IFM_CH; channel++)
					{
						ap_uint<CONV_0_IN_BIT> input = (ap_uint<CONV_0_IN_BIT>)(1);
						/*if ((oy == 3+CONV_0_P) & (ox == 0+CONV_0_P)){
							input = (ap_uint<CONV_0_IN_BIT>)(0);
						}*/

						IMAGE[n_image][ox][oy][channel]= input;
						input_channel = input_channel >> CONV_0_IN_BIT;
						input_channel(CONV_0_IFM_CH*CONV_0_IN_BIT-1,(CONV_0_IFM_CH-1)*CONV_0_IN_BIT)=input;

						/*counter++;*/
					}
					input_stream.write(input_channel);
				}
			}
		}
	}
	static	ap_int<CONV_0_W_BIT> W1[CONV_0_OFM_CH][CONV_0_K][CONV_0_K][CONV_0_IFM_CH];
	// initialize the weights
	constexpr int TX = (CONV_0_IFM_CH*CONV_0_K*CONV_0_K) / CONV_0_SIMD;
	constexpr int TY = CONV_0_OFM_CH / CONV_0_PE;
	unsigned int kx=0;
	unsigned int ky=0;
	unsigned int chan_count=0;
//	unsigned int out_chan_count=0;

	for(int pe=0;pe <CONV_0_PE;pe++){
		unsigned int out_chan_count=pe;
		for (unsigned int oy = 0; oy < TY; oy++) {
			for (unsigned int ox = 0; ox <TX; ox++) {
				for(int simd=0;simd<CONV_0_SIMD;simd++){
					W1[out_chan_count][kx][ky][chan_count] = PARAM::weights_layer0.weights(oy*TX + ox)[pe][simd];
			    	chan_count++;
				    if (chan_count==CONV_0_IFM_CH){
				    	chan_count=0;
						kx++;
						if (kx==CONV_0_K){
							kx=0;
							ky++;
							if (ky==CONV_0_K){
								ky=0;
						    	out_chan_count += CONV_0_PE;
							    if (out_chan_count==CONV_0_OFM_CH){
							    	out_chan_count=0;
							    }
						    }
					    }
					}
				}
			}
		}
	}

	// initial bias
	static	ap_int<CONV_0_OUT_BIT> BIAS[CONV_0_OFM_CH];
	for(int i=0; i<CONV_0_OFM_CH; i++){
		BIAS[i] = PARAM::bias_layer0.weights(i)[0][0];
	}


	conv2d_layer1(input_stream, output_stream, MAX_IMAGES);
	std::cout << "Hardware computation complete. " << std::endl;

	conv_nonsquare<MAX_IMAGES,CONV_0_IFM_ROW+2*CONV_0_P,CONV_0_IFM_COL+2*CONV_0_P,CONV_0_OFM_ROW,CONV_0_OFM_COL,CONV_0_IFM_CH,CONV_0_OFM_CH, CONV_0_K, CONV_0_K, CONV_0_S, CONV_0_S, ap_uint<CONV_0_IN_BIT>, ap_int<CONV_0_OUT_BIT>, ap_int<CONV_0_W_BIT> >(IMAGE, W1, TEST);
	



	// get finial result
	// static	ap_int<CONV_0_OUT_BIT> TEST[MAX_IMAGES][CONV_0_OFM_ROW][CONV_0_OFM_COL][CONV_0_OFM_CH];
	for(int img_num = 0; img_num < MAX_IMAGES; img_num++){
		for (int y=0; y < CONV_0_OFM_COL; y++){
			for(int x=0; x < CONV_0_OFM_ROW; x++){
				for(int channel=0; channel < CONV_0_OFM_CH; channel++){
					TEST[img_num][x][y][channel] += BIAS[channel];
					if (TEST[img_num][x][y][channel] < 0) {
						TEST[img_num][x][y][channel] = 0;
					}
				}
			}
		}
	}
	std::cout << "Verification computation complete. " << std::endl;


//	Testbench_conv_nonsquare(input_stream, output_stream, MAX_IMAGES);


	std::cout << "Output image size is " << CONV_0_OFM_ROW << " X " << CONV_0_OFM_COL << " X " << CONV_0_OFM_CH << std::endl;


	int err_counter = 0, err_perimage=0;
	ap_int<CONV_0_OUT_BIT> out_chan;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < CONV_0_OFM_COL; oy++) {
			for (unsigned int ox = 0; ox < CONV_0_OFM_ROW; ox++) {
				for(int e=0;e<1;e++){
					ap_uint<CONV_0_OFM_CH*CONV_0_OUT_BIT> outElem = output_stream.read();
//					std::cout << "RES = " << hex << outElem << std::endl;
//					std::cout << "0 =" << hex << TEST[n_image][ox][oy][0] << std::endl;
//					std::cout << "1 =" << hex << TEST[n_image][ox][oy][1] << std::endl;
					for(unsigned int channel = 0; channel < CONV_0_OFM_CH; channel++){
						ap_int<CONV_0_OUT_BIT> EXP = TEST[n_image][ox][oy][channel + e * CONV_0_OFM_CH];
						out_chan(CONV_0_OUT_BIT-1,0) = outElem((channel + 1)*CONV_0_OUT_BIT-1,channel*CONV_0_OUT_BIT);

//						std::cout << "RES["<<oy <<"]["<<ox<<"]["<<channel<<"]=" << EXP << std::endl;

						if (EXP != out_chan){
							std::cout << "ERROR: Expected["<<oy <<"]["<<ox<<"]["<<channel<<"]=" << EXP << " actual " <<  out_chan << std::endl;
							//return 1;
							err_counter ++;
							err_perimage++;
							//if(err_counter>10)
								//return 1;
						}

					}
				}
			}
		}
		if(err_perimage == 0){
			std::cout << "Image # " << n_image << " passed the testing."<< std::endl;
		}
		else{
			err_perimage=0;
			std::cout << "Image # " << n_image << " failed the testing."<< std::endl;
		}
	}

	//std::cout << TEST[0][0][0][0] << std::endl;
	//std::cout << TEST[0][0][0][1] << std::endl;
	//std::cout << TEST[0][1][0][0] << std::endl;
	//std::cout << TEST[0][1][0][1] << std::endl;

	//std::cout << TEST[0][0][0][0] << std::endl;
	//std::cout << TEST[0][0][0][1] << std::endl;
	//std::cout << TEST[0][0][1][0] << std::endl;
	//std::cout << TEST[0][0][1][1] << std::endl;



	if(err_counter == 0){
		return 0;
	}
	else{
		return 1;
	}

}

//template<	unsigned int InputDim_x,
//	unsigned int InputDim_y,
//	unsigned int InNumChannels,
//	unsigned int OutNumChannels,
//	unsigned int SIMD,
//	unsigned int PE,
//	typename TWW,
//	typename TBB>
//	void deconv522(TWW const& weights, TBB const& bias, stream<ap_uint<InNumChannels* CONV_0_IN_BIT> >& in, stream<ap_uint<OutNumChannels* CONV_0_IN_BIT> >& out, unsigned int numReps);

int test_deconv2d_layer4()
{
	//unsigned int N = 1;
	//unsigned int C = 4;
	//unsigned int Y = 8;
	//unsigned int X = 8;
//	static	ap_uint<CONV_4_IN_BIT> IMAGE[N][X][Y][C];
//	static	ap_int<CONV_4_OUT_BIT> TEST[N][X][Y][C];

	stream<ap_uint<CONV_4_IFM_CH*CONV_4_IN_BIT> > input_stream("input_stream");
	stream<ap_uint<CONV_4_IFM_CH*CONV_4_IN_BIT> > input_stream_pad("input_stream_pad");
	stream<ap_uint<CONV_4_OFM_CH*CONV_4_OUT_BIT> > output_stream("output_stream");


	std::cout << "Input image size is " << CONV_4_IFM_ROW << " X " << CONV_4_IFM_COL << " X " << CONV_4_IFM_CH << std::endl;

	static	ap_uint<CONV_4_IN_BIT> IMAGE[MAX_IMAGES][2 * CONV_4_IFM_ROW + 2 * CONV_4_P][2 * CONV_4_IFM_COL + 2 * CONV_4_P][CONV_4_IFM_CH];
	static	ap_int<CONV_4_OUT_BIT> TEST[MAX_IMAGES][CONV_4_OFM_ROW][CONV_4_OFM_COL][CONV_4_OFM_CH];
	//stream<ap_uint<CONV_4_IFM_CH* CONV_4_IN_BIT> > input_stream("input_stream");
	//stream<ap_uint<CONV_4_IFM_CH* CONV_4_IN_BIT> > input_stream_pad("input_stream_pad");
	//stream<ap_uint<CONV_4_OFM_CH* CONV_4_OUT_BIT> > output_stream("output_stream");
	//unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < 2 * CONV_4_IFM_COL + 2 * CONV_4_P; oy++) {
			for (unsigned int ox = 0; ox < 2 * CONV_4_IFM_ROW + 2 * CONV_4_P; ox++) {
				ap_uint<CONV_4_IN_BIT* CONV_4_IFM_CH> input_channel = 0;
				if (((ox < CONV_4_P) | (ox >= CONV_4_IFM_ROW * 2 + CONV_4_P)) | ((oy < CONV_4_P) | (oy >= CONV_4_IFM_COL * 2 + CONV_4_P)) | ((ox + 1 - CONV_4_P)%2 == 0) | ((oy + 1 - CONV_4_P) % 2 == 0)) {
					input_channel = 0;
					for (unsigned int channel = 0; channel < CONV_4_IFM_CH; channel++) {
						IMAGE[n_image][ox][oy][channel] = 0;
					}
				}
				else {
					for (unsigned int channel = 0; channel < CONV_4_IFM_CH; channel++)
					{
						ap_uint<CONV_4_IN_BIT> input = (ap_uint<CONV_4_IN_BIT>)(1);
				/*		if ((oy == 3 + CONV_4_P) & (ox == 0 + CONV_4_P)) {
							input = (ap_uint<CONV_4_IN_BIT>)(0);
						}*/

						IMAGE[n_image][ox][oy][channel] = input;
						input_channel = input_channel >> CONV_4_IN_BIT;
						input_channel(CONV_4_IFM_CH * CONV_4_IN_BIT - 1, (CONV_4_IFM_CH - 1) * CONV_4_IN_BIT) = input;

						//counter++;
					}
					input_stream.write(input_channel);
				}
			}
		}
	}

	static	ap_int<CONV_4_W_BIT> W1[CONV_4_OFM_CH][CONV_4_K][CONV_4_K][CONV_4_IFM_CH];
	// initialize the weights
	constexpr int TX = (CONV_4_IFM_CH * CONV_4_K * CONV_4_K) / CONV_4_SIMD;
	constexpr int TY = CONV_4_OFM_CH / CONV_4_PE;
	unsigned int kx = 0;
	unsigned int ky = 0;
	unsigned int chan_count = 0;
	//	unsigned int out_chan_count=0;

	for (int pe = 0; pe < CONV_4_PE; pe++) {
		unsigned int out_chan_count = pe;
		for (unsigned int oy = 0; oy < TY; oy++) {
			for (unsigned int ox = 0; ox < TX; ox++) {
				for (int simd = 0; simd < CONV_4_SIMD; simd++) {
					W1[out_chan_count][kx][ky][chan_count] = PARAM::weights_layer4.weights(oy * TX + ox)[pe][simd];
					chan_count++;
					if (chan_count == CONV_4_IFM_CH) {
						chan_count = 0;
						kx++;
						if (kx == CONV_4_K) {
							kx = 0;
							ky++;
							if (ky == CONV_4_K) {
								ky = 0;
								out_chan_count += CONV_4_PE;
								if (out_chan_count == CONV_4_OFM_CH) {
									out_chan_count = 0;
								}
							}
						}
					}
				}
			}
		}
	}

	// initial bias
	static	ap_int<CONV_4_OUT_BIT> BIAS[CONV_4_OFM_CH];
	for (int i = 0; i < CONV_4_OFM_CH; i++) {
		BIAS[i] = PARAM::bias_layer4.weights(i)[0][0];
	}

	//for (int n = 0; n < MAX_IMAGES; n++) {
	//	for (int x = 1; x < 6; x++) {
	//		for (int y = 0; y < 5; y++) {
	//			for (int c = 0; c < CONV_4_IFM_CH; c++) {
	//			std::cout << "image[" << n << "][" << x << "][" << y << "][" << c << "]is " << IMAGE[n][x][y][c] << std :: endl;
	//			std::cout << "W1[" << 0 << "][" << x << "][" << y << "][" << c << "]is " << W1[0][x][y][c] << std::endl;
	//			}
	//		}
	//	}
	//}

	conv_nonsquare<MAX_IMAGES, 2 * CONV_4_IFM_ROW + 2 * CONV_4_P, 2 * CONV_4_IFM_COL + 2 * CONV_4_P, 
		CONV_4_OFM_ROW, CONV_4_OFM_COL, CONV_4_IFM_CH, CONV_4_OFM_CH, CONV_4_K, CONV_4_K, 1, 1, 
		ap_uint<CONV_4_IN_BIT>, ap_int<CONV_4_OUT_BIT>, ap_int<CONV_4_W_BIT> >
		(IMAGE, W1, TEST);

	// get finial result
	// static	ap_int<CONV_4_OUT_BIT> TEST[MAX_IMAGES][CONV_4_OFM_ROW][CONV_4_OFM_COL][CONV_4_OFM_CH];
	for (int img_num = 0; img_num < MAX_IMAGES; img_num++) {
		for (int y = 0; y < CONV_4_OFM_COL; y++) {
			for (int x = 0; x < CONV_4_OFM_ROW; x++) {
				for (int channel = 0; channel < CONV_4_OFM_CH; channel++) {
					TEST[img_num][x][y][channel] += BIAS[channel];
					if (TEST[img_num][x][y][channel] < 0) {
						TEST[img_num][x][y][channel] = 0;
					}
				}
			}
		}
	}



	//	Testbench_conv_nonsquare(input_stream, output_stream, MAX_IMAGES);
	deconv2d_layer4(input_stream, output_stream, MAX_IMAGES);

	std::cout << "Output image size is " << CONV_4_OFM_ROW << " X " << CONV_4_OFM_COL << " X " << CONV_4_OFM_CH << std::endl;


	int err_counter = 0, err_perimage = 0;
	ap_int<CONV_4_OUT_BIT> out_chan;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < CONV_4_OFM_COL; oy++) {
			for (unsigned int ox = 0; ox < CONV_4_OFM_ROW; ox++) {
				for (int e = 0; e < 1; e++) {
					ap_uint<CONV_4_OFM_CH* CONV_4_OUT_BIT> outElem = output_stream.read();
					//					std::cout << "RES = " << hex << outElem << std::endl;
					//					std::cout << "0 =" << hex << TEST[n_image][ox][oy][0] << std::endl;
					//					std::cout << "1 =" << hex << TEST[n_image][ox][oy][1] << std::endl;
					for (unsigned int channel = 0; channel < CONV_4_OFM_CH; channel++) {
						ap_int<CONV_4_OUT_BIT> EXP = TEST[n_image][ox][oy][channel + e * CONV_4_OFM_CH];
						out_chan(CONV_4_OUT_BIT - 1, 0) = outElem((channel + 1) * CONV_4_OUT_BIT - 1, channel * CONV_4_OUT_BIT);

						//						std::cout << "RES["<<oy <<"]["<<ox<<"]["<<channel<<"]=" << EXP << std::endl;

						if (EXP != out_chan) {
							std::cout << "ERROR: Expected[" << oy << "][" << ox << "][" << channel << "]=" << EXP << " actual " << out_chan << std::endl;
							//return 1;
							err_counter++;
							err_perimage++;
							//if(err_counter>10)
								//return 1;
						}

					}
				}
			}
		}
		if (err_perimage == 0) {
			std::cout << "Image # " << n_image << " passed the testing." << std::endl;
		}
		else {
			err_perimage = 0;
			std::cout << "Image # " << n_image << " failed the testing." << std::endl;
		}
	}

	//std::cout << TEST[0][0][0][0] << std::endl;
	//std::cout << TEST[0][0][0][1] << std::endl;
	//std::cout << TEST[0][1][0][0] << std::endl;
	//std::cout << TEST[0][1][0][1] << std::endl;

	//std::cout << TEST[0][0][0][0] << std::endl;
	//std::cout << TEST[0][0][0][1] << std::endl;
	//std::cout << TEST[0][0][1][0] << std::endl;
	//std::cout << TEST[0][0][1][1] << std::endl;



	if (err_counter == 0) {
		return 0;
	}
	else {
		return 1;
	}



	//unsigned int counter = 0;
	//for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
	//	for (unsigned int oy = 0; oy < CONV_4_IFM_COL; oy++) {
	//		for (unsigned int ox = 0; ox < CONV_4_IFM_ROW; ox++) {
	//			ap_uint<CONV_4_IN_BIT*CONV_4_IFM_CH> input_channel = 0;

	//			input_stream.write(input_channel);
	//		}
	//	}
	//}

	//deconv522(input_stream, output_stream, 1);

	//for(int i=0; i < 16*16; i++){
	//	output_stream.read();
	//}
	//return 0;
}

int main(){
	return test_conv2d_layer0();
	//return test_deconv2d_layer4();
}


