//#define KERNEL_DIM_X 5	//CONV_0_K
//#define KERNEL_DIM_Y 5	//CONV_0_K
//#define SIMD1 1			//CONV_0_SIMD
//#define PE1 2				//CONV_0_PE
//#define MMV1 2			//*
//#define WIDTH 4			//CONV_0_W_BIT
//#define IFM_Channels1 4	//CONV_0_IFM_CH
//#define OFM_Channels1 4	//CONV_0_OFM_CH
//#define IFMDim1_x 8		//CONV_0_IFM_ROW
//#define IFMDim1_y 8		//CONV_0_IFM_COL
//#define OFMDim1_x 4		//CONV_0_OFM_ROW
//#define OFMDim1_y 4		//CONV_0_OFM_COL
//#define STRIDE_x 2		//CONV_0_S
//#define STRIDE_y 2		//CONV_0_S
//#define PADDING 2			//CONV_0_P
//#define INPUT_PRECISION 8 //CONV_0_IN_BIT
//#define TILE1 200			
//#define ACTIVATION_PRECISION 8 //CONV_0_OUT_BIT


// conv_0
#define CONV_0_K 5 
#define CONV_0_S 2 
#define CONV_0_P 2 
#define CONV_0_IFM_CH 3 
#define CONV_0_IFM_ROW 768 
#define CONV_0_IFM_COL 512 
#define CONV_0_OFM_CH 128 
#define CONV_0_OFM_ROW 384 
#define CONV_0_OFM_COL 256 
#define CONV_0_SIMD 3 
#define CONV_0_PE 8 
#define CONV_0_IN_BIT 8 
#define CONV_0_OUT_BIT 8 
#define CONV_0_W_BIT 4 
//#define CONV_0_INC_BIT 18 
//#define CONV_0_BIAS_BIT 24 
#define CONV_0_W_TILES 400 
//#define CONV_0_A_TILES 8 
//#define CONV_0_L_SHIFT 8 

// conv_1
#define CONV_1_K 3 
#define CONV_1_S 1 
#define CONV_1_P 1 
#define CONV_1_IFM_CH 32 
#define CONV_1_IFM_ROW 112 
#define CONV_1_IFM_COL 112 
#define CONV_1_OFM_CH 32 
#define CONV_1_OFM_ROW 112 
#define CONV_1_OFM_COL 112 
#define CONV_1_SIMD 9 
#define CONV_1_PE 1 
#define CONV_1_IN_BIT 4 
#define CONV_1_OUT_BIT 4 
#define CONV_1_W_BIT 4 
#define CONV_1_INC_BIT 20 
#define CONV_1_BIAS_BIT 21 
#define CONV_1_W_TILES 32 
#define CONV_1_A_TILES 32 
#define CONV_1_L_SHIFT 8 

// conv_2
#define CONV_2_K 1 
#define CONV_2_S 1 
#define CONV_2_P 0 
#define CONV_2_IFM_CH 32 
#define CONV_2_IFM_ROW 112 
#define CONV_2_IFM_COL 112 
#define CONV_2_OFM_CH 64 
#define CONV_2_OFM_ROW 112 
#define CONV_2_OFM_COL 112 
#define CONV_2_SIMD 8 
#define CONV_2_PE 2 
#define CONV_2_IN_BIT 4 
#define CONV_2_OUT_BIT 4 
#define CONV_2_W_BIT 4 
#define CONV_2_INC_BIT 15 
#define CONV_2_BIAS_BIT 22 
#define CONV_2_W_TILES 128 
#define CONV_2_A_TILES 32 
#define CONV_2_L_SHIFT 8 

// conv_3
#define CONV_3_K 3 
#define CONV_3_S 2 
#define CONV_3_P 1 
#define CONV_3_IFM_CH 64 
#define CONV_3_IFM_ROW 112 
#define CONV_3_IFM_COL 112 
#define CONV_3_OFM_CH 64 
#define CONV_3_OFM_ROW 56 
#define CONV_3_OFM_COL 56 
#define CONV_3_SIMD 9 
#define CONV_3_PE 1 
#define CONV_3_IN_BIT 4 
#define CONV_3_OUT_BIT 4 
#define CONV_3_W_BIT 4 
#define CONV_3_INC_BIT 16 
#define CONV_3_BIAS_BIT 21 
#define CONV_3_W_TILES 64 
#define CONV_3_A_TILES 64 
#define CONV_3_L_SHIFT 8 

// conv_4
#define CONV_4_K 5 
#define CONV_4_S 2 
#define CONV_4_P 2 
#define CONV_4_IFM_CH 192
#define CONV_4_IFM_ROW 48
#define CONV_4_IFM_COL 32
#define CONV_4_OFM_CH 128
#define CONV_4_OFM_ROW 96
#define CONV_4_OFM_COL 64
#define CONV_4_SIMD 12
#define CONV_4_PE 16
#define CONV_4_IN_BIT 8 
#define CONV_4_OUT_BIT 8 
#define CONV_4_W_BIT 4 
//#define CONV_4_INC_BIT 14 
//#define CONV_4_BIAS_BIT 20 
#define CONV_4_W_TILES 3200
//#define CONV_4_A_TILES 64 
//#define CONV_4_L_SHIFT 8 

// conv_5
#define CONV_5_K 3 
#define CONV_5_S 1 
#define CONV_5_P 1 
#define CONV_5_IFM_CH 128 
#define CONV_5_IFM_ROW 56 
#define CONV_5_IFM_COL 56 
#define CONV_5_OFM_CH 128 
#define CONV_5_OFM_ROW 56 
#define CONV_5_OFM_COL 56 
#define CONV_5_SIMD 9 
#define CONV_5_PE 1 
#define CONV_5_IN_BIT 4 
#define CONV_5_OUT_BIT 4 
#define CONV_5_W_BIT 4 
#define CONV_5_INC_BIT 17 
#define CONV_5_BIAS_BIT 22 
#define CONV_5_W_TILES 128 
#define CONV_5_A_TILES 128 
#define CONV_5_L_SHIFT 8 

// conv_6
#define CONV_6_K 1 
#define CONV_6_S 1 
#define CONV_6_P 0 
#define CONV_6_IFM_CH 128 
#define CONV_6_IFM_ROW 56 
#define CONV_6_IFM_COL 56 
#define CONV_6_OFM_CH 128 
#define CONV_6_OFM_ROW 56 
#define CONV_6_OFM_COL 56 
#define CONV_6_SIMD 8 
#define CONV_6_PE 4 
#define CONV_6_IN_BIT 4 
#define CONV_6_OUT_BIT 4 
#define CONV_6_W_BIT 4 
#define CONV_6_INC_BIT 14 
#define CONV_6_BIAS_BIT 21 
#define CONV_6_W_TILES 512 
#define CONV_6_A_TILES 32 
#define CONV_6_L_SHIFT 8 

// conv_7
#define CONV_7_K 3 
#define CONV_7_S 2 
#define CONV_7_P 1 
#define CONV_7_IFM_CH 128 
#define CONV_7_IFM_ROW 56 
#define CONV_7_IFM_COL 56 
#define CONV_7_OFM_CH 128 
#define CONV_7_OFM_ROW 28 
#define CONV_7_OFM_COL 28 
#define CONV_7_SIMD 3 
#define CONV_7_PE 1 
#define CONV_7_IN_BIT 4 
#define CONV_7_OUT_BIT 4 
#define CONV_7_W_BIT 4 
#define CONV_7_INC_BIT 15 
#define CONV_7_BIAS_BIT 21 
#define CONV_7_W_TILES 384 
#define CONV_7_A_TILES 128 
#define CONV_7_L_SHIFT 8 
