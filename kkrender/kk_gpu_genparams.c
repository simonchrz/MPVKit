// kk_gpu_genparams.c — native Color-Parameter-Generatoren (libplacebo-Drop-Vorbereitung).
// Alle Matrizen EINMAL aus libplacebo gedumpt + eingebacken (exakt, kein Formel-Bug):
//  - SDR-YUV->RGB-Decode (pl_color_repr_decode) für 601/709/240M/2020-NC × limited/full, 8-bit
//  - Primaries->BT.709 (pl_get_color_mapping_matrix), relative-colorimetric
//  - IPT-Matrizen (pl_ipt_*) für BT.2020
// Ersetzt die pl_*-Generator-Calls im Hook; A/B-verifizierbar solange libplacebo gelinkt.
#include <string.h>

// [sys][full] -> 12 floats (9 Matrix row-major + 3 Offset). sys: 0=601,1=709,2=240M,3=2020NC.
static const float SDR_DEC[4][2][12] = {
  { {1.16438353f,0.0f,1.59602678f, 1.16438353f,-0.39176226f,-0.81296760f, 1.16438353f,2.01723218f,0.0f, -0.87420219f,0.53166777f,-1.08563077f},
    {1.0f,0.0f,1.40751958f, 1.0f,-0.34549114f,-0.71694779f, 1.0f,1.77897632f,0.0f, -0.70651960f,0.53330266f,-0.89297634f} },
  { {1.16438353f,0.0f,1.79274106f, 1.16438353f,-0.21324860f,-0.53290933f, 1.16438353f,2.11240172f,0.0f, -0.97294509f,0.30148265f,-1.13340223f},
    {1.0f,0.0f,1.58099997f, 1.0f,-0.18806176f,-0.46996728f, 1.0f,1.86290550f,0.0f, -0.79359996f,0.33030477f,-0.93510550f} },
  { {1.16438353f,0.0f,1.79365182f, 1.16438353f,-0.25653285f,-0.54272479f, 1.16438353f,2.07984376f,0.0f, -0.97340220f,0.32813662f,-1.11705935f},
    {1.0f,0.0f,1.58180320f, 1.0f,-0.22623368f,-0.47862345f, 1.0f,1.83419299f,0.0f, -0.79400319f,0.35381064f,-0.92069298f} },
  { {1.16438353f,0.0f,1.67867422f, 1.16438353f,-0.18732612f,-0.65042442f, 1.16438353f,2.14177227f,0.0f, -0.91568798f,0.34745857f,-1.14814508f},
    {1.0f,0.0f,1.48040557f, 1.0f,-0.16520098f,-0.57360262f, 1.0f,1.88880706f,0.0f, -0.74310553f,0.37085044f,-0.94810706f} },
};
// Primaries->709 (9). prim: 0=601-525, 1=601-625, 2=2020. Sonst identity (709).
static const float PRIM709[3][9] = {
  {0.93954235f,0.05018139f,0.01027650f, 0.01777219f,0.96579295f,0.01643495f, -0.00162159f,-0.00436973f,1.00599134f},
  {1.04404354f,-0.04404321f,0.00000005f, -0.00000002f,1.0f,0.0f, 0.0f,0.01179345f,0.98820662f},
  {1.66049135f,-0.58764106f,-0.07284985f, -0.12455055f,1.13289988f,-0.00834941f, -0.01815077f,-0.10057883f,1.11872959f},
};
// IPT (BT.2020), konstant.
const float KK_IPT_RGB2LMS_2020[9] = {0.41203642f,0.52391189f,0.06405497f, 0.16666023f,0.72039515f,0.11294612f, 0.02411236f,0.07547495f,0.90040785f};
const float KK_IPT_LMS2RGB_2020[9] = {3.43681455f,-2.50677371f,0.06995196f, -0.79105836f,1.98360193f,-0.19254486f, -0.02572681f,-0.09914176f,1.12487423f};
const float KK_IPT_LMS2IPT[9]      = {0.40000001f,0.40000001f,0.20000000f, 4.45499992f,-4.85099983f,0.39600000f, 0.80559999f,0.35720000f,-1.16279995f};
const float KK_IPT_IPT2LMS[9]      = {1.0f,0.09756890f,0.20522600f, 1.0f,-0.11387600f,0.13321701f, 1.0f,0.03261510f,-0.67688698f};

// sys: 0=601,1=709,2=240M,3=2020NC; full: 0/1. Schreibt 12 floats (Matrix+Offset).
void kk_sdr_decode_matrix(int sys, int full, float out[12]) {
    if (sys < 0 || sys > 3) sys = 1;        // Default 709
    memcpy(out, SDR_DEC[sys][full ? 1 : 0], 12 * sizeof(float));
}
// prim: 0=601-525,1=601-625,2=2020; sonst identity. Schreibt 9 floats.
void kk_primaries_to709(int prim, float out[9]) {
    if (prim >= 0 && prim <= 2) memcpy(out, PRIM709[prim], 9 * sizeof(float));
    else { static const float I[9] = {1,0,0, 0,1,0, 0,0,1}; memcpy(out, I, sizeof I); }
}
