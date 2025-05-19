// This is a minimalistic implementation of a diffusion model using CUDA and cuDNN.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <chrono>

//--------------------------------- helpers -----------------------------------
#define ck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{ if (code != cudaSuccess) { fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); exit(code);} }
#define ckCublas(ans) { if(ans!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error %d\n",ans); exit(1);} }
#define ckCudnn(ans)  { if(ans!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"cuDNN error %s\n",cudnnGetErrorString(ans)); exit(1);} }

struct Tensor {
    float *d;               // device ptr
    int n,c,h,w;            // NCHW layout
    size_t bytes() const { return (size_t)n*c*h*w*sizeof(float);} };

static void alloc(Tensor &t,int n,int c,int h,int w){ t.n=n; t.c=c; t.h=h; t.w=w; ck(cudaMalloc(&t.d,t.bytes())); }
static void freeT(Tensor &t){ if(t.d) cudaFree(t.d); t.d=nullptr; }

//--------------------------- simple random util ------------------------------
__global__ void k_fill_rand(float* x,int n,unsigned long long seed){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ curandStatePhilox4_32_10_t st; curand_init(seed,i,0,&st); x[i]=curand_normal(&st);} }
static void randn(Tensor &t,unsigned long long seed){ int N=t.n*t.c*t.h*t.w; k_fill_rand<<<(N+255)/256,256>>>(t.d,N,seed); }

//-------------------------- time embedding -----------------------------------
__global__ void k_sin_cos_embed(float* out,const float* timesteps,int B,int d){ int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=B*d) return; int b=idx/d; int i=idx% d; float t=timesteps[b]; float inv=powf(10000.f,-2.f*(i/2)/float(d)); float val=t*inv; out[idx]= (i%2==0)?sinf(val):cosf(val);} 
static Tensor time_embed(const std::vector<float>& ts){ int B=ts.size(); int D=128; Tensor t; alloc(t,B,D,1,1); float* h; ck(cudaMallocHost(&h,B*sizeof(float))); memcpy(h,ts.data(),B*sizeof(float)); float* d; ck(cudaMalloc(&d,B*sizeof(float))); ck(cudaMemcpy(d,h,B*sizeof(float),cudaMemcpyHostToDevice)); k_sin_cos_embed<<<(B*D+255)/256,256>>>(t.d,d,B,D); cudaFreeHost(h); cudaFree(d); return t; }

//------------------------------ Linear layer ---------------------------------
struct Linear { int in,out; float* w; float* b; };

static void linear_forward(const Linear& L,const Tensor& x,Tensor& y,cublasHandle_t cb){ // x: [B,in]
    const float alpha=1.f,beta=0.f; ckCublas(cublasSgemm(cb,CUBLAS_OP_T,CUBLAS_OP_N,L.out,x.n,L.in,&alpha,L.w,L.in,x.d,L.in,&beta,y.d,L.out)); // y=(x*W^T)
    // add bias
    int B=x.n; int O=L.out;
    dim3 bl((B*O+255)/256); __global__ void k_bias(float* y,const float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=b[i%O]; } k_bias<<<bl,256>>>(y.d,L.b,B*O);
}

//------------------------------ Conv block ----------------------------------
struct Conv2D { int in,out,k; cudnnFilterDescriptor_t wDesc; cudnnConvolutionDescriptor_t convDesc; float* w; float* b; };

static void conv_forward(const Conv2D& c,const Tensor& x,Tensor& y,cudnnHandle_t ch){
    cudnnTensorDescriptor_t xDesc,yDesc,bDesc; cudnnCreateTensorDescriptor(&xDesc); cudnnCreateTensorDescriptor(&yDesc); cudnnCreateTensorDescriptor(&bDesc);
    cudnnSetTensor4dDescriptor(xDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,x.n,x.c,x.h,x.w);
    int outN,outC,outH,outW; cudnnGetConvolution2dForwardOutputDim(c.convDesc,xDesc,c.wDesc,&outN,&outC,&outH,&outW);
    alloc(y,outN,outC,outH,outW);
    cudnnSetTensor4dDescriptor(yDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,outN,outC,outH,outW);
    cudnnSetTensor4dDescriptor(bDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,outC,1,1);
    size_t ws; cudnnGetConvolutionForwardWorkspaceSize(ch,xDesc,c.wDesc,c.convDesc,yDesc,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,&ws);
    void* workspace; ck(cudaMalloc(&workspace,ws)); const float alpha=1.f,beta=0.f; ckCudnn(cudnnConvolutionForward(ch,&alpha,xDesc,x.d,c.wDesc,c.w,c.convDesc,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,workspace,ws,&beta,yDesc,y.d));
    ckCudnn(cudnnAddTensor(ch,&alpha,bDesc,c.b,&alpha,yDesc,y.d));
    cudaFree(workspace); cudnnDestroyTensorDescriptor(xDesc); cudnnDestroyTensorDescriptor(yDesc); cudnnDestroyTensorDescriptor(bDesc);
}

//------------------------- SiLU activation -----------------------------------
__global__ void k_silu(float* x,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; x[i]=v/(1.f+expf(-v)); }}
static void silu(Tensor& t){ int N=t.n*t.c*t.h*t.w; k_silu<<<(N+255)/256,256>>>(t.d,N);} 

//--------------------- Residual block (Conv‑SiLU‑Conv) ------------------------
struct ResBlock { Conv2D c1,c2; Linear t_proj; int channels; };

static Tensor resblock_forward(const ResBlock& R,const Tensor& x,const Tensor& t_emb,cudnnHandle_t ch,cublasHandle_t cb){ Tensor h1,h2,y;
    conv_forward(R.c1,x,h1,ch); silu(h1);
    conv_forward(R.c2,h1,h2,ch);
    // inject time embedding
    Tensor t_proj; alloc(t_proj,x.n,R.channels,1,1); linear_forward(R.t_proj,t_emb,t_proj,cb);
    int N=x.n*R.channels*h2.h*h2.w; __global__ void k_add(float* y,const float* t,int ch,int spatial){int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return; int b=idx/(ch*spatial); int c=(idx/spatial)%ch; y[idx]+=t[b*ch+c]; }
    int spatial=h2.h*h2.w; k_add<<<(N+255)/256,256>>>(h2.d,t_proj.d,R.channels,spatial);
    // residual
    y=h2; __global__ void k_add2(float* y,const float* x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=x[i]; } int tot=N; k_add2<<<(tot+255)/256,256>>>(y.d,x.d,tot);
    silu(y); freeT(h1); freeT(t_proj); return y; }

//--------------------------- UNet definition ---------------------------------
struct UNet {
    std::vector<ResBlock> down,up;
    Conv2D middle1,middle2,final_conv;
    Linear time_linear1,time_linear2;
};

static Tensor unet_forward(UNet& U,const Tensor& x,const Tensor& t_emb,cudnnHandle_t ch,cublasHandle_t cb){ std::vector<Tensor> skips; Tensor h=x; // downsample path
    for(auto& rb:U.down){ h=resblock_forward(rb,h,t_emb,ch,cb); skips.push_back(h); /* naive 2× downsample */ if(&rb!=&U.down.back()){ Tensor d; alloc(d,h.n,h.c,h.h/2,h.w/2); // nearest
        __global__ void k_down(float* o,const float* i,int B,int C,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x; int N=B*C*(H/2)*(W/2); if(idx>=N) return; int b=idx/((C*(H/2)*(W/2))); int rem=idx%((C*(H/2)*(W/2))); int c=rem/((H/2)*(W/2)); int r=rem%((H/2)*(W/2)); int y=r/(W/2); int x=r%(W/2); int src_idx=((b*C+c)*H+2*y)*W+2*x; o[idx]=i[src_idx]; } k_down<<<(d.n*d.c*d.h*d.w+255)/256,256>>>(d.d,h.d,h.n,h.c,h.h,h.w); freeT(h); h=d; }}
    // bottleneck
    Tensor mid1,mid2; conv_forward(U.middle1,h,mid1,ch); silu(mid1); conv_forward(U.middle2,mid1,mid2,ch); silu(mid2); freeT(mid1);
    h=mid2; // upsample
    for(int i=0;i<U.up.size();++i){ auto& rb=U.up[i]; if(i>0){ /* nearest upsample 2× */ Tensor u; alloc(u,h.n,h.c,h.h*2,h.w*2); __global__ void k_up(float* o,const float* i,int B,int C,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x; int N=B*C*(H*2)*(W*2); if(idx>=N) return; int b=idx/((C*(H*2)*(W*2))); int rem=idx%((C*(H*2)*(W*2))); int c=rem/((H*2)*(W*2)); int r=rem%((H*2)*(W*2)); int y=r/((W*2)); int x=r%((W*2)); int src_idx=((b*C+c)*H+y/2)*W+x/2; o[idx]=i[src_idx]; } k_up<<<(u.n*u.c*u.h*u.w+255)/256,256>>>(u.d,h.d,h.n,h.c,h.h/2,h.w/2); freeT(h); h=u; }
        // concat skip
        Tensor cat; alloc(cat,h.n,h.c+skips.back().c,h.h,h.w); __global__ void k_cat(float* o,const float* a,const float* b,int B,int C1,int C2,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x; int N=B*(C1+C2)*H*W; if(idx>=N) return; int sp=H*W; int b0=idx/(sp*(C1+C2)); int c= (idx/sp)% (C1+C2); int pix=idx%sp; if(c<C1) o[idx]=a[(b0*C1+c)*sp+pix]; else o[idx]=b[(b0*C2+(c-C1))*sp+pix]; } k_cat<<<(cat.n*cat.c*cat.h*cat.w+255)/256,256>>>(cat.d,skips.back().d,h.d,h.n,skips.back().c,h.c,h.h,h.w);
        freeT(skips.back()); skips.pop_back(); freeT(h); h=cat;
        h=resblock_forward(rb,h,t_emb,ch,cb); }
    Tensor out; conv_forward(U.final_conv,h,out,ch); return out; }

//------------------------------ DDIM sampler ---------------------------------
static std::vector<float> make_sigmas(int steps){ std::vector<float> betas(steps); float b0=1e-4,b1=0.02; for(int i=0;i<steps;++i){ betas[i]=b0+(b1-b0)*i/(steps-1);} std::vector<float> alphas(steps); for(int i=0;i<steps;++i) alphas[i]=1-betas[i]; std::vector<float> alphas_cum(steps); float acc=1; for(int i=0;i<steps;++i){ acc*=alphas[i]; alphas_cum[i]=acc;} std::vector<float> sigmas(steps); for(int i=0;i<steps;++i){ sigmas[i]=sqrtf((1-alphas_cum[i])/alphas_cum[i]); } return sigmas; }

static void ddim_sample(UNet& net,int steps,std::mt19937& rng,cudnnHandle_t ch,cublasHandle_t cb){ auto sigmas=make_sigmas(steps); Tensor x0; alloc(x0,1,3,64,64); randn(x0,rng());
    for(int i=steps-1;i>=0;--i){ float t=sigmas[i]; Tensor t_emb=time_embed({t}); Tensor eps=unet_forward(net,x0,t_emb,ch,cb); // predict noise
        // x_{i-1}=x_i - sigma_i*eps + noise
        __global__ void k_update(float* x,const float* e,float sigma,int N){int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx<N){ x[idx]=x[idx]-sigma*e[idx]; }} k_update<<<(x0.n*x0.c*x0.h*x0.w+255)/256,256>>>(x0.d,eps.d,t,x0.n*x0.c*x0.h*x0.w);
        freeT(eps); freeT(t_emb);
        if(i>0){ Tensor n; alloc(n,x0.n,x0.c,x0.h,x0.w); randn(n,rng()); float sigma_prev=sigmas[i-1]; __global__ void k_addnoise(float* x,const float* n,float s,int N){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) x[i]+=s*n[i]; } k_addnoise<<<(x0.n*x0.c*x0.h*x0.w+255)/256,256>>>(x0.d,n.d,sigma_prev,x0.n*x0.c*x0.h*x0.w); freeT(n); }
    }
    // save image
    std::vector<float> h(x0.n*x0.c*x0.h*x0.w); ck(cudaMemcpy(h.data(),x0.d,x0.bytes(),cudaMemcpyDeviceToHost)); std::vector<unsigned char> img(64*64*3); for(int i=0;i<64*64*3;++i){ float v=(h[i]+1.f)*0.5f; v=fminf(fmaxf(v,0.f),1.f); img[i]=(unsigned char)(v*255.f);} stbi_write_png("out.png",64,64,3,img.data(),64*3); freeT(x0);
}

//------------------------------- main ----------------------------------------
int main(int argc,char** argv){ if(argc<3){ printf("Usage: %s weights.bin out.png [steps]\n",argv[0]); return 0;} const char* wfile=argv[1]; const char* outfile=argv[2]; int steps=(argc>3)?atoi(argv[3]):50; unsigned seed=1234; // simplistic arg parse

    cudnnHandle_t ch; cublasHandle_t cb; ckCudnn(cudnnCreate(&ch)); ckCublas(cublasCreate(&cb));
    UNet net; // TODO: load weights from wfile into net structures (exercise for reader)
    std::mt19937 rng(seed);
    auto t0=std::chrono::high_resolution_clock::now(); ddim_sample(net,steps,rng,ch,cb); auto t1=std::chrono::high_resolution_clock::now();
    float ms=std::chrono::duration<float, std::milli>(t1-t0).count(); printf("Generated in %.2f ms\n",ms);
    cudnnDestroy(ch); cublasDestroy(cb); return 0; }

// -----------------------------------------------------------------------------
// This program is purposefully minimal; see README for weight‑packing script and
// training details.  Licensed under MIT.  hack. enjoy. –Ion Linti 2025
