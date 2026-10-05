"""Torch-owned tensors, deterministic tile reductions, bundled NVRTC on Windows."""

import ctypes as ct
from functools import lru_cache
from pathlib import Path

import torch

CUDA = r"""
__device__ float load_value(const void* data,int index,int bf16) {
    return bf16 ? __uint_as_float((unsigned int)((const unsigned short*)data)[index]<<16)
                : ((const float*)data)[index];
}
__device__ void store_value(void* data,int index,int bf16,float value) {
    if (!bf16) { ((float*)data)[index]=value; return; }
    unsigned int bits=__float_as_uint(value);
    unsigned short result=((bits&0x7fffffffU)>0x7f800000U) ? 0x7fc0U
                          : (unsigned short)((bits+0x7fffU+((bits>>16)&1U))>>16);
    ((unsigned short*)data)[index]=result;
}
__device__ int shift_value(int k) { return k==0 ? 1 : k==1 ? 17 : 129; }
__device__ int neighbor_column(int column,int k,int width,int self_gate) {
    return self_gate ? column : (column+shift_value(k))%width;
}
__device__ void activation(float u,float& feature,float& derivative) {
    float denominator=1.0f+expf(-u),s=1.0f/denominator;
    feature=u/denominator;derivative=s*(1.0f+u*(1.0f-s));
}
extern "C" __global__ void curve_round_check(const float* input,void* output,int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index<count) store_value(output,index,1,input[index]);
}
extern "C" __global__ void curve_forward(
    const void* z,const float* a,const float* b,const float* c,const float* e,
    void* output,int rows,int width,int self_gate,float mean,float scale,int bf16) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=rows*width) return;
    int column=index%width,base=index-column;float local=load_value(z,index,bf16),sum=0;
    for (int k=0;k<3;k++) {
        int parameter=k*width+column;
        float remote=load_value(z,base+neighbor_column(column,k,width,self_gate),bf16);
        float feature,derivative;activation(a[parameter]*local+b[parameter],feature,derivative);
        sum+=feature*(c[parameter]*remote+e[parameter]);
    }
    store_value(output,index,bf16,scale*(sum/1.7320508075688772f-mean));
}
extern "C" __global__ void curve_input_gradient(
    const void* z,const void* gradient,const float* a,const float* b,const float* c,const float* e,
    void* output,int rows,int width,int self_gate,float multiplier,int bf16) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=rows*width) return;
    int column=index%width,base=index-column;float local=load_value(z,index,bf16);
    float t=load_value(gradient,index,bf16)*multiplier,sum=0;
    for (int k=0;k<3;k++) {
        int parameter=k*width+column;
        float neighbor=load_value(z,base+neighbor_column(column,k,width,self_gate),bf16);
        float feature,derivative;activation(a[parameter]*local+b[parameter],feature,derivative);
        sum+=t*a[parameter]*derivative*(c[parameter]*neighbor+e[parameter]);
        int previous=self_gate ? column : (column+width-shift_value(k)%width)%width;
        int previous_parameter=k*width+previous;
        float previous_z=load_value(z,base+previous,bf16),previous_feature,unused;
        activation(a[previous_parameter]*previous_z+b[previous_parameter],previous_feature,unused);
        float previous_t=load_value(gradient,base+previous,bf16)*multiplier;
        sum+=previous_t*previous_feature*c[previous_parameter];
    }
    store_value(output,index,bf16,sum);
}
extern "C" __global__ void curve_coefficient_gradient(
    const void* z,const void* gradient,const float* a,const float* b,const float* c,const float* e,
    float* partial,int rows,int width,int self_gate,float multiplier,int bf16) {
    // Eight token lanes by 32 channels, four tokens per lane: 32-token tile.
    int channel=blockIdx.x*32+threadIdx.x%32,lane=threadIdx.x/32;
    float sums[12];for (int p=0;p<12;p++) sums[p]=0;
    if (channel<width) {
        for (int j=0;j<4;j++) {
            int row=blockIdx.y*32+lane+j*8;if (row>=rows) continue;
            int base=row*width;float local=load_value(z,base+channel,bf16);
            float t=load_value(gradient,base+channel,bf16)*multiplier;
            for (int k=0;k<3;k++) {
                int parameter=k*width+channel;
                float remote=load_value(z,base+neighbor_column(channel,k,width,self_gate),bf16);
                float feature,derivative;activation(a[parameter]*local+b[parameter],feature,derivative);
                float db=t*derivative*(c[parameter]*remote+e[parameter]);
                sums[k]+=local*db;sums[3+k]+=db;
                sums[6+k]+=t*feature*remote;sums[9+k]+=t*feature;
            }
        }
    }
    __shared__ float shared[12][256];
    for (int p=0;p<12;p++) shared[p][threadIdx.x]=sums[p];
    __syncthreads();
    for (int step=4;step>0;step/=2) {
        if (lane<step) for (int p=0;p<12;p++)
            shared[p][threadIdx.x]+=shared[p][threadIdx.x+step*32];
        __syncthreads();
    }
    if (lane==0 && channel<width) for (int p=0;p<12;p++)
        partial[(blockIdx.y*12+p)*width+channel]=shared[p][threadIdx.x];
}
"""


def require_success(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with CUDA/NVRTC status {code}")


class Kernels:
    def __init__(self, device_index):
        library = Path(torch.__file__).parent / "lib"
        candidates = list(library.glob("nvrtc64_*_0.dll"))
        if len(candidates) != 1:
            raise RuntimeError("Expected one bundled Windows NVRTC library")
        self.nvrtc = ct.CDLL(str(candidates[0]))
        self.driver = ct.WinDLL("nvcuda.dll")
        self.nvrtc.nvrtcCreateProgram.argtypes = [
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_char_p,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
        ]
        self.nvrtc.nvrtcCompileProgram.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)]
        for name in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
            getattr(self.nvrtc, name).argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
        for name in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
            getattr(self.nvrtc, name).argtypes = [ct.c_void_p, ct.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes = [ct.POINTER(ct.c_void_p)]
        self.driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [
            ct.POINTER(ct.c_void_p),
            ct.c_void_p,
            ct.c_char_p,
        ]
        self.driver.cuLaunchKernel.argtypes = [
            ct.c_void_p,
            *([ct.c_uint] * 7),
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
        ]
        self.driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
        self.driver.cuCtxPushCurrent_v2.argtypes = [ct.c_void_p]
        self.driver.cuCtxPopCurrent_v2.argtypes = [ct.POINTER(ct.c_void_p)]
        self.context = ct.c_void_p()
        require_success(self.driver.cuCtxGetCurrent(ct.byref(self.context)), "get Torch context")
        if not self.context.value:
            raise RuntimeError("Torch CUDA context must exist before compiling kernels")
        program = ct.c_void_p()
        require_success(
            self.nvrtc.nvrtcCreateProgram(
                ct.byref(program), CUDA.encode(), b"channel_curve.cu", 0, None, None
            ),
            "create program",
        )
        major, minor = torch.cuda.get_device_capability(device_index)
        options = (ct.c_char_p * 3)(
            f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++14", b"--fmad=false"
        )
        try:
            code = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
            if code:
                size = ct.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
                log = ct.create_string_buffer(size.value)
                self.nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(f"CUDA compilation failed: {log.value.decode()}")
            size = ct.c_size_t()
            require_success(self.nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), "PTX size")
            ptx = ct.create_string_buffer(size.value)
            require_success(self.nvrtc.nvrtcGetPTX(program, ptx), "PTX output")
            self.module = ct.c_void_p()
            require_success(self.driver.cuModuleLoadData(ct.byref(self.module), ptx), "load module")
        finally:
            require_success(self.nvrtc.nvrtcDestroyProgram(ct.byref(program)), "destroy program")
        self.functions = {}
        for name in (
            "curve_forward",
            "curve_input_gradient",
            "curve_coefficient_gradient",
            "curve_round_check",
        ):
            function = ct.c_void_p()
            require_success(
                self.driver.cuModuleGetFunction(ct.byref(function), self.module, name.encode()),
                "find kernel",
            )
            self.functions[name] = function

    def launch(self, name, grid, arguments, device):
        values = [
            ct.c_void_p(v.data_ptr()) if isinstance(v, torch.Tensor) else v for v in arguments
        ]
        pointers = (ct.c_void_p * len(values))(*(ct.cast(ct.byref(v), ct.c_void_p) for v in values))
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        # Autograd can call Python on a different OS thread; reuse Torch's existing
        # context for this launch and restore the thread's previous context.
        require_success(self.driver.cuCtxPushCurrent_v2(self.context), "push Torch context")
        try:
            require_success(
                self.driver.cuLaunchKernel(
                    self.functions[name], *grid, 256, 1, 1, 0, stream, pointers, None
                ),
                "launch " + name,
            )
        finally:
            restored = ct.c_void_p()
            require_success(
                self.driver.cuCtxPopCurrent_v2(ct.byref(restored)), "restore caller context"
            )


@lru_cache(maxsize=4)
def kernels(device_index):
    return Kernels(device_index)


def validate(z, a, b, c, e):
    if (
        not z.is_cuda
        or z.dtype not in (torch.float32, torch.bfloat16)
        or not z.is_contiguous()
        or z.ndim < 2
    ):
        raise ValueError("CUDA curves require contiguous FP32/BF16 token-by-channel tensors")
    width = z.shape[-1]
    if width < 1:
        raise ValueError("CUDA curve width must be positive")
    rows = z.numel() // width
    if rows < 1 or rows > 32 * 65535 or rows * width >= 2**31:
        raise ValueError("Invalid CUDA curve shape")
    for p in (a, b, c, e):
        if (
            p.device != z.device
            or p.dtype != torch.float32
            or p.shape != (3, width)
            or not p.is_contiguous()
        ):
            raise ValueError("Curve coefficients must be contiguous CUDA FP32 [3,width]")
    return rows, width


def forward(z, a, b, c, e, self_gate, mean, scale):
    rows, width = validate(z, a, b, c, e)
    output = torch.empty_like(z)
    with torch.cuda.device(z.device):
        kernels(z.device.index).launch(
            "curve_forward",
            ((rows * width + 255) // 256, 1, 1),
            [
                z,
                a,
                b,
                c,
                e,
                output,
                ct.c_int(rows),
                ct.c_int(width),
                ct.c_int(self_gate),
                ct.c_float(mean),
                ct.c_float(scale),
                ct.c_int(z.dtype == torch.bfloat16),
            ],
            z.device,
        )
    return output


def backward(z, gradient, a, b, c, e, self_gate, scale):
    rows, width = validate(z, a, b, c, e)
    gradient = gradient.contiguous()
    if gradient.shape != z.shape or gradient.dtype != z.dtype or gradient.device != z.device:
        raise ValueError("Invalid incoming CUDA curve gradient")
    dz = torch.empty_like(z)
    tiles = (rows + 31) // 32
    partial = torch.empty((tiles, 12, width), device=z.device, dtype=torch.float32)
    tail = [
        ct.c_int(rows),
        ct.c_int(width),
        ct.c_int(self_gate),
        ct.c_float(scale / (3**0.5)),
        ct.c_int(z.dtype == torch.bfloat16),
    ]
    with torch.cuda.device(z.device):
        program = kernels(z.device.index)
        program.launch(
            "curve_input_gradient",
            ((rows * width + 255) // 256, 1, 1),
            [z, gradient, a, b, c, e, dz, *tail],
            z.device,
        )
        program.launch(
            "curve_coefficient_gradient",
            ((width + 31) // 32, tiles, 1),
            [z, gradient, a, b, c, e, partial, *tail],
            z.device,
        )
        coefficients = partial.sum(0).reshape(4, 3, width)
    return dz, *coefficients.unbind(0)
