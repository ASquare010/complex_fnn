"""Staged fused basis bank and exact adjoints; shared runtime copy provenance recorded."""

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
__device__ void activation(float u,float& feature,float& derivative) {
    float denominator=1.0f+expf(-u),s=1.0f/denominator;
    feature=u/denominator;derivative=s*(1.0f+u*(1.0f-s));
}
extern "C" __global__ void curve_round_check(const float* input,void* output,int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index<count) store_value(output,index,1,input[index]);
}

__device__ int bank_index(int row,int column,int k,int rows,int size) {
    return ((column/size)*rows+row)*(6*size)+6*(column%size)+k;
}
__device__ bool allowed(int column,int k,int removal) {
    return !(removal==1 && column%2==0) && !(removal==2 && k>=3) && !(removal==3 && k<3);
}
extern "C" __global__ void basis_forward(const void* z,const float* a,const float* b,
    float* bank,int rows,int width,int size,int removal,int bf16) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=rows*width) return;
    int row=index/width,column=index%width;
    float value=load_value(z,index,bf16);
    for(int k=0;k<3;k++) {
        float feature,unused;activation(a[k*width+column]*value+b[k*width+column],feature,unused);
        bank[bank_index(row,column,k,rows,size)]=allowed(column,k,removal)?value*feature:0;
        bank[bank_index(row,column,k+3,rows,size)]=allowed(column,k+3,removal)?feature:0;
    }
}
extern "C" __global__ void basis_input_gradient(const void* z,const float* gradient,
    const float* a,const float* b,void* dz,int rows,int width,int size,int removal,int bf16) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=rows*width) return;
    int row=index/width,column=index%width;float value=load_value(z,index,bf16),sum=0;
    for(int k=0;k<3;k++) {
        float feature,derivative;activation(a[k*width+column]*value+b[k*width+column],feature,derivative);
        float tp=allowed(column,k,removal)?gradient[bank_index(row,column,k,rows,size)]:0;
        float tl=allowed(column,k+3,removal)?gradient[bank_index(row,column,k+3,rows,size)]:0;
        float du=(tp*value+tl)*derivative;
        sum+=tp*feature+du*a[k*width+column];
    }
    store_value(dz,index,bf16,sum);
}
extern "C" __global__ void basis_coefficient_gradient(const void* z,const float* gradient,
    const float* a,const float* b,float* partial,int rows,int width,int size,int removal,int bf16) {
    int column=blockIdx.x*32+threadIdx.x%32,lane=threadIdx.x/32;
    float sums[6];for(int p=0;p<6;p++) sums[p]=0;
    if(column<width) for(int j=0;j<4;j++) {
        int row=blockIdx.y*32+lane+j*8;if(row>=rows) continue;
        float value=load_value(z,row*width+column,bf16);
        for(int k=0;k<3;k++) {
            float feature,derivative;activation(a[k*width+column]*value+b[k*width+column],feature,derivative);
            float tp=allowed(column,k,removal)?gradient[bank_index(row,column,k,rows,size)]:0;
            float tl=allowed(column,k+3,removal)?gradient[bank_index(row,column,k+3,rows,size)]:0;
            float du=(tp*value+tl)*derivative;sums[k]+=value*du;sums[3+k]+=du;
        }
    }
    __shared__ float shared[6][256];
    for(int p=0;p<6;p++) shared[p][threadIdx.x]=sums[p];
    __syncthreads();
    for(int step=4;step>0;step/=2) {
        if(lane<step) for(int p=0;p<6;p++) shared[p][threadIdx.x]+=shared[p][threadIdx.x+step*32];
        __syncthreads();
    }
    if(lane==0 && column<width) for(int p=0;p<6;p++)
        partial[(blockIdx.y*6+p)*width+column]=shared[p][threadIdx.x];
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
                ct.byref(program), CUDA.encode(), b"basis_readout.cu", 0, None, None
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
            "basis_forward",
            "basis_input_gradient",
            "basis_coefficient_gradient",
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


def validate(z, a, b, groups, removal):
    if (
        not z.is_cuda
        or z.dtype not in (torch.float32, torch.bfloat16)
        or not z.is_contiguous()
        or z.ndim < 2
    ):
        raise ValueError("Basis CUDA requires contiguous FP32/BF16 tokens")
    width = z.shape[-1]
    if width < 1 or groups < 1 or width % groups or removal not in (0, 1, 2, 3):
        raise ValueError("Invalid groups/removal")
    rows = z.numel() // width
    if rows < 1 or rows > 32 * 65535 or 6 * rows * width >= 2**31:
        raise ValueError("Invalid bank dimensions")
    for p in (a, b):
        if (
            p.device != z.device
            or p.dtype != torch.float32
            or p.shape != (3, width)
            or not p.is_contiguous()
        ):
            raise ValueError("Templates must be CUDA FP32 [3,width]")
    return rows, width, width // groups


def forward(z, a, b, groups, removal):
    rows, width, size = validate(z, a, b, groups, removal)
    bank = torch.empty((groups, rows, 6 * size), device=z.device, dtype=torch.float32)
    tail = [
        ct.c_int(rows),
        ct.c_int(width),
        ct.c_int(size),
        ct.c_int(removal),
        ct.c_int(z.dtype == torch.bfloat16),
    ]
    with torch.cuda.device(z.device):
        kernels(z.device.index).launch(
            "basis_forward", ((rows * width + 255) // 256, 1, 1), [z, a, b, bank, *tail], z.device
        )
    return bank


def adjoint(z, gradient, a, b, removal, template_gradients):
    groups = gradient.shape[0]
    rows, width, size = validate(z, a, b, groups, removal)
    if (
        gradient.dtype != torch.float32
        or gradient.device != z.device
        or not gradient.is_contiguous()
        or gradient.shape != (groups, rows, 6 * size)
    ):
        raise ValueError("Invalid group-major basis derivative")
    dz = torch.empty_like(z)
    tail = [
        ct.c_int(rows),
        ct.c_int(width),
        ct.c_int(size),
        ct.c_int(removal),
        ct.c_int(z.dtype == torch.bfloat16),
    ]
    with torch.cuda.device(z.device):
        program = kernels(z.device.index)
        program.launch(
            "basis_input_gradient",
            ((rows * width + 255) // 256, 1, 1),
            [z, gradient, a, b, dz, *tail],
            z.device,
        )
        if template_gradients:
            tiles = (rows + 31) // 32
            partial = torch.empty((tiles, 6, width), device=z.device, dtype=torch.float32)
            program.launch(
                "basis_coefficient_gradient",
                ((width + 31) // 32, tiles, 1),
                [z, gradient, a, b, partial, *tail],
                z.device,
            )
            da, db = partial.sum(0).reshape(2, 3, width).unbind(0)
        else:
            da = db = None
    return dz, da, db
