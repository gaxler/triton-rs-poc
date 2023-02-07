use core::mem::size_of;
use std::alloc::{alloc, Layout};
use std::os::raw::{c_int, c_void};

use cuda_sys::cuda::*;

pub struct CuContext {
    ctx_st: CUctx_st,
    pub ctx: CUcontext,
    pub device: CUdevice,
}

impl CuContext {
    pub fn init(device_id: u32) -> CuContext {
        let mut device: CUdevice = 0;
        let mut ctx_st: CUctx_st = CUctx_st { _address: 0 };
        let mut ctx: CUcontext = &mut ctx_st as *mut _;
        unsafe {
            cuda_fail_dbg(cuDeviceGet(
                &mut device as *mut CUdevice,
                device_id as c_int,
            ));
            cuda_fail_dbg(cuCtxCreate_v2(&mut ctx as *mut CUcontext, 0, device));
        }

        CuContext {
            ctx_st: ctx_st,
            ctx: ctx,
            device: device,
        }
    }
}

impl Drop for CuContext {
    fn drop(&mut self) {
        unsafe {
            cuda_fail_dbg(cuCtxDestroy_v2(self.ctx));
        }
    }
}

pub fn mem_alloc(bytesize: usize) -> CUdeviceptr {
    let mut dptr: CUdeviceptr = 0;
    unsafe {
        let res = cuMemAlloc_v2(&mut dptr as *mut _, bytesize);
        cuda_fail_dbg(res);
    }
    dptr
}

#[inline]
fn alloc_vec<T>(size: usize) -> Option<*mut T> {
    let l = Layout::array::<T>(size).unwrap();
    let ptr = unsafe { alloc(l) };
    if ptr.is_null() {
        return None;
    }
    Some(ptr as *mut T)
}

#[inline]
fn cuda_fail_dbg(res: cudaError_t) {
    if res != cudaError_t::CUDA_SUCCESS {
        cuda_fail_dbg(res);
    }
}

pub struct DeviceBuf {
    pub data_ptr: CUdeviceptr,
    byte_size: usize,
}

impl DeviceBuf {
    pub fn alloc<T>(num_elem: usize) -> DeviceBuf {
        let size = num_elem * size_of::<T>();
        let dptr = mem_alloc(size);
        DeviceBuf {
            data_ptr: dptr,
            byte_size: size,
        }
    }

    pub fn alloc_from<T>(data: &[T]) -> DeviceBuf {
        let ptr = data.as_ptr() as *const c_void;

        let size = size_of::<T>() * data.len();
        let dptr = mem_alloc(size);
        let dev_buf = DeviceBuf::alloc::<T>(data.len());
        unsafe {
            cuda_fail_dbg(cuMemcpyHtoD_v2(dev_buf.data_ptr, ptr, size));
        }
        dev_buf
    }

    pub fn allocate_on_cpu<T>(&self) -> Vec<T> {
        let cap = self.byte_size / size_of::<T>();
        let ptr = alloc_vec::<T>(cap).unwrap();

        let v = unsafe {
            cuda_fail_dbg(cuMemcpyDtoH_v2(
                ptr as *mut c_void,
                self.data_ptr,
                self.byte_size,
            ));
            Vec::from_raw_parts(ptr, cap, cap)
        };
        v
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        unsafe {
            cuda_fail_dbg(cuMemFree_v2(self.data_ptr));
        }
    }
}

pub fn init(flags: u32) {
    unsafe {
        cuda_fail_dbg(cuInit(flags));
    }
}

pub struct CudaStream {
    pub stream: CUstream,
}

impl CudaStream {
    pub fn create_default() -> CudaStream {
        let mut st = CUstream_st { _address: 0 };
        let mut s: CUstream = &mut st as *mut CUstream_st;
        unsafe {
            let def = CUstream_flags_enum::CU_STREAM_DEFAULT as u32;
            cuda_fail_dbg(cuStreamCreate(&mut s as *mut CUstream, def));
        }
        CudaStream { stream: s }
    }
}
