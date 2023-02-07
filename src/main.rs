mod cuda;

mod dispatcher;

fn main() {
    cuda::init(0);
    let ctx = cuda::CuContext::init(0);
    let stream = cuda::CudaStream::create_default();

    let (x_cu, y_cu) = {
        let mut x = Vec::<f32>::new();
        let mut y = Vec::<f32>::new();
        for i in 0..16_000_000 {
            x.push(i as f32);
            y.push(i as f32)
        }

        (
            cuda::DeviceBuf::alloc_from(&x),
            cuda::DeviceBuf::alloc_from(&y),
        )
    };

    let (gX, gY, gZ, num_warps): (u32, u32, u32, u32) = (32, 1, 1, 3);
    let num_elem = 16_000_000;
    let cpu_res = unsafe {
        let res = cuda::DeviceBuf::alloc::<f32>(16_000_000);
        dispatcher::tt_vec_add_64(
            stream.stream,
            gX,
            gY,
            gZ,
            num_warps,
            x_cu.data_ptr,
            y_cu.data_ptr,
            res.data_ptr,
            num_elem,
        );
        res.allocate_on_cpu::<f32>()
    };

    for (idx, v) in cpu_res.iter().enumerate().take(10) {
        println!("{}): {}", idx, v)
    }

    let cpu_res = unsafe {
        let res = cuda::DeviceBuf::alloc::<f32>(16_000_000);
        dispatcher::tt_vec_add_128(
            stream.stream,
            gX,
            gY,
            gZ,
            num_warps,
            x_cu.data_ptr,
            y_cu.data_ptr,
            res.data_ptr,
            num_elem,
        );
        res.allocate_on_cpu::<f32>()
    };

    for (idx, v) in cpu_res.iter().enumerate().take(10) {
        println!("{}): {}", idx, v)
    }
}
