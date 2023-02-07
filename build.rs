extern crate bindgen;

use glob::glob;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn make_sure_triton_installed() {
    if std::env::var_os("DRY_RUN").is_some() {
        println!("cargo:warning=[DRY RUN]: installing triton is not yet implemented");
        return;
    }

    let tt_python = std::env::var_os("TT_PYTHON").expect("No triton python interperter set");
    let output = Command::new(tt_python)
        .arg("-c")
        .arg("import triton; import triton.language as tl")
        .output()
        .unwrap();

    if !output.status.success() {
        println!("cargo:warning=[TRITON] Trying to install triton");
        let output = Command::new("bash")
            .arg("install_ttc.sh")
            .output()
            .expect("Failed to install Triton");

        if !output.status.success() {
            let s = String::from_utf8(output.stderr).unwrap();
            panic!("Failed to install Triton: \n {}", s)
        }
    }
}

fn build_kernels(dispatcher_name: &str, kernel_prefix: Option<&str>) {
    println!("cargo:rerun-if-changed=src/kernels/build.sh");
    let mut cmd = Command::new("bash");
    cmd.arg("src/kernels/build.sh").arg(dispatcher_name);

    if let Some(prefix) = kernel_prefix {
        cmd.arg(prefix);
    }

    let output = cmd.output().expect("tt build failed");

    if output.stderr.len() > 0 {
        println!(
            "cargo:warning=Kernel build.sh strerr: {}",
            String::from_utf8(output.stderr).unwrap()
        );
    }
}

fn find_csources(root: &str) -> impl Iterator<Item = PathBuf> {
    let gpattern = format!("{}/**/*.c", root);
    let f = glob(&gpattern).unwrap();
    f.filter_map(|p| p.map_or(None, |v| Some(v)))
}

fn find_cuda() -> Option<PathBuf> {
    let whereis_out = Command::new("whereis").arg("libcuda.so").output().unwrap();
    let stdout = String::from_utf8(whereis_out.stdout).unwrap();
    stdout
        .trim_end_matches(&['\r', '\n'])
        .split(" ")
        .skip(1)
        .filter_map(|v| Path::new(v).parent())
        .for_each(|pt| println!("cargo:rustc-link-search={}", pt.display()));

    let cuda_path = std::env::var_os("CUDA_PATH")?;
    let cuda_path_buf = PathBuf::from(cuda_path).join("include");

    if !cuda_path_buf.exists() {
        return None;
    }
    Some(cuda_path_buf)
}

fn build_c_interface(c_path: &PathBuf, dispatcher_name: &str) -> Option<PathBuf> {
    let mut builder = cc::Build::new();
    builder
        .files(find_csources(c_path.to_str().unwrap()))
        .include(c_path);

    match find_cuda() {
        Some(cuda_include_path) => {
            builder.include(cuda_include_path.clone());
            builder.compile(dispatcher_name);
            Some(cuda_include_path)
        }
        None => {
            let _msg = "[CUDA]: No CUDA found, compiling cuda dependent code will fail";
            if std::env::var_os("DRY_RUN").is_none() {
                panic!("{}", _msg);
            }
            println!("cargo:warning=[DRY RUN]{}", _msg);
            None
        }
    }
}

fn main() {
    static DISPATCHER_FNAME: &str = "dispatcher";
    static KERNEL_PREFIX: &str = "tt_";
    let out_dir = std::env::var_os("OUT_DIR").expect("Not OUT_DIR set");
    let csrc = std::env::var_os("C_SOURCES_TARGET").expect("No C source target was defined");
    let c_path = Path::new(&out_dir).join(csrc);
    std::fs::create_dir_all(&c_path).unwrap();

    make_sure_triton_installed();
    build_kernels(DISPATCHER_FNAME, Some(KERNEL_PREFIX));
    build_c_interface(&c_path, DISPATCHER_FNAME);

    let mut dispatcher_h = c_path.join(DISPATCHER_FNAME);
    dispatcher_h.set_extension("h");

    let allow_funcs_ptrn = format!("{}.*", KERNEL_PREFIX);

    let bindings = bindgen::Builder::default()
        .header(dispatcher_h.to_str().unwrap())
        .allowlist_function(allow_funcs_ptrn)
        .blocklist_item("cu.*")
        .blocklist_item("CU.*")
        .blocklist_item("_.*")
        .generate()
        .expect("Generation failed");

    let mut tgt_file = Path::new(&out_dir).join(DISPATCHER_FNAME);
    tgt_file.set_extension("rs");
    bindings
        .write_to_file(tgt_file)
        .expect("Error writing bindings");
}
