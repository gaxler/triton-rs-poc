Initial PoC for using triton kernels in a Rust code base.

[OpenAI’s Triton](https://github.com/openai/triton) is a tool for creating GPU kernels using Python. Triton kernels can be written in Python and executed using a JIT compiler within the script. The tool is currently being developed to include an Ahead of Time compiler, which will convert Triton kernels into C source code.

Triton’s AoT compilation is a work in progress. This repo is based on an unstable draft pull request. This intended as a demonstration, everything in here was quickly duck taped together. So you can expect horrible failures on your machine 😅

Quick overview of the flow of this PoC : Install Triton (if needed), Compile a simple kernel, bind the resulting C code to Rust, launch the kernel on a GPU using CUDA driver API.