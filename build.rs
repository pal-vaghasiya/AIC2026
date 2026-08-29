fn main() {
    // Compile the C++ library using CMake
    let dst = cmake::Config::new(".")
        .build_target("controlplane_cpp")
        .build();

    // Link the compiled C++ static library
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=controlplane_cpp");

    // Link C++ standard library and system dependencies
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
    #[cfg(not(target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp"); // OpenMP dependency
    }

    // Link ONNX Runtime library
    // If not found in default paths, CMake will build or link to /usr/local/lib/libonnxruntime.so
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // Re-run this build script if C++ files or CMake files change
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=cpp/src/onnx_classifier.cpp");
    println!("cargo:rerun-if-changed=cpp/src/llama_validator.cpp");
    println!("cargo:rerun-if-changed=cpp/src/ffi_bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/include/onnx_classifier.h");
    println!("cargo:rerun-if-changed=cpp/include/llama_validator.h");
}
