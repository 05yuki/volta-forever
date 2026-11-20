//! The libraries to generate: for each CUDA 13 library a program asks for, the CUDA 12.9 library that
//! serves it, the headers that declare its functions, and how it reports "not supported".

/// How a library reports a status, and the enumerator it uses for "not supported".
pub struct Status {
    pub ty: &'static str,
    pub not_supported: &'static str,
}

pub struct LibSpec {
    /// Short name (output file names, the environment variable VF_FWD_<KEY>).
    pub key: &'static str,
    /// The CUDA 13 file (and SONAME) to generate.
    pub new_name: &'static str,
    /// The CUDA 12.9 library that serves it, found through the loader by this name.
    pub old_name: &'static str,
    /// Header groups, each compiled as one C file (headers that cannot be included together are in
    /// separate groups). The first group holds the status type.
    pub headers: &'static [&'static [&'static str]],
    /// Macros defined when the headers are read (the experimental parts of the API are behind one).
    pub defines: &'static [&'static str],
    pub status: Option<Status>,
    /// Exported names that begin with the first string are the same entry points under the second
    /// (internal entry points that carry the library's version number).
    pub rename_prefix: Option<(&'static str, &'static str)>,
    /// Undocumented exports whose name begins with one of these are taken to return the status type.
    pub status_prefixes: &'static [&'static str],
    /// The library has a `...ForStream` family that the 12.9 library answers without the stream.
    pub for_stream: bool,
}

const NPP: &[&[&str]] = &[&["npp.h"]];

macro_rules! npp {
    ($key:expr, $new:expr, $old:expr) => {
        LibSpec {
            key: $key,
            new_name: $new,
            old_name: $old,
            headers: NPP,
            defines: &[],
            status: Some(Status {
                ty: "NppStatus",
                not_supported: "NPP_NOT_SUPPORTED_MODE_ERROR",
            }),
            rename_prefix: None,
            status_prefixes: &["nppi", "npps"],
            for_stream: false,
        }
    };
}

pub const LIBS: &[LibSpec] = &[
    LibSpec {
        key: "cublas",
        new_name: "libcublas.so.13",
        old_name: "libcublas.so.12",
        headers: &[&["cublas_v2.h", "cublasXt.h"], &["cublas.h"]],
        defines: &[],
        status: Some(Status {
            ty: "cublasStatus_t",
            not_supported: "CUBLAS_STATUS_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["cublas"],
        for_stream: false,
    },
    LibSpec {
        key: "cublasLt",
        new_name: "libcublasLt.so.13",
        old_name: "libcublasLt.so.12",
        headers: &[&["cublasLt.h"]],
        defines: &[],
        status: Some(Status {
            ty: "cublasStatus_t",
            not_supported: "CUBLAS_STATUS_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["cublasLt"],
        for_stream: true,
    },
    LibSpec {
        key: "cufft",
        new_name: "libcufft.so.12",
        old_name: "libcufft.so.11",
        // cufft_device.h is C++ (default arguments): its functions are not read.
        headers: &[&["cufft.h", "cufftXt.h"]],
        defines: &[],
        status: Some(Status {
            ty: "cufftResult",
            not_supported: "CUFFT_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["cufft"],
        for_stream: false,
    },
    LibSpec {
        key: "cufftw",
        new_name: "libcufftw.so.12",
        old_name: "libcufftw.so.11",
        headers: &[&["cufftw.h"]],
        defines: &[],
        // The functions of cuFFTW return FFTW types, not a status.
        status: None,
        rename_prefix: None,
        status_prefixes: &[],
        for_stream: false,
    },
    LibSpec {
        key: "cusolver",
        new_name: "libcusolver.so.12",
        old_name: "libcusolver.so.11",
        headers: &[&[
            "cusolverDn.h",
            "cusolverSp.h",
            "cusolverRf.h",
            "cusolverSp_LOWLEVEL_PREVIEW.h",
        ]],
        defines: &[],
        status: Some(Status {
            ty: "cusolverStatus_t",
            not_supported: "CUSOLVER_STATUS_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["cusolver"],
        for_stream: false,
    },
    LibSpec {
        key: "cusolverMg",
        new_name: "libcusolverMg.so.12",
        old_name: "libcusolverMg.so.11",
        headers: &[&["cusolverMg.h"]],
        defines: &[],
        status: Some(Status {
            ty: "cusolverStatus_t",
            not_supported: "CUSOLVER_STATUS_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["cusolverMg"],
        for_stream: false,
    },
    LibSpec {
        key: "cusparse",
        new_name: "libcusparse.so.12",
        old_name: "libcusparse.so.12",
        headers: &[&["cusparse.h"]],
        defines: &["CUSPARSE_ENABLE_EXPERIMENTAL_API"],
        status: Some(Status {
            ty: "cusparseStatus_t",
            not_supported: "CUSPARSE_STATUS_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["cusparse"],
        for_stream: false,
    },
    LibSpec {
        key: "curand",
        new_name: "libcurand.so.10",
        old_name: "libcurand.so.10",
        headers: &[&["curand.h"]],
        defines: &[],
        // cuRAND has no "not supported" status; the 13.5 library has no function the 12.9 one lacks.
        status: Some(Status {
            ty: "curandStatus_t",
            not_supported: "CURAND_STATUS_INTERNAL_ERROR",
        }),
        rename_prefix: None,
        status_prefixes: &["curand"],
        for_stream: false,
    },
    npp!("nppc", "libnppc.so.13", "libnppc.so.12"),
    npp!("nppial", "libnppial.so.13", "libnppial.so.12"),
    npp!("nppicc", "libnppicc.so.13", "libnppicc.so.12"),
    npp!("nppidei", "libnppidei.so.13", "libnppidei.so.12"),
    npp!("nppif", "libnppif.so.13", "libnppif.so.12"),
    npp!("nppig", "libnppig.so.13", "libnppig.so.12"),
    npp!("nppim", "libnppim.so.13", "libnppim.so.12"),
    npp!("nppist", "libnppist.so.13", "libnppist.so.12"),
    npp!("nppisu", "libnppisu.so.13", "libnppisu.so.12"),
    npp!("nppitc", "libnppitc.so.13", "libnppitc.so.12"),
    npp!("npps", "libnpps.so.13", "libnpps.so.12"),
    LibSpec {
        key: "nvjpeg",
        new_name: "libnvjpeg.so.13",
        old_name: "libnvjpeg.so.12",
        headers: &[&["nvjpeg.h"]],
        defines: &[],
        status: Some(Status {
            ty: "nvjpegStatus_t",
            not_supported: "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
        }),
        rename_prefix: None,
        status_prefixes: &["nvjpeg"],
        for_stream: false,
    },
    LibSpec {
        key: "nvrtc",
        new_name: "libnvrtc.so.13",
        old_name: "libnvrtc.so.12",
        headers: &[&["nvrtc.h"]],
        defines: &[],
        // NVRTC has no "not supported" result; an internal error is the closest.
        status: Some(Status {
            ty: "nvrtcResult",
            not_supported: "NVRTC_ERROR_INTERNAL_ERROR",
        }),
        rename_prefix: None,
        status_prefixes: &["nvrtc", "__nvrtc"],
        for_stream: false,
    },
    LibSpec {
        key: "nvrtc-builtins",
        new_name: "libnvrtc-builtins.so.13.5",
        old_name: "libnvrtc-builtins.so.12.9",
        headers: &[],
        defines: &[],
        status: None,
        rename_prefix: Some(("__nvrtc_builtins__13_5_", "__nvrtc_builtins__12_9_")),
        status_prefixes: &[],
        for_stream: false,
    },
];
