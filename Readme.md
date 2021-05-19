# Caffe-rs

[![GitHub](https://img.shields.io/badge/GitHub-mematrix/caffe--rs-lightgrey?style=flat&logo=github&color=orange)](https://github.com/mematrix/caffe-rs) 
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

[![Rust](https://github.com/mematrix/caffe-rs/actions/workflows/rust-stable-latest.yml/badge.svg)](https://github.com/mematrix/caffe-rs/actions/workflows/rust-stable-latest.yml)
[![Rust - Nightly](https://github.com/mematrix/caffe-rs/actions/workflows/rust-nightly.yml/badge.svg)](https://github.com/mematrix/caffe-rs/actions/workflows/rust-nightly.yml)
[![Rust - Stable Minimum](https://github.com/mematrix/caffe-rs/actions/workflows/rust-stable-min.yml/badge.svg)](https://github.com/mematrix/caffe-rs/actions/workflows/rust-stable-min.yml)

## Toolchain Required
With **no unstable** feature enabled, this project needs a toolchain version `>=1.51.0` to be built.

## Build

```shell
cd path/to/caffe-rs
cargo build
```

## Usage

### Register custom layer
The library exports two macros:

- `register_layer_class!` which is used to register the Layer class which impls the trait `caffe_rs::layer::CaffeLayer<T>` and has a public fn names `new` with a single `caffe_rs::proto::caffe::LayerParameter` param.
```rust
use caffe_rs::proto::caffe::LayerParameter;
use caffe_rs::blob::BlobType;
use caffe_rs::layer::CaffeLayer;
use std::marker::PhantomData;

struct TestLayer<T: BlobType> {
    phantom: PhantomData<T>,
}

impl<T: BlobType> CaffeLayer<T> for TestLayer<T> {
    // Impl the necessary functions.
}

impl<T: BlobType> TestLayer<T> {
    pub fn new(_param: &LayerParameter) -> Self {
        TestLayer {
            phantom: PhantomData
        }
    }
}

// Note: the name does not contains the trailing 'Layer'.
register_layer_class!(Test);
```

- `register_layer_creator!` which is used to register the custom layer creator function. The creator signature is defined as likes the `test`:
```rust
use caffe_rs::proto::caffe::LayerParameter;
use caffe_rs::layer::SharedLayer;
use caffe_rs::blob::BlobType;

fn test<T: BlobType>(p: &LayerParameter) -> Rc<RefCell<Layer<T>>> {
    unimplemented!();
}

// register `TestLayer` with a creator.
register_layer_creator!(Test, test);
```

**And important**: you need import the dependency crate `paste` to use the register macros.

On your `Cargo.toml`:

```toml
[dependencies]
paste = "1"
```

and import the macro on your crate root (`src/lib.rs` or `src/main.rs`):

```rust
#[macro_use] extern crate paste;
```

## License
**Caffe-rs** is released under the [BSD 2-Clause license](https://github.com/mematrix/caffe-rs/blob/master/LICENSE) that is the same as the [Caffe project](https://github.com/BVLC/caffe/).
