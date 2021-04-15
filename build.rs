use std::env;
use std::fs;
use std::path::Path;

use protoc_rust::Customize;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let proto_gen_dir = format!("{}/proto_gen", out_dir);

    if Path::new(&proto_gen_dir).exists() {
        fs::remove_dir_all(&proto_gen_dir).unwrap();
    }

    fs::create_dir(&proto_gen_dir).unwrap();

    protoc_rust::Codegen::new()
        .customize(Customize {
            gen_mod_rs: Some(true),
            carllerche_bytes_for_bytes: Some(true),
            carllerche_bytes_for_string: Some(true),
            ..Default::default()
        })
        .out_dir(proto_gen_dir)
        .input("src/proto/caffe.proto")
        .include("src/proto")
        .run()
        .expect("protoc");
}
