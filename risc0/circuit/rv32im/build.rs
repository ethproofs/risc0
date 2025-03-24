extern crate qapi_codegen;

use std::{env, path};

fn main() {
    let out_dir = path::Path::new(&env::var("OUT_DIR").unwrap()).join("qmp.rs");
    let schema_dir = "/home/rich/qemu/qapi";
    for inc in qapi_codegen::codegen(schema_dir, out_dir, "QmpCommand".into()).unwrap() {
        println!("rerun-if-changed={}", inc.display());
    }
}
