#![feature(min_specialization)]

#[macro_use] extern crate log;
#[macro_use] extern crate static_init;

#[macro_use]
mod macros;
mod proto;
mod util;
mod common;
mod synced_mem;
mod blob;
mod layer;
mod layer_factory;
mod net;
mod layers;

#[cfg(test)]
mod tests {
    use test_env_log::test;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
