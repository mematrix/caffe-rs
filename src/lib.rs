#[macro_use] extern crate log;
#[macro_use] extern crate static_init;
#[macro_use] extern crate paste;

#[macro_use]
mod macros;
mod proto;
mod util;
mod common;
mod synced_mem;
mod blob;
mod filler;
mod internal_thread;
mod data_transformer;

mod layer;
#[macro_use]
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
