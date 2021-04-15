#[macro_use] extern crate log;

#[macro_use]
mod macros;
mod proto;
mod util;
mod synced_mem;
mod blob;

#[cfg(test)]
mod tests {
    use test_env_log::test;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
