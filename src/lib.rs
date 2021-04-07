#[macro_use] extern crate log;

mod synced_mem;

#[cfg(test)]
mod tests {
    use test_env_log::test;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
