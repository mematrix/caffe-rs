
/// Stub out GPU calls as unavailable.
macro_rules! no_gpu {
    () => {
        assert!(false, "Cannot use GPU in CPU-only Caffe: check mode.");
    };
}
