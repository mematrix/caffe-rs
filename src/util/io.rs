use std::fs::{File};

use protobuf::{Message, CodedInputStream};
use protobuf::text_format::print_to_string;


pub fn read_proto_from_text_file<T: Message>(filename: &str, proto: &mut T) -> bool {
    // todo: protobuf impl not support text_format currently. use binary read instead.
    // todo: user should transform the .prototxt file into a binary format.
    let mut file = File::open(filename).unwrap();
    let mut istream = CodedInputStream::new(&mut file);
    let success = proto.merge_from(&mut istream);
    success.is_ok()
}

#[inline]
pub fn read_proto_from_text_file_or_die<T: Message>(filename: &str, proto: &mut T) {
    let r = read_proto_from_text_file(filename, proto);
    assert!(r);
}

pub fn write_proto_to_text_file<T: Message>(proto: &T, filename: &str) {
    // todo: protobuf impl only support serialize to string currently. use OutputStream in the future.
    let msg = print_to_string(proto);
    std::fs::write(filename, msg).unwrap();
}

pub fn read_proto_from_binary_file<T: Message>(filename: &str, proto: &mut T) -> bool {
    let mut file = File::open(filename).unwrap();
    let mut istream = CodedInputStream::new(&mut file);
    let success = proto.merge_from(&mut istream);
    success.is_ok()
}

#[inline]
pub fn read_proto_from_binary_file_or_die<T: Message>(filename: &str, proto: &mut T) {
    let r = read_proto_from_binary_file(filename, proto);
    assert!(r);
}

pub fn write_proto_to_binary_file<T: Message>(proto: &T, filename: &str) {
    let mut file = File::create(filename).unwrap();
    proto.write_to_writer(&mut file).unwrap();
}
