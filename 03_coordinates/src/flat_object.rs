pub trait FlatObject {
    fn stride_size() -> u32;
}

pub trait FlatObjectContainer<O:FlatObject> {
    fn flat_size(&self) -> u32;
    fn flat_ptr(&self) -> *const u8;
}