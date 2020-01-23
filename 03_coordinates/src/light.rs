extern crate gfx_hal as hal;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LightSource {
    pub xyz: [f32; 4],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LightInfo {
    pub sources_num: [u32; 4],
    pub junk: [u32; 4],
}