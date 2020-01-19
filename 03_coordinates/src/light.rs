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

impl LightSource {
    pub fn empty() -> Self {
        Self {
            xyz: [0.0, 0.0, 0.0, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }
    pub fn white(xyz: [f32; 4]) -> Self {
        Self {
            xyz,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
    pub fn red(xyz: [f32; 4]) -> Self {
        Self {
            xyz,
            color: [1.0, 0.0, 0.0, 1.0],
        }
    }
    pub fn green(xyz: [f32; 4]) -> Self {
        Self {
            xyz,
            color: [0.0, 1.0, 0.0, 1.0],
        }
    }
    pub fn blue(xyz: [f32; 4]) -> Self {
        Self {
            xyz,
            color: [0.0, 0.0, 1.0, 1.0],
        }
    }
}
