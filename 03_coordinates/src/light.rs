extern crate gfx_hal as hal;

use std::mem;

use crate::buffer_object::{BlocksSource, FlatBlock};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LightSource {
    pub xyz: [f32; 4],
    pub color: [f32; 4],
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
            xyz: xyz,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
    pub fn red(xyz: [f32; 4]) -> Self {
        Self {
            xyz: xyz,
            color: [1.0, 0.0, 0.0, 1.0],
        }
    }
    pub fn green(xyz: [f32; 4]) -> Self {
        Self {
            xyz: xyz,
            color: [0.0, 1.0, 0.0, 1.0],
        }
    }
    pub fn blue(xyz: [f32; 4]) -> Self {
        Self {
            xyz: xyz,
            color: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

impl FlatBlock for LightSource {
    fn stride_size() -> u32 {
        return mem::size_of::<Self>() as u32;
    }
}

pub struct LightSources {
    pub sources: [LightSource; 10],
}

impl BlocksSource<LightSource> for LightSources {
    fn len(&self) -> u32 {
        self.sources.len() as u32
    }

    fn ptr(&self) -> *const u8 {
        &self.sources[0] as *const _ as *const u8
    }
}