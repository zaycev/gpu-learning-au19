# Sample code for GPU learning group

Repository of code samples for GPU programming class with Rust and GFX-HAL.
The code here uses Metal backend and should run on OSX. With a slight modification
it should be possible to use Vulkan backend and run it on every other supported 
platform.

## 01 Hello World, Swapchain, Framebuffer, Render Pass

Here we just initialize the GPU API, a swap-chain, its buffers and the render pass. Then during the main loop we grab the cursor coordinates from window events, translate them into the RGB color and send to graphics engine to clear the screen.

How to run:
```
cd 01_hello_world
cargo run
```

![gif](https://github.com/zaycev/gpu-learning-au19/raw/master/01_hello_world/01_hello_world.gif "")

## 02 Graphics Pipeline, Vertex Buffer, Shaders

![gif](https://github.com/zaycev/gpu-learning-au19/raw/master/02_triangle/02_triangle.gif "")

## 03 Coordinates, Depth Buffer, Uniform Buffers