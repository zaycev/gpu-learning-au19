# Sample code for GPU learning group

## 01 Hello World

Here we just initialize the GPU API, a swap-chain, its buffers and the render pass. Then during the main loop we grab the cursor coordinates from window events, translate them into the RGB color and send to graphics engine to clear the screen.

How to run:
```
cd 01_hello_world
cargo run
```

![gif](https://github.com/zaycev/gpu-learning-au19/raw/master/01_hello_world/01_hello_world.gif "")
