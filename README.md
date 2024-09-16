## GPU-accelerated MobileNet on Electron

This project is a proof-of-concept to demonstrate that GPU acceleration of TensorFlow models on non-Nvidia systems is usable. The goal is to get this to work on a Raspberry Pi 5 so that accelerated models can be ran without an external accelerator model. I am using MobileNet because we happened to have it lying around.

#### Installation:  
`git clone --depth 1 https://github.com/geoffreycs/detector`  
`cd detector`  
`npm install`

#### Usage:  
WebGPU backend: `npm run webgpu`  
WebGL backend: `npm run webgl`  
CPU-only backend: `npm run wasm`  

#### Design
This uses Electron to run TensorFlow.js. Electron instead of plain Node.JS is needed to allow the WebGPU and WebGL backends to run. Google's repository for TensorFlow.js does contain some work for using headless OpenGL to accelerate models on plain Node.JS, but the work seems to have stalled four years ago (and is Linux-only).

#### Known issues:  
The WebGPU backend, despite nominally being the fastest, has the most lag due to a bottleneck in copying data to and from the GPU. However, when deployed onto a device with much more constrained CPU resources (RPi 5), it should reveal itself to be much faster than CPU-only execution, I/O constraints notwithstanding.