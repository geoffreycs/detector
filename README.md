## GPU-accelerated MobileNet on Electron - WebSocket streaming branch

This project is a proof-of-concept to demonstrate that GPU acceleration of TensorFlow models on non-Nvidia systems is usable. The goal is to get this to work on a Raspberry Pi 5 so that accelerated models can be ran without an external accelerator model. I am using MobileNet because we happened to have it lying around.

### Requirements:
This branch requires a stream being broadcast over the network by [`ffmpeg-ws-relay`](https://github.com/geoffreycs/ffmpeg-ws-relay-modernized). The default IP address is set to something you probably will not want to use, so you can either edit `window.js` on line 107 or just punch in a new value every time you run the program.

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