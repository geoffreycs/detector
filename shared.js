const fs = require('fs');
const http = require('http');
const path = require('path');
const { ipcRenderer } = require('electron/renderer');

/**
 * @callback stdMath
 * @param {Number} x
 * @return {Number}
 */

exports.asmExport = (() => {
    /**
     * @param {{Math: {abs: stdMath}, Float64Array: Float64ArrayConstructor}} stdlib
     * @param {null} foreign
     * @param {ArrayBuffer} heap
     */
    const asmBuilder = function (stdlib, foreign, heap) {
        "use asm";
        const abs = stdlib.Math.abs;
        const work = new stdlib.Float64Array(heap);
        var mX = 0.0;
        var mY = 0.0;
        var w = 0.0;
        var h = 0.0;
        var idx = 0;

        /**
         * @param {Number} a 
         * @param {Number} b
         * @param {Number} c
         * @param {Number} d
         * @param {Number} lastX
         * @param {Number} lastY
         */
        function reformat(a, b, c, d, lastX, lastY) {
            a = +a;
            b = +b;
            c = +c;
            d = +d;
            lastX = +lastX;
            lastY = +lastY;

            mX = ((d + b) / 2.0) * 300.0;
            mY = ((a + c) / 2.0) * 300.0;
            w = 300.0 * (d - b);
            h = 300.0 * (c - a);

            work[0] = 300.0 * b;
            work[1] = 300.0 * a;
            work[2] = w;
            work[3] = h;
            work[4] = mX;
            work[5] = mY;
            work[6] = +abs(mX - lastX);
            work[7] = +abs(mY - lastY);
            work[8] = w * h;
            work[9] = w / h;
        }

        function dimsAvg() {
            work[40] = (+work[10] + +work[11] + +work[12] + +work[13] + +work[14]) / 5.0;
            work[41] = (+work[15] + +work[16] + +work[17] + +work[18] + +work[19]) / 5.0;
            work[42] = (+work[20] + +work[21] + +work[22] + +work[23] + +work[24]) / 5.0;
            work[43] = (+work[25] + +work[26] + +work[27] + +work[28] + +work[29]) / 5.0;
        }

        function setAll() {
            work[10] = +work[0];
            work[11] = +work[0];
            work[12] = +work[0];
            work[13] = +work[0];
            work[14] = +work[0];

            work[15] = +work[1];
            work[16] = +work[1];
            work[17] = +work[1];
            work[18] = +work[1];
            work[19] = +work[1];

            work[20] = +work[2];
            work[21] = +work[2];
            work[22] = +work[2];
            work[23] = +work[2];
            work[24] = +work[2];

            work[25] = +work[3];
            work[26] = +work[3];
            work[27] = +work[3];
            work[28] = +work[3];
            work[29] = +work[3];

            work[30] = +work[4];
            work[31] = +work[4];
            work[32] = +work[4];
            work[33] = +work[4];
            work[34] = +work[4];

            work[35] = +work[5];
            work[36] = +work[5];
            work[37] = +work[5];
            work[38] = +work[5];
            work[39] = +work[5];
        }

        /**
         * @param {Number} conf 
         */
        function accConf(conf) {
            conf = +conf;
            work[((idx + 46) << 3) >> 3] = conf;
            idx = (idx + 1) | 0;
            idx = ((idx | 0) % 5) | 0;
        }

        /**
         * @returns {Number}
         */
        function avgConf() {
            return ((+work[46] + +work[47] + +work[48] + +work[49] + +work[50]) / 5.0);
        }

        return {
            reformat: reformat,
            dimsAvg: dimsAvg,
            setAll: setAll,
            accConf: accConf,
            avgConf: avgConf
        }
    }

    // const mem = new ArrayBuffer(80);
    const mem = new ArrayBuffer(0x1000);

    const module = asmBuilder({ Math: { abs: Math.abs }, Float64Array }, null, mem);

    return {
        /**
         * @param {Float32Array} box_raw
         * @param {Number} lastX
         * @param {Number} lastY
         */
        reformat: (box_raw, lastX, lastY) => {
            module.reformat(box_raw[0], box_raw[1], box_raw[2], box_raw[3], lastX, lastY);
        },
        dimsAvg: module.dimsAvg,
        setAll: module.setAll,
        accConf: module.accConf,
        avgConf: module.avgConf,
        converted: new Float64Array(mem, 0, 10), // 0-9
        x_accum: new Float64Array(mem, 80, 5), // 10-14
        y_accum: new Float64Array(mem, 120, 5), // 15-19
        w_accum: new Float64Array(mem, 160, 5), // 20-24
        h_accum: new Float64Array(mem, 200, 5), // 25-29
        m1_accum: new Float64Array(mem, 240, 5), // 30-34
        m2_accum: new Float64Array(mem, 280, 5), // 35-39
        avgs: new Float64Array(mem, 320, 6) // 40-45
    }
})();

const MIME_TYPES = {
    js: "text/javascript",
    wasm: "application/wasm",
    txt: "text/plain"
};
const assets = path.join(process.cwd(), "./node_modules/@tensorflow/tfjs-tflite/wasm");
const toBool = [() => true, () => false];
const port = Math.round(Math.random() * (10000 - 9000) + 9000);
const prepareFile = async (url) => {
    const paths = [assets, url];
    const filePath = path.join(...paths);
    const pathTraversal = !filePath.startsWith(assets);
    const exists = await fs.promises.access(filePath).then(...toBool);
    const found = !pathTraversal && exists;
    const streamPath = found ? filePath : path.join(process.cwd(), "./404.txt");
    const ext = path.extname(streamPath).substring(1).toLowerCase();
    const stream = fs.createReadStream(streamPath);
    return { found, ext, stream };
};
const server = http.createServer(async (req, res) => {
    try {
        if (req.socket.remoteAddress.includes("127.0.0.1")) {
            const file = await prepareFile(req.url);
            const statusCode = file.found ? 200 : 404;
            const mimeType = MIME_TYPES[file.ext];
            res.writeHead(statusCode, { "Content-Type": mimeType });
            file.stream.pipe(res);
            console.log(`${req.method} ${req.url} ${statusCode}`);
        } else {
            console.log("Ignored request from " + req.socket.remoteAddress);
        }
    }
    catch (err) {
        onError(err);
    }
});
exports.server = server;
exports.port = port;

/**
 * @param {Error} error 
 */
exports.onError = function (error) {
    console.error(error);
    ipcRenderer.send('error');
    //throw "Execution halted due to above error"
}

const BASE_VERTEX_SHADER = `
  attribute vec2 position;
  varying vec2 texCoords;

  void main() {
    texCoords = (position + 1.0) / 2.0;

    texCoords.y = 1.0 - texCoords.y;
    
    gl_Position = vec4(position, 0, 1.0);
  }
`;
const BASE_FRAGMENT_SHADER = `
  precision highp float;
  
  varying vec2 texCoords;
  uniform sampler2D textureSampler;

  void main() {
    vec4 color = texture2D(textureSampler, texCoords);
    gl_FragColor = color;
  }
`;

/**
 * @param {OffscreenCanvas | HTMLCanvasElement} canvas
 */
exports.getGL = function (canvas) {
    /**
     * @type {WebGL2RenderingContext}
     */
    const gl = canvas.getContext("webgl2");
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    // Create our vertex shader
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, BASE_VERTEX_SHADER);
    gl.compileShader(vertexShader);
    // Create our fragment shader
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, BASE_FRAGMENT_SHADER);
    gl.compileShader(fragmentShader);
    // Create our program
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    // Enable the program
    gl.useProgram(program);
    // Bind VERTICES as the active array buffer.
    const VERTICES = new Float32Array([-1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, VERTICES, gl.STATIC_DRAW);
    // Set and enable our array buffer as the program's "position" variable
    const positionLocation = gl.getAttribLocation(program, "position");
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(positionLocation);
    // Create a texture
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    /**
     * @param {HTMLImageElement | ImageBitmap | ImageData | HTMLCanvasElement | HTMLVideoElement} image 
     * @returns {void}
     */
    const module = image => {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    return module;
}