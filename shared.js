const fs = require('fs');
const http = require('http');
const path = require('path');
const { ipcRenderer } = require('electron/renderer');

/**
 * @param {String} name 
 * @returns {String[]}
 */
exports.loadLabels = function (name) {
    /**
     * @type {String[]}
     */
    var labels = [];
    const lines = fs.readFileSync(name).toString().split(/\r?\n/);
    if (lines[1] == '') {
        labels = [lines[0].split("  ")[1]]
    } else {
        lines.forEach(
            /**
             * @param {String} element 
             */
            element => {
                labels.push(element.split("  ")[1]);
            });
    }
    return labels;
}

function ArrayChunk() {
    // Stolen from
    // ourcodeworld.com/articles/read/278/how-to-split-an-array-into-chunks-of-the-same-size-easily-in-javascript
    /**
     * @param {Float32Array} arrayIn 
     * @returns {Float32Array[]}
     */
    function chunkArray(arrayIn) {
        let index = 0 | 0;
        const arrayLength = arrayIn.length;
        const tempArray = [];
        for (index = 0; index < arrayLength; index += 4) {
            tempArray.push(arrayIn.slice(index, index + 4));
        }
        return tempArray;
    }
    return chunkArray;
}
exports.chunkArray = ArrayChunk();

/**
 * @callback setOut
 * @param {Number} w
 * @param {Number} x
 * @param {Number} y
 * @param {Number} z
 */

exports.Reformatter = () => {
    /**
     * @param {null} stdlib
     * @param {{setOut: setOut}} foreign
     * @param {null} heap
     */
    const asmBuilder = function (stdlib, foreign, heap) {
        "use asm";
        const setOut = foreign.setOut;
        var o1 = 0.0;
        var o2 = 0.0;
        var o3 = 0.0;
        var o4 = 0.0;
        /**
         * @param {Number} a 
         * @param {Number} b
         * @param {Number} c
         * @param {Number} d
         */
        function reformat(a, b, c, d) {
            a = +a;
            b = +b;
            c = +c;
            d = +d;

            o1 = 300.0 * b;
            o2 = 300.0 * a;
            o3 = 300.0 * (d - b);
            o4 = 300.0 * (c - a);

            setOut(o1, o2, o3, o4);

        }
        return {
            reformat: reformat
        }
    }

    const out = new Float64Array(new ArrayBuffer(32));
    const setOut = function (w, x, y, z) {
        out[0] = w;
        out[1] = x;
        out[2] = y;
        out[3] = z;
    }
    const module = asmBuilder(null, { setOut }, null);

    /**
     * @param {Float32Array} box_raw
     * @returns {Float64Array<ArrayBuffer>}
     */
    return box_raw => {
        module.reformat(...box_raw);
        return out;
    }
}

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
    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

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