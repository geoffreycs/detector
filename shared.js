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
    let labels = [];
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
        var index = 0;
        const arrayLength = arrayIn.length;
        let tempArray = [];
        for (index = 0; index < arrayLength; index += 4) {
            tempArray.push(arrayIn.slice(index, index + 4));
        }
        return tempArray;
    }
    return chunkArray;
}
exports.chunkArray = ArrayChunk();

/**
 * @param {Number} dw 
 * @param {Number} dh 
 */
exports.Reformatter = (dw, dh) => {
    /**
     * @param {Float32Array} box_raw
     * @returns {{x: Number, y: Number, w: Number, h: Number}}
     */
    return (box_raw) => {
        return {
            x: dw * box_raw[1],
            y: dh * box_raw[0],
            w: dw * (box_raw[3] - box_raw[1]),
            h: dh * (box_raw[2] - box_raw[0])
        }
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