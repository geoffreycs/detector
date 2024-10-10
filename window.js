const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
// const tflite = require('tfjs-tflite-node');
//const { TFLiteModel } = require('@tensorflow/tfjs-tflite/dist/tflite_model');
const { chunkArray, Reformatter, loadLabels, server, port, onError } = require('./shared');
const fs = require('fs');
// const labels = loadLabels("drone/drone-detect_labels.txt");
const labels = loadLabels("alexandra/alexandrainst_drone_detect_labels.txt");
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
const NodeMediaServer = require('node-media-server');
const nms = new NodeMediaServer({
    rtmp: {
        port: 1935,
        chunk_size: 1,
        gop_cache: false,
        ping: 30,
        ping_timeout: 60
    },
    http: {
        port: 8000,
        allow_origin: location.origin
    }
});

let ratio = Infinity;
const reformat = Reformatter(300, 300);

async function main() {
    try {
        console.log("Creating output renderer");
        /**
         * @type {HTMLCanvasElement}
         */
        const canvas = document.getElementById('display');
        const ctx2 = canvas.getContext('2d');
        ctx2.font = "15px Arial";
        ctx2.fillText("Waiting for webcam", 20, (canvas.height / 2) - 7);
        const desc = document.getElementById("class");

        console.log("Starting HTTP server to self-serve modules on port " + port.toString());
        server.listen(port);

        console.log("Loading model");
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/')
        tflite.setWasmPath('http://127.0.0.1:' + port.toString() + '/');
        // const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("drone/drone-detect1.tflite")).buffer);
        const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("alexandra/alexandrainst_drone_detect.tflite")).buffer);
        console.log("Closing HTTP server")
        server.close();
        server.removeAllListeners();

        console.log("Starting media relay");
        /**
         * @type {HTMLVideoElement}
         */
        const webcam = document.querySelector('#webcam');
        nms.run();

        var videoPlay = false;
        nms.on('postPublish', () => { videoPlay = true; });
        /**
         * @param {Function} resolve 
         */
        const waitPublish = function (resolve) {
            if (videoPlay) {
                resolve();
            }
            else {
                setTimeout(() => waitPublish(resolve), 50);
            }
        }
        /**
         * @returns {Promise<void>}
         */
        const WPwrapper = () => new Promise(
            /**
             * @param {Function} resolve 
             */
            (resolve) => {
                waitPublish(resolve);
            }
        );
        await WPwrapper();
        function createPlayer() {
            const flvPlayer = flvjs.createPlayer({
                type: 'flv',
                url: 'ws://localhost:8000/live/stream.flv'
            });
            flvPlayer.attachMediaElement(webcam);
            flvPlayer.load();
            /**
             * @type {Promise<void>}
             */
            const start = flvPlayer.play();
            start.then(() => {
                nms.on('donePublish', async function () {
                    console.log("Lost video source. Attempting to reacquire.");
                    flvPlayer.pause();
                    flvPlayer.unload();
                    flvPlayer.detachMediaElement(webcam);
                    flvPlayer.destroy();
                    videoPlay = false;
                    await WPwrapper();
                    setTimeout(createPlayer, 100);
                });
            });
            start.catch(onError);
        }
        createPlayer();

        console.log("Waiting for video start");
        /**
         * @param {Function} resolve 
         */
        const checkReady = function (resolve) {
            ratio = Math.min(canvas.width / webcam.videoWidth, canvas.height / webcam.videoHeight);
            if (ratio === Infinity) {
                setTimeout(() => checkReady(resolve), 50);
            } else {
                resolve();
            }
        }
        await new Promise(
            /**
             * @param {Function} resolve 
             */
            (resolve) => {
                checkReady(resolve);
            }
        );
        console.log("Running model");
        const vid_params = [(canvas.height - (webcam.videoHeight * ratio)) / 2, webcam.videoWidth * ratio, webcam.videoHeight * ratio];
        const cvs_params = [canvas.width, canvas.height];

        const expandTensor = (tf.getBackend() == 'webgpu') ?
            /**
             * @param {tf.Tensor3D} source 
             * @returns {tf.Tensor}
             */
            source => {
                const inGPU = source.dataToGPU();
                const expanded = tf.expandDims(inGPU.tensorRef, 0);
                inGPU.tensorRef.dispose();
                return expanded;
            } :
            /**
             * @param {tf.Tensor3D} source 
             * @returns {tf.Tensor}
             */
            source => tf.expandDims(source, 0);

        const doInference = async function () {
            ctx1.drawImage(webcam, 0, 0, webcam.videoWidth, webcam.videoHeight, 0, vid_params[0], vid_params[1], vid_params[2]);
            const img = tf.browser.fromPixels(osc);
            const input = expandTensor(img);

            /**
             * @type {{TFLite_Detection_PostProcess: tf.Tensor,
             * "TFLite_Detection_PostProcess:1": tf.Tensor,
             * "TFLite_Detection_PostProcess:2": tf.Tensor,
             * "TFLite_Detection_PostProcess:3": tf.Tensor }}
             */
            const output = model.predict(input);
            /**
             * @type {Float32Array[]}
             */
            const dataOut = [chunkArray(await output.TFLite_Detection_PostProcess.data()),
            output['TFLite_Detection_PostProcess:1'].dataSync(),
            await output['TFLite_Detection_PostProcess:2'].data(),
            output['TFLite_Detection_PostProcess:3'].dataSync()
            ];

            img.dispose();
            input.dispose();
            output.TFLite_Detection_PostProcess.dispose();
            output['TFLite_Detection_PostProcess:1'].dispose();
            output['TFLite_Detection_PostProcess:2'].dispose();
            output['TFLite_Detection_PostProcess:3'].dispose();

            const converted = reformat(dataOut[0][0]);
            requestAnimationFrame(() => {
                ctx2.clearRect(0, 0, cvs_params[0], cvs_params[1]);
                ctx2.drawImage(osc, 0, 0);
                ctx2.beginPath();
                ctx2.rect(converted.x, converted.y, converted.w, converted.h);
                ctx2.stroke();
            });
            const tag = labels[dataOut[1][0]];
            if (tag == undefined) {
                desc.innerText = "id " + dataOut[1][0].toString() + ", " + String(dataOut[2][0]);
            } else {
                desc.innerText = tag + ", " + String(dataOut[2][0]);
            }
            setTimeout(doInference, 5);
        }

        const runner = doInference();
        runner.catch(onError);
    }
    catch (err) {
        onError(err);
    }
};