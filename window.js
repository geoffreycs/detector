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

let ratio = Infinity;
let vid_params = [0.0, 0.0, 0.0, 0.0];
const reformat = Reformatter(300, 300);
const urlCreator = window.URL || window.webkitURL;

async function main() {
    try {
        console.log("Creating output renderer");
        /**
         * @type {HTMLCanvasElement}
         */
        const canvas = document.getElementById('display');
        const ctx2 = canvas.getContext('2d');
        ctx2.font = "15px Arial";
        ctx2.fillText("Waiting for input", 20, (canvas.height / 2) - 7);
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

        console.log("Waiting for video selection");
        /**
         * @type {HTMLVideoElement}
         */
        const webcam = document.querySelector('#webcam');
        /**
         * @type {HTMLInputElement}
         */
        const source = document.querySelector('#source');
        /**
         * @param {FileList} FileList 
         */
        var firstPlay = false;
        const handleFiles = function (FileList) {
            ctx1.clearRect(0, 0, osc.width, osc.height);
            if (firstPlay) {
                console.log("Changing media source");
                urlCreator.revokeObjectURL(webcam.src);
                webcam.controls = false;
                webcam.pause();
            } else {
                firstPlay = true;
            }
            if (FileList) {
                const file = FileList.item(0);
                webcam.src = urlCreator.createObjectURL(file);
            } else {
                const file = source.files[0];
                webcam.src = urlCreator.createObjectURL(file);
            }
            webcam.load();
            const play = webcam.play();
            play.catch(onError);
            play.then(() => {
                webcam.controls = true;
                ratio = Math.min(canvas.width / webcam.videoWidth, canvas.height / webcam.videoHeight);
                vid_params = [(canvas.height - (webcam.videoHeight * ratio)) / 2, webcam.videoWidth * ratio, webcam.videoHeight * ratio, (canvas.width - (webcam.videoWidth * ratio)) / 2];
            });
        }
        source.addEventListener("change", () => handleFiles(null), false);
        /**
         * @param {DragEvent} e 
         */
        function dragKill(e) {
            e.stopPropagation();
            e.preventDefault();
        }
        const body = document.getElementsByTagName("body")[0];
        body.addEventListener("dragenter", dragKill, false);
        body.addEventListener("dragover", dragKill, false);
        body.addEventListener("drop", e => {
            dragKill(e);
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files)
        }, false);
        /**
         * @param {Function} resolve 
         */
        const waitUpload = function (resolve) {
            if (!firstPlay) {
                setTimeout(() => waitUpload(resolve), 100);
            }
            else {
                resolve();
            }
        }
        await new Promise(
            /**
             * @param {Function} resolve 
             */
            (resolve) => {
                waitUpload(resolve);
            }
        );
        console.log("Running model");
        
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
            ctx1.drawImage(webcam, 0, 0, webcam.videoWidth, webcam.videoHeight, vid_params[3], vid_params[0], vid_params[1], vid_params[2]);
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