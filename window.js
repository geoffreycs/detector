const fs = require('fs');
const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
const { asmExport, server, port, getGL, onError } = require('./shared');
const { converted, x_accum, y_accum, w_accum, h_accum, m1_accum, m2_accum, avgs, dimsAvg,
    reformat, setAll, accConf, avgConf } = asmExport;
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
const blank = (new OffscreenCanvas(2000, 2000)).getContext('2d');
blank.clearRect(0, 0, 2000, 2000);

const cnvGL = document.createElement('canvas');
cnvGL.hidden = true;
cnvGL.height = 300;
cnvGL.width = 300;
const GLctx = getGL(cnvGL);
const drawGL = GLctx.module;

let ratio = 0.0;
let vid_params = [0.0, 0.0, 0.0, 0.0];
var pause = true;
var firstPlay = false;

const header = "VideoTime,RawConf,AvgConf,Size,Status,Rej\n";
let log = header;

const rolling = new Float64Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

async function init() {
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
    const model = await tflite.loadTFLiteModel(fs.readFileSync("drone/drone-detect1.tflite"));
    console.log("Closing HTTP server")
    server.close();
    server.removeAllListeners();

    /**
     * @type {HTMLButtonElement}
     */
    const save = document.getElementById("save");
    const download = document.createElement("a");
    download.download = "log.csv";
    save.onclick = function () {
        if (download.href) {
            webkitURL.revokeObjectURL(download.href);
        }
        download.href = URL.createObjectURL(new Blob([log], { type: "text/plain" }));
        download.click();
    }

    console.log("Waiting for video selection");
    /**
     * @type {HTMLVideoElement}
     */
    const webcam = document.querySelector('#webcam');
    webcam.addEventListener("play", () => {
        pause = false;
        const onFrame = function () {
            if (!pause) {
                drawGL(webcam);
                webcam.requestVideoFrameCallback(onFrame);
            }
        }
        onFrame();
    });
    webcam.addEventListener("pause", () => { pause = true });
    webcam.onseeked = function () {
        if (pause) {
            drawGL(webcam);
            pause = false;
        }
    };
    webcam.onended = () => { pause = true };
    /**
     * @type {HTMLInputElement}
     */
    const source = document.querySelector('#source');
    /**
     * @param {FileList} FileList 
     */
    const handleFiles = function (FileList) {
        const newFile = FileList ? FileList.item(0) : source.files[0];
        if (newFile.type.includes("video/")) {
            if (firstPlay) {
                console.log("Changing media source");
                window.URL.revokeObjectURL(webcam.src);
                webcam.controls = false;
                webcam.pause();
                log = header;
            } else {
                firstPlay = true;
            }
            webcam.src = window.webkitURL.createObjectURL(newFile);
            webcam.load();
            ctx1.clearRect(0, 0, osc.width, osc.height);
            ctx2.clearRect(0, 0, canvas.width, canvas.height);
            const clearing = function () {

            }
            const play = webcam.play();
            play.catch(onError);
            play.then(function () {
                drawGL(blank.canvas);
                cnvGL.width = webcam.videoWidth;
                cnvGL.height = webcam.videoHeight;
                GLctx.resize();
                GLctx.clear();
                webcam.controls = true;
                ratio = Math.min(canvas.width / webcam.videoWidth, canvas.height / webcam.videoHeight);
                vid_params = [(canvas.height - (webcam.videoHeight * ratio)) / 2, webcam.videoWidth * ratio, webcam.videoHeight * ratio, (canvas.width - (webcam.videoWidth * ratio)) / 2];
                blank.canvas.height = webcam.videoHeight;
                blank.canvas.width = webcam.videoWidth;
            });
        } else {
            console.log("Ignoring dropped file of type", newFile.type);
        }
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

    /**
     * @type {Number}
     */
    const timings = {
        "cpu": 50,
        "webgl": 50,
        "webgpu": 30,
        "wasm": 50
    }[tf.getBackend()];
    const discardOldThres = timings;
    const trackLostThres = timings / 2;
    const cnv_w = canvas.width;
    const cnv_h = canvas.height;

    /**
     * @type {HTMLParagraphElement}
     */
    const perf = document.querySelector("#perf");

    var idx_t = 0 | 0;
    var idx_d = 0 | 0;
    var last = 4 | 0;
    let lostCount = 0 | 0;
    let trackExpired = true;
    let trackStale = true;
    let lastX = 0.0;
    let lastY = 0.0;
    var status = 0 | 0;
    const regainMax = 20.0; // max inter-frame jump when track expired
    const trackMax = 50.0; // max inter-frame jump in active or stale track
    const maxsize = 20000.0; // max size on screen
    const minsize = 20.0; // min size on screen
    const close = 1000.0 // size on screen before confidence threshold is raised
    const bigConf = .5 // min confidence for up-close detections
    const maxsquat = 2.9; // max W/H
    const closesquat = .7; // min W/H when close
    /**
     * @param {Number} dX 
     * @param {Number} dY 
     * @param {Number} confInst
     * @returns {Boolean}
     */
    const checkHeuristics = function (dX, dY, confInst) {
        return ((dX > trackMax || dY > trackMax || converted[9] > maxsquat) && !trackExpired)
            || converted[8] > maxsize || converted[8] < minsize || (converted[8] > close && (confInst < bigConf || closesquat > converted[9]));
    }

    const doInference = async function () {
        if (!pause) {
            let i = 0;
            const start = performance.now();
            ctx1.drawImage(cnvGL, 0, 0, webcam.videoWidth, webcam.videoHeight, vid_params[3], vid_params[0], vid_params[1], vid_params[2]);
            const bitmap = await createImageBitmap(osc);
            const img = tf.browser.fromPixels(bitmap);
            const input = tf.expandDims(img, 0);

            /**
             * @type {{TFLite_Detection_PostProcess: tf.Tensor,
             * "TFLite_Detection_PostProcess:1": tf.Tensor,
             * "TFLite_Detection_PostProcess:2": tf.Tensor,
             * "TFLite_Detection_PostProcess:3": tf.Tensor }}
             */
            const output = model.predict(input);

            /**
             * @type {Float32Array}
             */
            const pointsOut = await output.TFLite_Detection_PostProcess.data();
            /**
             * @type {Float32Array}
             */
            const confOut = output['TFLite_Detection_PostProcess:2'].dataSync();
            const numDetect = output['TFLite_Detection_PostProcess:3'].bufferSync().values[0];

            img.dispose();
            input.dispose();
            output.TFLite_Detection_PostProcess.dispose();
            output['TFLite_Detection_PostProcess:1'].dispose();
            output['TFLite_Detection_PostProcess:2'].dispose();
            output['TFLite_Detection_PostProcess:3'].dispose();

            trackExpired = (lostCount > discardOldThres);

            reformat(new Float32Array(pointsOut.buffer, pointsOut.byteOffset, 4), lastX, lastY);
            let dX = converted[6];
            let dY = converted[7];
            i = 0;
            while (checkHeuristics(dX, dY, confOut[i]) && (i + 1) < numDetect) {
                reformat(new Float32Array(pointsOut.buffer, pointsOut.byteOffset + (i + 1) * 16, 4), lastX, lastY);
                dX = converted[6];
                dY = converted[7];
                i++;
            }
            if (checkHeuristics(dX, dY, confOut[i])) {
                i = 0;
                confOut[i] = 0.0;
            }
            lastX = converted[4];
            lastY = converted[5];
            if (confOut[i] > .4) {
                if (trackExpired && (dX > regainMax || dY > regainMax)) {
                    setAll();
                } else {
                    x_accum[idx_d] = converted[0];
                    y_accum[idx_d] = converted[1];
                    w_accum[idx_d] = converted[2];
                    h_accum[idx_d] = converted[3];
                    m1_accum[idx_d] = lastX;
                    m2_accum[idx_d] = lastY;
                }
                lostCount = 0;
                accConf(confOut[i]);
            } else {
                x_accum[idx_d] = x_accum[last];
                y_accum[idx_d] = y_accum[last];
                w_accum[idx_d] = w_accum[last];
                h_accum[idx_d] = h_accum[last];
                m1_accum[idx_d] = m1_accum[last];
                m2_accum[idx_d] = m2_accum[last];
                lostCount++;
            }
            last = idx_d;
            idx_d = (idx_d + 1) % 5;

            trackStale = (lostCount > trackLostThres);

            if (lostCount === 0) {
                ctx2.strokeStyle = 'green';
                status = 0;
            } else if (!trackStale) {
                ctx2.strokeStyle = 'blue';
                status = 1;
            } else if (!trackExpired) {
                accConf(confOut[i]);
                ctx2.strokeStyle = 'purple';
                status = 2;
            } else {
                ctx2.strokeStyle = 'red';
                status = 3;
            }

            dimsAvg();
            ctx2.clearRect(0, 0, cnv_w, cnv_h);
            ctx2.drawImage(bitmap, 0, 0);
            ctx2.beginPath();
            ctx2.rect(avgs[0], avgs[1], avgs[2], avgs[3]);
            ctx2.stroke();

            const smoothConf = avgConf();

            desc.innerText = confOut[i].toFixed(7) + ", " + smoothConf.toFixed(7) + ", "
                + String(lostCount).padStart(3, '0');

            const msec = performance.now() - start;
            rolling[idx_t] = msec;
            idx_t = (idx_t + 1) % 10;
            const total = rolling[0] + rolling[1] + rolling[2] + rolling[3] + rolling[4] +
                rolling[5] + rolling[6] + rolling[7] + rolling[8] + rolling[9];
            perf.innerText = msec.toFixed(2).padStart(6, '0') + "ms, " +
                (total / 10).toFixed(2).padStart(6, '0') + "ms, rej " + i.toString() + ", " +
                String(webcam.currentTime);
            log = log + String(webcam.currentTime) + "," + confOut[i].toFixed(7) +
                "," + smoothConf.toFixed(7) + "," + converted[8].toFixed(10) + "," +
                status.toString() + "," + new String(i) + "\n";
        }
        setTimeout(() => doInference().catch(onError), 5);
        pause = webcam.paused;
    }

    const runner = doInference();
    runner.catch(onError);

};

const main = () => { init().catch(onError); }