const fs = require('fs');

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
    lines.forEach(
        /**
         * @param {String} element 
         */
        element => {
            labels.push(element.split("  ")[1]);
        });
    return labels;
}

function ArrayChunk() {
    //Stolen from
    // ourcodeworld.com/articles/read/278/how-to-split-an-array-into-chunks-of-the-same-size-easily-in-javascript
    /**
     * @param {Float32Array} arrayIn 
     * @returns {Float32Array[]}
     */
    function chunkArray(arrayIn) {
        var index = 0;
        const arrayLength = arrayIn.length;
        var tempArray = [];
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
const Reformatter = (dw, dh) => {
    /**
     * @param {Float32Array} box_raw
     * @returns {{x: Number, y: Number, w: Number, h: Number}}
     */
    return (box_raw) => {
        return {
            x: dw*box_raw[1],
            y: dh*box_raw[0],
            w: dw*(box_raw[3] - box_raw[1]),
            h: dh*(box_raw[2] - box_raw[0])
        }
    }
}
exports.Reformatter = Reformatter;