const net = require('net');

/**
 * @param {net.Socket} client 
 */
const onError = function (client) {
    client.removeAllListeners();
    client.end();
    client.destroy();
    setTimeout(main, 500);
}

function main() {
    const client = net.createConnection(1337, "127.0.0.1", () => {
        // client.on('close', () => {
        //     postMessage("Socket closed \"gracefully\"");
        //     onError(client);
        // });
        /**
         * @param {MessageEvent<Number[][]>} msg
         */
        onmessage = msg => {
            client.write(JSON.stringify(msg.data));
        };
        postMessage("connected");
    });
    client.on('error', err => {
        onmessage = () => { };
        postMessage(err);
        onError(client);
    });
}

main();