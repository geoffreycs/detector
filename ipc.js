const net = require('net');
let destroying = false;

/**
 * @param {net.Socket} client 
 */
const onError = function (client) {
    destroying = true;
    client.end();    
    client.destroy();
    setTimeout(main, 1000);
    destroying = false;
}

function main() {
    const client = net.createConnection(1337, "127.0.0.1", () => {
        client.on('close', () => {
            if (!destroying) {
                onError(client);
            }
        });
        /**
         * @param {MessageEvent<Number[][]>} msg
         */
        onmessage = msg => {
            client.write(JSON.stringify(msg.data));
        };
        postMessage("connected");
    });
    client.on('error', err => {
        postMessage(err);
        onError(client);
    });
}

main();