const { app, BrowserWindow } = require('electron');

app.disableDomainBlockingFor3DAPIs();
app.whenReady().then(() => {
    const win = new BrowserWindow({
        width: 700,
        height: 400,
        webPreferences: {
            sandbox: false,
            nodeIntegration: true,
            contextIsolation: false,
            nodeIntegrationInWorker: true
        }
    })

    // win.webContents.openDevTools();
    win.loadFile('gpu_worker.html');

});

app.on('window-all-closed', () => app.quit());
app.on('gpu-info-update', () => console.log(app.getGPUFeatureStatus()));