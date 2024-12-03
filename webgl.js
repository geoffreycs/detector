const { app, BrowserWindow, ipcMain } = require('electron');

app.disableDomainBlockingFor3DAPIs();
app.whenReady().then(() => {
    const win = new BrowserWindow({
        width: 700,
        height: 470,
        webPreferences: {
            sandbox: false,
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    ipcMain.on('error', () => { win.webContents.openDevTools({ mode: 'detach' }); });
    win.loadFile('gl_worker.html');
    //win.webContents.openDevTools();

});

app.on('window-all-closed', () => app.quit());
app.on('gpu-info-update', () => console.log(app.getGPUFeatureStatus()));