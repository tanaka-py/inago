// .vscode/wait-for-fastapi.js
const http = require('http');

function waitForServer(url, interval = 1000) {
  console.log(`Waiting for FastAPI to be ready at ${url}...`);
  const timer = setInterval(() => {
    http.get(url, (res) => {
      if (res.statusCode === 200) {
        console.log('FastAPI is ready!');
        clearInterval(timer);
        process.exit(0);
      }
    }).on('error', () => {});
  }, interval);
}

waitForServer('http://localhost:8000');
