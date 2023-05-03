const PROXY_CONFIG = {
    "/api/*": {
        "target": process.env.FRONTEND_TARGET || "http://localhost:47316",
        "secure": false
    }
}

console.log('STARTING FRONTEND WITH BACKEND TARGET URL: ' + PROXY_CONFIG[Object.keys(PROXY_CONFIG)[0]].target)

module.exports = PROXY_CONFIG;
