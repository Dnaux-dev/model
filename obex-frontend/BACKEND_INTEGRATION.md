# Backend Integration Guide

This guide explains how to integrate the OBEX notification system with your backend AI threat detection service.

## Overview

The notification system has been modified to receive threats from a backend AI service instead of simulating them client-side. The system now includes:

- **Backend Integration Service**: Handles WebSocket connections and API calls
- **Real-time Threat Reception**: Receives threats from backend AI services
- **Testing Functionality**: Maintains testing capabilities for development
- **Connection Management**: Automatic reconnection and error handling

## Architecture

```
Backend AI Service → WebSocket/API → BackendIntegrationService → NotificationStore → UI Components
```

### Components

1. **`src/services/backendIntegration.js`**: Main backend integration service
2. **`src/services/threatDetection.js`**: Updated to use backend integration
3. **`src/pages/Notification.jsx`**: Updated to show backend connection status
4. **`src/Dashboard.jsx`**: Updated with backend testing options

## Backend Requirements

### WebSocket Endpoint

Your backend should provide a WebSocket endpoint that sends threat data in the following format:

```javascript
// WebSocket URL: ws://your-backend-url/threats
// Message format:
{
  "type": "intrusion|object|vehicle|behavior|loitering|weapon",
  "title": "Threat Title",
  "description": "Detailed threat description",
  "severity": "low|medium|high|critical",
  "zone": "Zone Name",
  "cameraId": "camera-1",
  "cameraName": "Camera Name",
  "confidence": 85,
  "coordinates": {
    "x": 45,
    "y": 67
  },
  "timestamp": "2024-01-15T10:30:00.000Z",
  "metadata": {
    "detectionMethod": "AI Computer Vision",
    "model": "YOLO v8 + Custom Behavior Analysis",
    "processingTime": 150,
    "frameRate": 30
  }
}
```

### API Endpoint (Alternative)

If WebSocket is not available, you can use REST API polling:

```javascript
// GET http://your-backend-url/api/threats
// Response: Array of threat objects
[
  {
    type: "intrusion",
    title: "Unauthorized Access Detected",
    // ... other threat properties
  },
];
```

## Configuration

### WebSocket Configuration

Update the WebSocket URL in `src/services/backendIntegration.js`:

```javascript
// Line 15: Update the default backend URL
connectToBackend(backendUrl = 'ws://your-backend-url/threats') {
```

### API Configuration

Update the API URL in `src/services/backendIntegration.js`:

```javascript
// Line 108: Update the default API URL
async pollBackendForThreats(apiUrl = 'http://your-backend-url/api/threats') {
```

### Reconnection Settings

Configure reconnection behavior in `src/services/backendIntegration.js`:

```javascript
// Lines 7-9: Adjust reconnection settings
this.maxReconnectAttempts = 5; // Maximum reconnection attempts
this.reconnectDelay = 5000; // Delay between attempts (ms)
```

## Implementation Steps

### 1. Backend Setup

1. **Create WebSocket Server**:

   ```python
   # Example Python WebSocket server
   import asyncio
   import websockets
   import json

   async def threat_handler(websocket, path):
       while True:
           # Your AI threat detection logic here
           threat = detect_threat()
           if threat:
               await websocket.send(json.dumps(threat))
           await asyncio.sleep(1)

   start_server = websockets.serve(threat_handler, "localhost", 8000)
   asyncio.get_event_loop().run_until_complete(start_server)
   asyncio.get_event_loop().run_forever()
   ```

2. **Create REST API** (alternative):

   ```python
   # Example Flask API
   from flask import Flask, jsonify

   app = Flask(__name__)

   @app.route('/api/threats')
   def get_threats():
       threats = get_pending_threats()
       return jsonify(threats)
   ```

### 2. Frontend Integration

The frontend is already configured to connect to your backend. Just update the URLs:

1. **Update WebSocket URL**:

   ```javascript
   // In src/services/backendIntegration.js
   connectToBackend("ws://your-backend-url/threats");
   ```

2. **Update API URL** (if using polling):
   ```javascript
   // In src/services/backendIntegration.js
   pollBackendForThreats("http://your-backend-url/api/threats");
   ```

### 3. Testing

1. **Start your backend service**
2. **Start the frontend application**
3. **Add camera streams to the dashboard**
4. **Click "Test Backend" to simulate backend threats**
5. **Check the notification page for received threats**

## Testing Features

### Manual Testing

The system includes two testing options:

1. **Test Local**: Simulates local threat detection (original functionality)
2. **Test Backend**: Simulates receiving threats from backend

### Connection Status

The UI shows real-time connection status:

- **Green dot**: Connected to backend
- **Red dot**: Disconnected from backend

## Error Handling

### Automatic Reconnection

The system automatically attempts to reconnect when the connection is lost:

- Maximum 5 reconnection attempts
- 5-second delay between attempts
- Logs reconnection attempts to console

### Error Logging

All connection errors and threat processing errors are logged to the browser console:

- WebSocket connection errors
- Threat data parsing errors
- Invalid threat data warnings

## Security Considerations

### WebSocket Security

1. **Use WSS (WebSocket Secure)** in production:

   ```javascript
   connectToBackend("wss://your-backend-url/threats");
   ```

2. **Implement authentication**:
   ```javascript
   // Add authentication headers or tokens
   const ws = new WebSocket(url, {
     headers: {
       Authorization: "Bearer your-token",
     },
   });
   ```

### API Security

1. **Use HTTPS** in production
2. **Implement API authentication**
3. **Rate limiting** for polling endpoints

## Monitoring and Debugging

### Console Logs

Monitor these console messages:

- `Connected to backend threat service`
- `Backend threat received and processed`
- `Disconnected from backend threat service`
- `Attempting to reconnect (1/5)...`

### Connection Status

Check the UI indicators:

- Dashboard: Shows backend connection status
- Notification page: Shows backend connection status

### Network Tab

Monitor WebSocket connections in browser DevTools:

- Network tab → WS filter
- Check for connection errors
- Monitor message traffic

## Troubleshooting

### Common Issues

1. **Connection Refused**:

   - Check if backend server is running
   - Verify WebSocket URL is correct
   - Check firewall settings

2. **No Threats Received**:

   - Verify backend is sending threats
   - Check threat data format
   - Monitor console for errors

3. **Reconnection Loops**:
   - Check backend server stability
   - Verify network connectivity
   - Check WebSocket URL

### Debug Steps

1. **Check Console Logs**:

   ```javascript
   // Add debug logging
   console.log("WebSocket state:", this.ws.readyState);
   console.log("Connection status:", this.isConnected);
   ```

2. **Test WebSocket Connection**:

   ```javascript
   // Test in browser console
   const ws = new WebSocket("ws://your-backend-url/threats");
   ws.onopen = () => console.log("Connected");
   ws.onmessage = (e) => console.log("Message:", e.data);
   ```

3. **Verify Threat Format**:
   ```javascript
   // Test threat data format
   const testThreat = {
     type: "intrusion",
     title: "Test Threat",
     description: "Test description",
     severity: "high",
   };
   ```

## Performance Considerations

### WebSocket vs API Polling

- **WebSocket**: Real-time, lower latency, persistent connection
- **API Polling**: Simpler, works with any HTTP server, higher latency

### Optimization Tips

1. **Batch Threats**: Send multiple threats in one message
2. **Compression**: Use WebSocket compression if available
3. **Rate Limiting**: Limit threat frequency to prevent UI overload

## Future Enhancements

### Planned Features

1. **Threat Acknowledgment**: Send acknowledgments back to backend
2. **Threat History**: Store and retrieve historical threats
3. **Advanced Filtering**: Filter threats by camera, zone, severity
4. **Real-time Analytics**: Live threat statistics and trends

### Customization

1. **Custom Threat Types**: Add new threat categories
2. **Custom Severity Levels**: Define custom severity scales
3. **Custom Actions**: Trigger custom actions for specific threats

## Support

For technical support or questions about backend integration:

1. Check the console logs for error messages
2. Verify your backend implementation matches the expected format
3. Test with the provided testing functionality
4. Review the troubleshooting section above

## Example Backend Implementations

### Python (FastAPI + WebSockets)

```python
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import json
import asyncio

app = FastAPI()

@app.websocket("/threats")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Your AI threat detection logic here
            threat = await detect_threat()
            if threat:
                await websocket.send_text(json.dumps(threat))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Client disconnected")
```

### Node.js (Express + Socket.io)

```javascript
const express = require("express");
const { createServer } = require("http");
const { Server } = require("socket.io");

const app = express();
const server = createServer(app);
const io = new Server(server);

io.on("connection", (socket) => {
  console.log("Client connected");

  // Your AI threat detection logic here
  setInterval(() => {
    const threat = detectThreat();
    if (threat) {
      socket.emit("threat", threat);
    }
  }, 1000);

  socket.on("disconnect", () => {
    console.log("Client disconnected");
  });
});

server.listen(8000);
```

This integration guide provides everything you need to connect your backend AI threat detection service to the OBEX notification system.
