// Fetch theft alerts from backend
export async function fetchTheftAlerts(limit = 50) {
    const res = await fetch(`http://localhost:8000/mongo/alerts?alert_type=theft&limit=${limit}`);
    if (res.ok) {
        const data = await res.json();
        return data || [];
    }
    return [];
}

// setZone for loitering and intrusion
export async function setZone(x1, y1, x2, y2) {
  const res = await fetch('http://localhost:8000/set_zone', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x1, y1, x2, y2 }),
  });
  return await res.json();
}

// Fetch loitering snapshots from backend
export async function fetchLoiteringSnapshots(limit = 50) {
    const res = await fetch(`http://localhost:8000/mongo/snapshots?limit=${limit}`);
    if (res.ok) {
        const data = await res.json();
        return data.snapshots || [];
    }
    return [];
}
import { useNotificationStore } from '../store/notification-store';

class BackendIntegrationService {
    constructor() {
        this.notificationStore = useNotificationStore.getState();
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 5000; // 5 seconds
    }

    // Connect to backend WebSocket
    connectToBackend(backendUrl = 'ws://localhost:8000/threats') {
        try {
            this.ws = new WebSocket(backendUrl);
            
            this.ws.onopen = () => {
                console.log('Connected to backend threat service');
                this.isConnected = true;
                this.reconnectAttempts = 0;
            };

            this.ws.onmessage = (event) => {
                try {
                    const threatData = JSON.parse(event.data);
                    this.handleBackendThreat(threatData);
                } catch (error) {
                    console.error('Error parsing threat data from backend:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('Disconnected from backend threat service');
                this.isConnected = false;
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
            };

        } catch (error) {
            console.error('Failed to connect to backend:', error);
        }
    }

    // Attempt to reconnect to backend
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.connectToBackend();
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
        }
    }

    // Disconnect from backend
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.isConnected = false;
            console.log('Disconnected from backend threat service');
        }
    }

    // Handle threat data received from backend
    handleBackendThreat(threatData) {
        // Validate threat data
        if (!threatData || !threatData.type || !threatData.title) {
            console.error('Invalid threat data received from backend:', threatData);
            return;
        }

        // Add timestamp if not provided
        if (!threatData.timestamp) {
            threatData.timestamp = new Date().toISOString();
        }

        // Add source information
        threatData.metadata = {
            ...threatData.metadata,
            source: 'Backend AI Service',
            receivedAt: new Date().toISOString()
        };

        // Add notification to store
        this.notificationStore.addNotification(threatData);
        
        console.log('Backend threat received and processed:', threatData);
    }

    // Send acknowledgment to backend
    sendAcknowledgment(threatId) {
        if (this.ws && this.isConnected) {
            const ack = {
                type: 'acknowledgment',
                threatId: threatId,
                timestamp: new Date().toISOString()
            };
            this.ws.send(JSON.stringify(ack));
        }
    }

    // Poll backend for threats (alternative to WebSocket)
    async pollBackendForThreats(apiUrl = 'http://localhost:8000/api/threats') {
        try {
            const response = await fetch(apiUrl);
            if (response.ok) {
                const threats = await response.json();
                threats.forEach(threat => {
                    this.handleBackendThreat(threat);
                });
            }
        } catch (error) {
            console.error('Error polling backend for threats:', error);
        }
    }

    // Get connection status
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            maxReconnectAttempts: this.maxReconnectAttempts
        };
    }
}

// Create singleton instance
export const backendIntegrationService = new BackendIntegrationService();

// Hook for React components
export const useBackendIntegration = () => {
    return {
        connectToBackend: backendIntegrationService.connectToBackend.bind(backendIntegrationService),
        disconnect: backendIntegrationService.disconnect.bind(backendIntegrationService),
        sendAcknowledgment: backendIntegrationService.sendAcknowledgment.bind(backendIntegrationService),
        pollBackendForThreats: backendIntegrationService.pollBackendForThreats.bind(backendIntegrationService),
        getConnectionStatus: backendIntegrationService.getConnectionStatus.bind(backendIntegrationService),
        isConnected: backendIntegrationService.isConnected
    };
};
