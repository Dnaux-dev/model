import { useNotificationStore } from '../store/notification-store';
import { backendIntegrationService } from './backendIntegration';

// Threat detection service for AI-powered security monitoring
export class ThreatDetectionService {
    constructor() {
        this.notificationStore = useNotificationStore.getState();
        this.isActive = false;
        this.detectionInterval = null;
        this.threatTypes = {
            intrusion: {
                title: 'Unauthorized Access Detected',
                description: 'AI detected suspicious movement in restricted zone',
                severity: 'high',
                icon: 'users',
                confidence: 85
            },
            object: {
                title: 'Unknown Object Identified',
                description: 'Unrecognized object detected in surveillance area',
                severity: 'medium',
                icon: 'package',
                confidence: 75
            },
            vehicle: {
                title: 'Suspicious Vehicle Activity',
                description: 'Vehicle detected in unauthorized parking zone',
                severity: 'medium',
                icon: 'car',
                confidence: 80
            },
            behavior: {
                title: 'Unusual Behavior Pattern',
                description: 'AI identified abnormal activity patterns',
                severity: 'high',
                icon: 'alert-circle',
                confidence: 90
            },
            loitering: {
                title: 'Suspicious Loitering Detected',
                description: 'Person detected loitering in restricted area',
                severity: 'medium',
                icon: 'users',
                confidence: 70
            },
            weapon: {
                title: 'Potential Weapon Detected',
                description: 'AI detected potential weapon in surveillance area',
                severity: 'critical',
                icon: 'alert-triangle',
                confidence: 95
            }
        };
    }

    // Start threat detection monitoring (now listens for backend)
    startDetection(cameraStreams = []) {
        if (this.isActive) return;
        
        this.isActive = true;
        console.log('Threat detection service started - connecting to backend');
        
        // Connect to backend service
        backendIntegrationService.connectToBackend();
        
        console.log('Ready to receive threats from backend AI service');
    }

    // Stop threat detection monitoring
    stopDetection() {
        if (!this.isActive) return;
        
        this.isActive = false;
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        // Disconnect from backend
        backendIntegrationService.disconnect();
        console.log('Threat detection service stopped');
    }

    // Handle threats received from backend
    handleBackendThreat(threatData) {
        // Validate threat data from backend
        if (!threatData || !threatData.type || !threatData.title) {
            console.error('Invalid threat data received from backend:', threatData);
            return;
        }

        // Add timestamp if not provided
        if (!threatData.timestamp) {
            threatData.timestamp = new Date().toISOString();
        }

        // Add notification to store
        this.notificationStore.addNotification(threatData);
        
        // Log threat for debugging
        console.log('Backend threat received:', threatData);
        
        // Trigger any additional actions (alerts, recordings, etc.)
        this.handleThreatResponse(threatData);
    }

    // Perform threat detection analysis
    performDetection(cameraStreams) {
        if (!cameraStreams || cameraStreams.length === 0) return;

        // Simulate AI analysis of camera feeds
        cameraStreams.forEach((stream, index) => {
            // Random chance of detecting a threat (for demo purposes)
            if (Math.random() < 0.2) { // 20% chance per camera
                this.detectThreat(stream, index);
            }
        });
    }

    // Detect and log a specific threat
    detectThreat(cameraStream, cameraIndex) {
        const threatTypes = Object.keys(this.threatTypes);
        const randomThreatType = threatTypes[Math.floor(Math.random() * threatTypes.length)];
        const threatConfig = this.threatTypes[randomThreatType];
        
        const zones = ['Zone A', 'Zone B', 'Zone C', 'Main Entrance', 'Parking Lot', 'Loading Dock', 'Security Gate'];
        const randomZone = zones[Math.floor(Math.random() * zones.length)];
        
        const threat = {
            type: randomThreatType,
            title: threatConfig.title,
            description: `${threatConfig.description} in ${randomZone}`,
            severity: threatConfig.severity,
            zone: randomZone,
            cameraId: cameraStream.id || `camera-${cameraIndex}`,
            cameraName: cameraStream.name || `Camera ${cameraIndex + 1}`,
            confidence: Math.floor(Math.random() * 20) + threatConfig.confidence, // Add some variance
            coordinates: {
                x: Math.floor(Math.random() * 100),
                y: Math.floor(Math.random() * 100)
            },
            timestamp: new Date().toISOString(),
            metadata: {
                detectionMethod: 'AI Computer Vision',
                model: 'YOLO v8 + Custom Behavior Analysis',
                processingTime: Math.floor(Math.random() * 500) + 100, // ms
                frameRate: Math.floor(Math.random() * 10) + 25 // fps
            }
        };

        // Add notification to store
        this.notificationStore.addNotification(threat);
        
        // Log threat for debugging
        console.log('Threat detected:', threat);
        
        // Trigger any additional actions (alerts, recordings, etc.)
        this.handleThreatResponse(threat);
    }

    // Handle threat response actions
    handleThreatResponse(threat) {
        // Critical threats should trigger immediate actions
        if (threat.severity === 'critical') {
            this.triggerEmergencyAlert(threat);
        }
        
        // High severity threats should trigger security protocols
        if (threat.severity === 'high') {
            this.triggerSecurityProtocol(threat);
        }
        
        // All threats should be logged and recorded
        this.logThreat(threat);
    }

    // Trigger emergency alert for critical threats
    triggerEmergencyAlert(threat) {
        console.log('🚨 EMERGENCY ALERT:', threat.title);
        // Here you would integrate with your alert system
        // e.g., send SMS, email, trigger sirens, etc.
    }

    // Trigger security protocol for high severity threats
    triggerSecurityProtocol(threat) {
        console.log('⚠️ Security Protocol Triggered:', threat.title);
        // Here you would integrate with your security system
        // e.g., lock doors, notify security personnel, etc.
    }

    // Log threat for audit purposes
    logThreat(threat) {
        // Here you would integrate with your logging system
        // e.g., database, external API, etc.
        console.log('📝 Threat logged:', threat);
    }

    // Manual threat detection (for testing)
    simulateThreat(type = null) {
        const threatTypes = type ? [type] : Object.keys(this.threatTypes);
        const randomThreatType = threatTypes[Math.floor(Math.random() * threatTypes.length)];
        const threatConfig = this.threatTypes[randomThreatType];
        
        const zones = ['Zone A', 'Zone B', 'Zone C', 'Main Entrance', 'Parking Lot'];
        const randomZone = zones[Math.floor(Math.random() * zones.length)];
        
        const threat = {
            type: randomThreatType,
            title: threatConfig.title,
            description: `${threatConfig.description} in ${randomZone}`,
            severity: threatConfig.severity,
            zone: randomZone,
            cameraId: 'test-camera',
            cameraName: 'Test Camera',
            confidence: threatConfig.confidence,
            coordinates: {
                x: Math.floor(Math.random() * 100),
                y: Math.floor(Math.random() * 100)
            },
            timestamp: new Date().toISOString(),
            metadata: {
                detectionMethod: 'Manual Test',
                model: 'Test Simulation',
                processingTime: 150,
                frameRate: 30
            }
        };

        this.notificationStore.addNotification(threat);
        return threat;
    }

    // Simulate receiving a threat from backend (for testing)
    receiveBackendThreat(threatData = null) {
        if (threatData) {
            // Use provided threat data
            this.handleBackendThreat(threatData);
            return threatData;
        } else {
            // Simulate a backend threat with realistic data
            const threatTypes = Object.keys(this.threatTypes);
            const randomThreatType = threatTypes[Math.floor(Math.random() * threatTypes.length)];
            const threatConfig = this.threatTypes[randomThreatType];
            
            const zones = ['Zone A', 'Zone B', 'Zone C', 'Main Entrance', 'Parking Lot'];
            const randomZone = zones[Math.floor(Math.random() * zones.length)];
            
            const simulatedBackendThreat = {
                type: randomThreatType,
                title: threatConfig.title,
                description: `${threatConfig.description} in ${randomZone}`,
                severity: threatConfig.severity,
                zone: randomZone,
                cameraId: `camera-${Math.floor(Math.random() * 10) + 1}`,
                cameraName: `Camera ${Math.floor(Math.random() * 10) + 1}`,
                confidence: Math.floor(Math.random() * 20) + threatConfig.confidence,
                coordinates: {
                    x: Math.floor(Math.random() * 100),
                    y: Math.floor(Math.random() * 100)
                },
                timestamp: new Date().toISOString(),
                metadata: {
                    detectionMethod: 'AI Computer Vision',
                    model: 'YOLO v8 + Custom Behavior Analysis',
                    processingTime: Math.floor(Math.random() * 500) + 100,
                    frameRate: Math.floor(Math.random() * 10) + 25,
                    source: 'Backend AI Service'
                }
            };

            this.handleBackendThreat(simulatedBackendThreat);
            return simulatedBackendThreat;
        }
    }

    // Get threat statistics
    getThreatStats() {
        const notifications = this.notificationStore.notifications;
        const now = new Date();
        const twentyFourHoursAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        
        return {
            total: notifications.length,
            unread: this.notificationStore.unreadCount,
            last24Hours: notifications.filter(n => new Date(n.timestamp) > twentyFourHoursAgo).length,
            bySeverity: {
                critical: notifications.filter(n => n.severity === 'critical').length,
                high: notifications.filter(n => n.severity === 'high').length,
                medium: notifications.filter(n => n.severity === 'medium').length,
                low: notifications.filter(n => n.severity === 'low').length
            },
            byType: {
                intrusion: notifications.filter(n => n.type === 'intrusion').length,
                object: notifications.filter(n => n.type === 'object').length,
                vehicle: notifications.filter(n => n.type === 'vehicle').length,
                behavior: notifications.filter(n => n.type === 'behavior').length,
                loitering: notifications.filter(n => n.type === 'loitering').length,
                weapon: notifications.filter(n => n.type === 'weapon').length
            }
        };
    }
}

// Export singleton instance
export const threatDetectionService = new ThreatDetectionService();

// Hook for React components
export const useThreatDetection = () => {
    return {
        startDetection: threatDetectionService.startDetection.bind(threatDetectionService),
        stopDetection: threatDetectionService.stopDetection.bind(threatDetectionService),
        simulateThreat: threatDetectionService.simulateThreat.bind(threatDetectionService),
        receiveBackendThreat: threatDetectionService.receiveBackendThreat.bind(threatDetectionService),
        getThreatStats: threatDetectionService.getThreatStats.bind(threatDetectionService),
        isActive: threatDetectionService.isActive
    };
};
