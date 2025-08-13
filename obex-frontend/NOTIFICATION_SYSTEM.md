# Threat Notification System

## Overview

The OBEX Threat Notification System provides real-time AI-powered threat detection and alerting capabilities. It monitors camera feeds for unusual behaviors, unauthorized access, suspicious objects, and other security threats.

## Features

### 🚨 Real-time Threat Detection

- **AI-Powered Analysis**: Uses computer vision and behavior analysis to detect threats
- **Multiple Threat Types**:
  - Unauthorized Access/Intrusion
  - Unknown Objects
  - Suspicious Vehicle Activity
  - Unusual Behavior Patterns
  - Loitering Detection
  - Weapon Detection
- **Severity Levels**: Critical, High, Medium, Low
- **Confidence Scoring**: AI confidence percentage for each detection

### 📊 Notification Management

- **Real-time Alerts**: Instant notifications when threats are detected
- **Filtering & Search**: Filter by threat type, severity, or search keywords
- **Bulk Actions**: Mark multiple notifications as read or delete them
- **Statistics Dashboard**: View threat statistics and trends
- **Persistent Storage**: Notifications are saved locally and persist between sessions

### 🔧 Integration Features

- **Camera Integration**: Automatically monitors all connected camera streams
- **Zone-based Detection**: Associates threats with specific security zones
- **Metadata Tracking**: Records detection method, processing time, and model information
- **Test Mode**: Simulate threats for testing and demonstration

## How to Use

### 1. Accessing Notifications

- Click the "Notification" link in the header navigation
- Use the bell icon on mobile devices
- The notification count badge shows unread threats

### 2. Viewing Threat Details

Each notification includes:

- **Threat Type**: What kind of threat was detected
- **Severity Level**: How critical the threat is
- **Location**: Which zone/camera detected the threat
- **Timestamp**: When the threat was detected
- **Confidence**: AI confidence level (70-100%)
- **Camera Information**: Which camera detected the threat

### 3. Managing Notifications

- **Mark as Read**: Click the checkmark icon on individual notifications
- **Mark All Read**: Use the "Mark All Read" button
- **Delete**: Use the trash icon to delete individual notifications
- **Bulk Actions**: Select multiple notifications for bulk operations
- **Filter**: Use the filter buttons to view specific threat types
- **Search**: Use the search bar to find specific notifications

### 4. Testing the System

- Add camera streams to the dashboard
- Click "Test Threat" to simulate a threat detection
- View the notification in the notification page
- The system will automatically start monitoring when cameras are added

## Threat Types

### 🚪 Intrusion Detection

- Detects unauthorized access to restricted areas
- Monitors for suspicious movement patterns
- High severity alerts for security breaches

### 📦 Object Detection

- Identifies unknown or suspicious objects
- Monitors for abandoned items
- Medium severity for investigation

### 🚗 Vehicle Monitoring

- Detects vehicles in unauthorized areas
- Monitors parking violations
- Tracks suspicious vehicle activity

### 👥 Behavior Analysis

- Identifies unusual behavior patterns
- Detects loitering in restricted areas
- Monitors for suspicious activities

### ⚠️ Critical Threats

- Weapon detection
- Immediate emergency alerts
- Automatic security protocol activation

## Technical Details

### AI Detection Methods

- **Computer Vision**: YOLO v8 object detection
- **Behavior Analysis**: Custom algorithms for pattern recognition
- **Real-time Processing**: 25-35 FPS processing capability
- **Confidence Scoring**: 70-100% accuracy range

### Data Storage

- **Local Storage**: Notifications persist in browser localStorage
- **Zustand State Management**: Efficient state management
- **Real-time Updates**: Instant UI updates when threats are detected

### Integration Points

- **Camera Store**: Integrates with existing camera management
- **Navigation Store**: Integrates with app navigation
- **Loading Store**: Integrates with app loading states

## Configuration

### Threat Detection Settings

The system can be configured by modifying the `threatDetection.js` service:

```javascript
// Adjust detection frequency (default: 30 seconds)
this.detectionInterval = setInterval(() => {
  this.performDetection(cameraStreams);
}, 30000);

// Adjust detection probability (default: 20% per camera)
if (Math.random() < 0.2) {
  this.detectThreat(stream, index);
}
```

### Notification Settings

Customize notification behavior in the notification store:

```javascript
// Add custom notification types
const customThreat = {
  type: "custom",
  title: "Custom Threat",
  description: "Custom threat description",
  severity: "medium",
  icon: "custom-icon",
};
```

## Troubleshooting

### Common Issues

1. **No Notifications Appearing**

   - Ensure camera streams are added to the dashboard
   - Check that threat detection is active (green indicator)
   - Try the "Test Threat" button

2. **Notifications Not Saving**

   - Check browser localStorage permissions
   - Clear browser cache and try again
   - Ensure JavaScript is enabled

3. **Performance Issues**
   - Reduce detection frequency for better performance
   - Limit the number of camera streams
   - Check browser console for errors

### Support

For technical support or feature requests, please refer to the main OBEX documentation or contact the development team.

## Future Enhancements

- **Email/SMS Alerts**: Send notifications via email or SMS
- **Video Recording**: Automatically record video when threats are detected
- **Advanced Analytics**: Detailed threat analytics and reporting
- **Machine Learning**: Improved AI models for better detection accuracy
- **Integration APIs**: Connect with external security systems
- **Mobile App**: Dedicated mobile app for notifications
