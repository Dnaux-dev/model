import React, { useState, useEffect, useRef } from "react";
import { useCameraStore } from './store/camera-store';
import { useEventStore } from "./store/history-store";
import { useNotificationStore } from "./store/notification-store";
import { Camera, AlertTriangle, MapPin, Clock, Calendar, Trash2, Maximize2, Minimize2 } from 'lucide-react';

export default function CameraCard({ cameraName, date, time, threatLevel, id, url, zoneCategory, ipAddress }) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [latestAlert, setLatestAlert] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [zoneStart, setZoneStart] = useState(null);
  const [currentZone, setCurrentZone] = useState(null);
  const [performance, setPerformance] = useState(null);
  const videoContainerRef = useRef(null);
  const { addNotification } = useNotificationStore();

  const threatColors = {
    Low: "bg-gradient-to-r from-emerald-500 to-green-500",
    Medium: "bg-gradient-to-r from-amber-500 to-yellow-500",
    High: "bg-gradient-to-r from-red-600 to-pink-600"
  };

  const threatShadows = {
    Low: "shadow-emerald-500/30",
    Medium: "shadow-amber-500/30",
    High: "shadow-red-500/40"
  };

  const threatIcons = {
    Low: "🟢",
    Medium: "🟡",
    High: "🔴"
  };

  // Poll for alerts and load zones if this is the backend feed
  useEffect(() => {
    if (id === 'backend-feed') {
      const fetchData = async () => {
        try {
          const [theftRes, suspiciousRes, intrusionRes, loiteringRes, zonesRes, performanceRes] = await Promise.all([
            fetch(`http://localhost:8000/alerts/${id}`),
            fetch('http://localhost:8000/suspicious_behavior'),
            fetch('http://localhost:8000/intrusion_alerts'),
            fetch('http://localhost:8000/loitering_alerts'),
            fetch('http://localhost:8000/zones'),
            fetch(`http://localhost:8000/performance/${id}`)
          ]);

          // Check response statuses
          console.log('Response Statuses:');
          console.log('Theft:', theftRes.status, theftRes.ok);
          console.log('Suspicious:', suspiciousRes.status, suspiciousRes.ok);
          console.log('Intrusion:', intrusionRes.status, intrusionRes.ok);
          console.log('Loitering:', loiteringRes.status, loiteringRes.ok);
          console.log('Zones:', zonesRes.status, zonesRes.ok);
          console.log('Performance:', performanceRes.status, performanceRes.ok);

          const theftData = await theftRes.json();
          const suspiciousData = await suspiciousRes.json();
          const intrusionData = await intrusionRes.json();
          const loiteringData = await loiteringRes.json();
          const zonesData = await zonesRes.json();
          const performanceData = await performanceRes.json();

          // Debug logging
          console.log('Backend Alert Data:');
          console.log('Theft Data:', theftData);
          console.log('Suspicious Data:', suspiciousData);
          console.log('Intrusion Data:', intrusionData);
          console.log('Loitering Data:', loiteringData);
          console.log('Performance Data:', performanceData);

          // Load existing zone if available
          if (zonesData.zones && zonesData.zones.length > 0 && !currentZone) {
            const zone = zonesData.zones[0];
            setCurrentZone({
              x1: zone[0],
              y1: zone[1], 
              x2: zone[2],
              y2: zone[3]
            });
          }

          let allAlerts = [];
          
          // Handle theft alerts
          if (theftData.theft_alerts && theftData.theft_alerts.length > 0) {
            allAlerts = allAlerts.concat(
              theftData.theft_alerts.map(a => ({
                type: 'theft',
                id: a.type || 'unknown',
                time: new Date(a.timestamp * 1000).toLocaleTimeString(),
                details: a.details || `${a.type} detected`,
                severity: a.severity || 'HIGH'
              }))
            );
          }
          
          // Handle suspicious behavior alerts
          if (suspiciousData.suspicious_behavior && suspiciousData.suspicious_behavior.length > 0) {
            allAlerts = allAlerts.concat(
              suspiciousData.suspicious_behavior.map(a => ({
                type: 'suspicious',
                id: a.type || 'behavior',
                time: new Date(a.timestamp * 1000).toLocaleTimeString(),
                details: a.details || `${a.type} behavior detected`,
                severity: a.severity || 'MEDIUM'
              }))
            );
          }

          // Handle intrusion alerts
          if (intrusionData.intrusion_alerts && intrusionData.intrusion_alerts.length > 0) {
            allAlerts = allAlerts.concat(
              intrusionData.intrusion_alerts.map(a => ({
                type: 'intrusion',
                id: a.track_id,
                time: new Date(a.entry_time * 1000).toLocaleTimeString(),
                details: `Person ${a.track_id} entered restricted zone`,
                severity: 'HIGH'
              }))
            );
          }

          // Handle loitering alerts
          if (loiteringData.loitering_alerts && loiteringData.loitering_alerts.length > 0) {
            allAlerts = allAlerts.concat(
              loiteringData.loitering_alerts.map(a => ({
                type: 'loitering',
                id: a.track_id,
                time: new Date(a.entry_time * 1000).toLocaleTimeString(),
                details: `Person ${a.track_id} loitering for ${a.duration?.toFixed(1) || 'unknown'}s`,
                severity: 'MEDIUM'
              }))
            );
          }

          // Sort alerts by most recent first
          allAlerts.sort((a, b) => new Date(b.time) - new Date(a.time));

          setAlerts(allAlerts);
          setLatestAlert(allAlerts[0] || null);
          setPerformance(performanceData);

          // Add new alerts to notification store
          allAlerts.forEach(alert => {
            addNotification({
              type: alert.type,
              title: `${alert.type.toUpperCase()} DETECTED`,
              message: alert.details,
              severity: alert.severity,
              cameraName: cameraName,
              timestamp: new Date().toISOString()
            });
          });
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };

      fetchData();
      const interval = setInterval(fetchData, 3000); // Poll every 3 seconds
      return () => clearInterval(interval);
    }
  }, [id, cameraName, addNotification, currentZone]);

  // Add the missing handleView function
  const handleView = (e) => {
    e.stopPropagation();
    const timestamp = new Date().toISOString();
    useEventStore.getState().addEvent({
      id,
      cameraName,
      date,
      time,
      ipAddress: url || ipAddress,
      threatLevel,
      zoneCategory: zoneCategory || "Unknown",
      type: 'VIEWED',
      timestamp,
    });
  };

  const toggleFullscreen = (e) => {
    e.stopPropagation();
    
    if (!isFullscreen) {
      if (videoContainerRef.current.requestFullscreen) {
        videoContainerRef.current.requestFullscreen();
      } else if (videoContainerRef.current.webkitRequestFullscreen) {
        videoContainerRef.current.webkitRequestFullscreen();
      } else if (videoContainerRef.current.msRequestFullscreen) {
        videoContainerRef.current.msRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
      }
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('msfullscreenchange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('msfullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Add handleConfirmation function if not present
  const handleConfirmation = (e) => {
    e.stopPropagation();
    const isConfirmed = window.confirm("Are you sure you want to delete this camera stream?");
    if (isConfirmed) {
      const timestamp = new Date().toISOString();
      useEventStore.getState().addEvent({
        id,
        cameraName,
        date,
        time,
        ipAddress: url || ipAddress,
        threatLevel,
        zoneCategory: zoneCategory || "Unknown",
        type: 'DELETED',
        timestamp,
      });
      useCameraStore.getState().removeFromCameraStreams(id);
    }
  };

  const getAlertColor = (alertType) => {
    switch (alertType) {
      case 'theft': return 'bg-red-600';
      case 'suspicious': return 'bg-orange-600';
      case 'intrusion': return 'bg-red-500';
      case 'loitering': return 'bg-yellow-500';
      default: return 'bg-yellow-600';
    }
  };

  const getAlertIcon = (alertType) => {
    switch (alertType) {
      case 'theft': return '🚨';
      case 'suspicious': return '⚠️';
      case 'intrusion': return '🚪';
      case 'loitering': return '⏱️';
      default: return '🔔';
    }
  };

  // Zone drawing functions
  const handleMouseDown = (e) => {
    if (!isDrawingZone) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setZoneStart({ x, y });
  };

  const handleMouseMove = (e) => {
    if (!isDrawingZone || !zoneStart) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setCurrentZone({
      x1: Math.min(zoneStart.x, x),
      y1: Math.min(zoneStart.y, y),
      x2: Math.max(zoneStart.x, x),
      y2: Math.max(zoneStart.y, y)
    });
  };

  const handleMouseUp = async () => {
    if (!isDrawingZone || !currentZone) return;
    
    try {
      // Send zone coordinates to backend
      const response = await fetch('http://localhost:8000/set_zone', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentZone),
      });
      
      if (response.ok) {
        console.log('Zone set successfully');
        setIsDrawingZone(false);
        setZoneStart(null);
        // Keep currentZone to show the drawn zone
      }
    } catch (error) {
      console.error('Error setting zone:', error);
    }
  };

  const clearZone = () => {
    setCurrentZone(null);
    setZoneStart(null);
    setIsDrawingZone(false);
  };

  // Test function to manually fetch alerts
  const testFetchAlerts = async () => {
    try {
      console.log('Testing alert endpoints...');
      
      const endpoints = [
        'http://localhost:8000/alerts/scene2',
        'http://localhost:8000/suspicious_behavior',
        'http://localhost:8000/intrusion_alerts',
        'http://localhost:8000/loitering_alerts',
        'http://localhost:8000/alerts'
      ];
      
      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint);
          console.log(`${endpoint}: Status ${response.status}`);
          if (response.ok) {
            const data = await response.json();
            console.log(`${endpoint}: Data`, data);
          } else {
            console.log(`${endpoint}: Error - ${response.statusText}`);
          }
        } catch (error) {
          console.log(`${endpoint}: Fetch error -`, error);
        }
      }
    } catch (error) {
      console.error('Test fetch error:', error);
    }
  };

  // Prefer provided url, else default to backend MJPEG endpoint
  const streamUrl = url || 'http://localhost:8000/video_feed';

  return (
    <section
      className="relative bg-gradient-to-br from-slate-800 via-slate-700 to-slate-800 rounded-3xl shadow-2xl shadow-slate-900/50 overflow-hidden w-full h-auto min-h-[120px] cursor-pointer hover:scale-105 transition-all duration-500 border border-slate-600/30 hover:border-cyan-400/50 group backdrop-blur-xl flex flex-col hover:shadow-cyan-400/20 hover:shadow-3xl"
      onClick={handleView}
    >
      {/* Video Container */}
      <div
        ref={videoContainerRef}
        className={`relative bg-gradient-to-br from-slate-900 via-black to-slate-900 
          h-64 sm:h-72 lg:h-96 
          w-full flex items-center justify-center text-white overflow-hidden rounded-t-3xl flex-shrink-0 
          group-hover:shadow-inner group-hover:shadow-cyan-400/20 transition-all duration-500
          ${isFullscreen ? 'fixed inset-0 z-50 !rounded-none !m-0 !h-screen !w-screen' : ''}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{ cursor: isDrawingZone ? 'crosshair' : 'pointer' }}
      >
        {/* Show backend MJPEG feed in an img tag */}
        <img
          src={streamUrl}
          alt="Live Feed"
          className="w-full h-full object-cover"
        />
        
        {/* Zone Overlay */}
        {currentZone && (
          <div
            className="absolute border-2 border-red-500 bg-red-500/20"
            style={{
              left: currentZone.x1,
              top: currentZone.y1,
              width: currentZone.x2 - currentZone.x1,
              height: currentZone.y2 - currentZone.y1,
            }}
          >
            <div className="absolute -top-6 left-0 text-xs bg-red-500 text-white px-2 py-1 rounded">
              RESTRICTED ZONE
            </div>
          </div>
        )}

        {/* Heatmap Overlay */}
        {showHeatmap && (
          <img
            src="http://localhost:8000/heatmap"
            alt="Heatmap"
            className="absolute inset-0 w-full h-full object-cover opacity-50 pointer-events-none"
          />
        )}

        {/* Only show threat overlay when there are actual alerts */}
        {!isFullscreen && alerts.length > 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center opacity-60 group-hover:opacity-80 transition-opacity duration-300">
            <AlertTriangle className="w-20 h-20 text-red-600/100 group-hover:scale-110 transition-transform duration-300 animate-ping" />
            <p className="text-white-600 text-sm mt-3 text-center px-4 font-bold animate-ping bg-red-600 rounded-full px-2 py-1.5 ">Threat Detected</p>
            <p className="text-red-300 text-xs mt-1 text-center px-4 opacity-75">Click to view details</p>
          </div>
        )}

        {/* Latest Alert Overlay */}
        {!isFullscreen && latestAlert && (
          <div className={`absolute top-2 left-2 right-2 ${getAlertColor(latestAlert.type)} text-white p-2 rounded-lg text-xs`}>
            <div className="flex items-center gap-2">
              <span>{getAlertIcon(latestAlert.type)}</span>
              <span className="font-bold">
                {latestAlert.type === 'theft' ? 'THEFT DETECTED!' :
                 latestAlert.type === 'suspicious' ? 'SUSPICIOUS BEHAVIOR!' :
                 latestAlert.type === 'intrusion' ? 'INTRUSION ALERT!' :
                 latestAlert.type === 'loitering' ? 'LOITERING DETECTED!' :
                 'ALERT!'}
              </span>
            </div>
            <div className="text-xs mt-1">
              {latestAlert.details}
            </div>
          </div>
        )}

        {/* Zone Management Controls */}
        <div className="absolute top-2 right-2 flex gap-2 z-20">
          <button
            onClick={(e) => {
              e.stopPropagation();
              testFetchAlerts();
            }}
            className="bg-purple-600 text-white px-2 py-1 rounded text-xs hover:scale-105 transition-transform duration-300"
          >
            Test Alerts
          </button>
          
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsDrawingZone(!isDrawingZone);
            }}
            className={`px-2 py-1 rounded text-xs ${isDrawingZone ? 'bg-red-600' : 'bg-blue-600'} text-white hover:scale-105 transition-transform duration-300`}
          >
            {isDrawingZone ? 'Cancel Zone' : 'Draw Zone'}
          </button>
          
          {currentZone && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearZone();
              }}
              className="bg-gray-600 text-white px-2 py-1 rounded text-xs hover:scale-105 transition-transform duration-300"
            >
              Clear Zone
            </button>
          )}
          
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowHeatmap(!showHeatmap);
            }}
            className="bg-green-600 text-white px-2 py-1 rounded text-xs hover:scale-105 transition-transform duration-300"
          >
            {showHeatmap ? 'Hide' : 'Show'} Heatmap
          </button>
        </div>

        {/* Performance Indicator */}
        {performance && (
          <div className="absolute bottom-2 left-2 bg-black/70 text-white p-2 rounded text-xs z-10">
            <div>FPS: {performance.fps}</div>
            <div>Faces: {performance.faces_detected}</div>
            <div>Objects: {performance.objects_detected}</div>
            {performance.zone_active && <div>Zone: Active</div>}
          </div>
        )}

        <button
          onClick={toggleFullscreen}
          className="absolute bottom-3 right-3 text-white bg-slate-800/80 hover:bg-slate-700/90 rounded-lg p-2 cursor-pointer transition-all duration-300 hover:scale-110 hover:shadow-lg hover:shadow-cyan-500/30 z-10"
        >
          {isFullscreen ? (
            <Minimize2 size={18} className="text-cyan-400" />
          ) : (
            <Maximize2 size={18} className="text-cyan-400" />
          )}
        </button>
      </div>

      {/* Card Info Section */}
      {!isFullscreen && (
        <div className="relative p-4 flex flex-col justify-between flex-1 bg-gradient-to-r from-slate-700/60 to-slate-800/60 backdrop-blur-sm border-t border-slate-600/20 group-hover:bg-gradient-to-r group-hover:from-slate-700/80 group-hover:to-slate-800/80 transition-all duration-300 min-h-[110px] lg:h-[40px]">
          <div className="space-y-3 mb-4">
            <div className="flex items-center gap-2 text-slate-300 text-xs font-medium opacity-90 group-hover:opacity-100 transition-opacity duration-300">
              <Calendar className="w-3 h-3 text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300" />
              <span className="truncate">{date}</span>
            </div>
            <div className="flex items-center gap-2 text-slate-300 text-xs font-medium opacity-90 group-hover:opacity-100 transition-opacity duration-300">
              <Clock className="w-3 h-3 text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300" />
              <span className="truncate">{time}</span>
            </div>
            <h3 className="text-white text-base font-bold uppercase tracking-wide group-hover:text-cyan-100 transition-colors duration-300 flex items-center gap-2 truncate">
              <Camera className="w-4 h-4 text-cyan-400 flex-shrink-0 group-hover:scale-110 transition-transform duration-300" />
              <span className="truncate">{cameraName}</span>
            </h3>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="relative group/badge">
                <span className={`text-white text-xs font-bold px-2 py-1.5 rounded-full ${threatColors[threatLevel]} text-center shadow-lg ${threatShadows[threatLevel]} backdrop-blur-sm flex items-center gap-1.5 group-hover/badge:scale-105 transition-transform duration-300`}>
                  <span className="text-xs">{threatIcons[threatLevel]}</span>
                  <span className="truncate">{threatLevel}</span>
                </span>
                <div className={`absolute inset-0 rounded-full blur-sm opacity-30 ${threatColors[threatLevel]} group-hover/badge:opacity-50 transition-opacity duration-300`}></div>
              </div>

              <div className="bg-gradient-to-r from-slate-700/90 to-slate-600/90 backdrop-blur-sm text-cyan-300 text-xs font-semibold px-2 py-1.5 rounded-full border border-cyan-400/30 shadow-lg flex items-center gap-1.5 group-hover:scale-105 transition-transform duration-300 group-hover:border-cyan-400/50 group-hover:shadow-cyan-400/20">
                <MapPin className="w-3 h-3 flex-shrink-0" />
                <span className="truncate max-w-[80px]">{zoneCategory || "Unknown"}</span>
              </div>

              {alerts.length > 0 && (
                <div className="bg-gradient-to-r from-red-600 to-red-700 backdrop-blur-sm text-white text-xs font-bold px-2 py-1.5 rounded-full border border-red-400/30 shadow-lg flex items-center gap-1.5 group-hover:scale-105 transition-transform duration-300 group-hover:border-red-400/50 group-hover:shadow-red-400/20 animate-pulse">
                  <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate">{alerts.length} ALERTS</span>
                </div>
              )}
            </div>
            
            <button
              onClick={handleConfirmation}
              className="relative text-white bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-500 hover:to-pink-500 rounded-xl p-2.5 cursor-pointer transition-all duration-300 transform hover:scale-110 hover:shadow-lg hover:shadow-red-500/30 group/btn flex items-center gap-1.5 flex-shrink-0 hover:shadow-red-500/50"
            >
              <Trash2 size={14} className="group-hover/btn:rotate-12 transition-transform duration-300" />
              <span className="text-xs font-medium hidden sm:block">Delete</span>
              <div className="absolute inset-0 bg-gradient-to-r from-red-600 to-pink-600 rounded-xl blur-sm opacity-50 group-hover/btn:opacity-70 transition-opacity duration-300"></div>
            </button>
          </div>
        </div>
      )}
    </section>
  );
}