import React, { useState, useEffect, useRef } from 'react';
import { useNotificationStore } from '../store/notification-store';
import { Bell, AlertTriangle, Shield, Clock, MapPin, Trash2, Eye, EyeOff, Square, X } from 'lucide-react';
import Header from '../Header';
import LogoLoader from '../LogoLoader';

// Force rebuild - Fixed SquareOff import issue

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [latestAlert, setLatestAlert] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [showRead, setShowRead] = useState(true);
  const [filterType, setFilterType] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const [showZoneEditor, setShowZoneEditor] = useState(false);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [zoneStart, setZoneStart] = useState(null);
  const [currentZone, setCurrentZone] = useState(null);
  const videoRef = useRef(null);
  
  const { notifications, markAsRead, deleteNotification, clearAllNotifications } = useNotificationStore();

  // Poll OBEX alerts and performance
  useEffect(() => {
    const fetchOBEXData = async () => {
      try {
        setIsLoading(true);
        const [theftRes, suspiciousRes, performanceRes, intrusionRes, loiteringRes] = await Promise.all([
          fetch('http://localhost:8000/alerts/scene2'),
          fetch('http://localhost:8000/suspicious_behavior'),
          fetch('http://localhost:8000/performance/scene2'),
          fetch('http://localhost:8000/intrusion_alerts'),
          fetch('http://localhost:8000/loitering_alerts')
        ]);

        const theftData = await theftRes.json();
        const suspiciousData = await suspiciousRes.json();
        const performanceData = await performanceRes.json();
        const intrusionData = await intrusionRes.json();
        const loiteringData = await loiteringRes.json();

        let allAlerts = [];
        
        // Handle theft alerts
        if (theftData.theft_alerts && theftData.theft_alerts.length > 0) {
          allAlerts = allAlerts.concat(
            theftData.theft_alerts.map(a => ({
              type: 'theft',
              id: a.type || 'unknown',
              time: new Date(a.timestamp * 1000).toLocaleTimeString(),
              date: new Date(a.timestamp * 1000).toLocaleDateString(),
              details: a.details || `${a.type} detected`,
              severity: a.severity || 'HIGH',
              timestamp: a.timestamp * 1000
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
              date: new Date(a.timestamp * 1000).toLocaleDateString(),
              details: a.details || `${a.type} behavior detected`,
              severity: a.severity || 'MEDIUM',
              timestamp: a.timestamp * 1000
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
              date: new Date(a.entry_time * 1000).toLocaleDateString(),
              details: `Person ${a.track_id} entered restricted zone`,
              severity: 'HIGH',
              timestamp: a.entry_time * 1000
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
              date: new Date(a.entry_time * 1000).toLocaleDateString(),
              details: `Person ${a.track_id} loitering for ${a.duration?.toFixed(1) || 'unknown'}s`,
              severity: 'MEDIUM',
              timestamp: a.entry_time * 1000
            }))
          );
        }

        // Sort alerts by most recent first
        allAlerts.sort((a, b) => b.timestamp - a.timestamp);

        setAlerts(allAlerts);
        setLatestAlert(allAlerts[0] || null);
        setPerformance(performanceData);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching OBEX data:', error);
        setIsLoading(false);
      }
    };

    fetchOBEXData();
    const interval = setInterval(fetchOBEXData, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getAlertColor = (alertType) => {
    switch (alertType) {
      case 'theft': return 'bg-gradient-to-r from-red-600 to-red-700';
      case 'suspicious': return 'bg-gradient-to-r from-orange-600 to-orange-700';
      case 'intrusion': return 'bg-gradient-to-r from-red-500 to-red-600';
      case 'loitering': return 'bg-gradient-to-r from-yellow-500 to-yellow-600';
      default: return 'bg-gradient-to-r from-yellow-600 to-yellow-700';
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

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'HIGH': return 'text-red-400';
      case 'MEDIUM': return 'text-yellow-400';
      case 'LOW': return 'text-green-400';
      default: return 'text-gray-400';
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

  const filteredAlerts = alerts.filter(alert => {
    if (filterType !== 'all' && alert.type !== filterType) return false;
    return true;
  });

  const alertStats = {
    total: alerts.length,
    theft: alerts.filter(a => a.type === 'theft').length,
    suspicious: alerts.filter(a => a.type === 'suspicious').length,
    intrusion: alerts.filter(a => a.type === 'intrusion').length,
    loitering: alerts.filter(a => a.type === 'loitering').length,
  };

  return (
    <>
      <Header />
      <LogoLoader />
      
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-red-400/10 to-pink-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-orange-400/10 to-red-500/10 rounded-full blur-3xl animate-pulse animation-delay-1000"></div>
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header Section */}
          <div className="mb-10">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div className="space-y-3">
                <div className="flex items-center gap-4">
                  <div className="relative group">
                    <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg transition-all duration-300 group-hover:scale-110">
                      <Bell className="text-white text-xl" />
                    </div>
                    <div className="absolute inset-0 bg-red-400/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  </div>
                  <div>
                    <h1 className="text-[24px] sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold bg-gradient-to-r from-white via-red-100 to-white bg-clip-text text-transparent">
                      Security Alerts
                    </h1>
                    <p className="text-gray-400 mt-2 text-lg">
                      Real-time threat detection and alert management
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <div className="flex items-center gap-3 bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-sm px-4 py-2 rounded-xl border border-slate-600/30">
                  <div className="w-3 h-3 bg-red-400 rounded-full shadow-lg shadow-red-400/50 animate-pulse"></div>
                  <span className="text-sm text-gray-300 font-medium">
                    {alertStats.total} Active Alerts
                  </span>
                </div>
                
                <button
                  onClick={() => setShowRead(!showRead)}
                  className="bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-500 hover:to-slate-600 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl border border-slate-500/30 flex items-center gap-2"
                >
                  {showRead ? <EyeOff size={18} /> : <Eye size={18} />}
                  <span>{showRead ? 'Hide Read' : 'Show All'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Stats Dashboard */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-10">
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Total Alerts</p>
                  <p className="text-3xl font-bold text-white mt-2 group-hover:text-red-400 transition-colors duration-300">{alertStats.total}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <Bell className="text-white text-xl" />
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Theft</p>
                  <p className="text-3xl font-bold text-red-400 mt-2 group-hover:text-red-300 transition-colors duration-300">{alertStats.theft}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-red-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <span className="text-white text-xl">🚨</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Suspicious</p>
                  <p className="text-3xl font-bold text-orange-400 mt-2 group-hover:text-orange-300 transition-colors duration-300">{alertStats.suspicious}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-orange-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <span className="text-white text-xl">⚠️</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Intrusion</p>
                  <p className="text-3xl font-bold text-red-400 mt-2 group-hover:text-red-300 transition-colors duration-300">{alertStats.intrusion}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-red-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <span className="text-white text-xl">🚪</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Loitering</p>
                  <p className="text-3xl font-bold text-yellow-400 mt-2 group-hover:text-yellow-300 transition-colors duration-300">{alertStats.loitering}</p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <span className="text-white text-xl">⏱️</span>
                </div>
              </div>
            </div>
          </div>

          {/* Filters and Controls */}
          <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl mb-8">
            <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
              <div className="flex flex-wrap gap-3 flex-1">
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-red-400/50 rounded-xl focus:ring-2 focus:ring-red-400/20 transition-all duration-300 backdrop-blur-sm"
                >
                  <option value="all">All Alerts</option>
                  <option value="theft">Theft</option>
                  <option value="suspicious">Suspicious Behavior</option>
                  <option value="intrusion">Intrusion</option>
                  <option value="loitering">Loitering</option>
                </select>
                
                <button
                  onClick={clearAllNotifications}
                  className="px-6 py-3 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-red-400/30 flex items-center gap-2"
                >
                  <Trash2 size={18} />
                  <span className="hidden sm:inline">Clear All</span>
                </button>
              </div>
            </div>
          </div>

          {/* Latest Alert Banner */}
          {latestAlert && (
            <div className={`mb-8 ${getAlertColor(latestAlert.type)} text-white p-6 rounded-2xl shadow-xl border border-red-400/30 animate-pulse`}>
              <div className="flex items-center gap-4">
                <span className="text-3xl">{getAlertIcon(latestAlert.type)}</span>
                <div className="flex-1">
                  <h3 className="text-xl font-bold mb-2">
                    {latestAlert.type === 'theft' ? 'THEFT DETECTED!' :
                     latestAlert.type === 'suspicious' ? 'SUSPICIOUS BEHAVIOR!' :
                     latestAlert.type === 'intrusion' ? 'INTRUSION ALERT!' :
                     latestAlert.type === 'loitering' ? 'LOITERING DETECTED!' :
                     'SECURITY ALERT!'}
                  </h3>
                  <p className="text-lg opacity-90">{latestAlert.details}</p>
                  <div className="flex items-center gap-4 mt-2 text-sm opacity-75">
                    <span className="flex items-center gap-1">
                      <Clock size={14} />
                      {latestAlert.time}
                    </span>
                    <span className="flex items-center gap-1">
                      <MapPin size={14} />
                      {latestAlert.date}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-bold px-3 py-1 rounded-full ${getSeverityColor(latestAlert.severity)} bg-white/20`}>
                    {latestAlert.severity}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Alerts List */}
          {isLoading ? (
            <div className="text-center py-20">
              <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-4 shadow-2xl animate-pulse">
                <Bell className="text-white text-2xl" />
              </div>
              <p className="text-gray-400 text-lg">Loading alerts...</p>
            </div>
          ) : filteredAlerts.length === 0 ? (
            <div className="text-center py-20">
              <div className="w-32 h-32 bg-gradient-to-r from-slate-700 to-slate-800 rounded-full flex items-center justify-center mx-auto mb-8 shadow-2xl border border-slate-600/30">
                <Shield className="text-slate-400 text-5xl" />
              </div>
              <h2 className="text-3xl font-bold text-white mb-4">No Alerts Detected</h2>
              <p className="text-gray-400 text-lg max-w-md mx-auto">
                Your security system is currently monitoring without any detected threats.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredAlerts.map((alert, index) => (
                <div
                  key={`${alert.type}-${alert.id}-${index}`}
                  className={`bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-[1.02] group ${getAlertColor(alert.type)}/20`}
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0">
                      <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                        <span className="text-white text-xl">{getAlertIcon(alert.type)}</span>
                      </div>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-bold text-white">
                          {alert.type === 'theft' ? 'Theft Detection' :
                           alert.type === 'suspicious' ? 'Suspicious Behavior' :
                           alert.type === 'intrusion' ? 'Intrusion Alert' :
                           alert.type === 'loitering' ? 'Loitering Detection' :
                           'Security Alert'}
                        </h3>
                        <div className={`text-sm font-bold px-3 py-1 rounded-full ${getSeverityColor(alert.severity)} bg-white/20`}>
                          {alert.severity}
                        </div>
                      </div>
                      
                      <p className="text-gray-300 mb-3">{alert.details}</p>
                      
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        <span className="flex items-center gap-1">
                          <Clock size={14} />
                          {alert.time}
                        </span>
                        <span className="flex items-center gap-1">
                          <MapPin size={14} />
                          {alert.date}
                        </span>
                        <span className="flex items-center gap-1">
                          <AlertTriangle size={14} />
                          ID: {alert.id}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Zone Editor Section */}
          <div className="mt-10 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white">Zone Management</h3>
              <button
                onClick={() => setShowZoneEditor(!showZoneEditor)}
                className="bg-gradient-to-r from-blue-500 to-cyan-600 hover:from-blue-600 hover:to-cyan-700 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-blue-400/30 flex items-center gap-2"
              >
                {showZoneEditor ? <X size={18} /> : <Square size={18} />}
                <span>{showZoneEditor ? 'Hide' : 'Show'} Zone Editor</span>
              </button>
            </div>
            
            {showZoneEditor && (
              <div className="space-y-4">
                <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-2xl p-6 border border-slate-600/30 shadow-2xl relative">
                  <img
                    ref={videoRef}
                    src="http://localhost:8000/video_feed"
                    alt="Zone Editor"
                    className="w-full h-64 md:h-80 lg:h-96 rounded-xl border border-slate-600/50 shadow-lg object-cover bg-slate-800"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    style={{ cursor: isDrawingZone ? 'crosshair' : 'pointer' }}
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

                  {/* Zone Controls */}
                  <div className="absolute top-4 left-4 flex gap-2 z-20">
                    <button
                      onClick={() => setIsDrawingZone(!isDrawingZone)}
                      className={`px-3 py-1 rounded text-sm ${isDrawingZone ? 'bg-red-600' : 'bg-blue-600'} text-white hover:scale-105 transition-transform duration-300`}
                    >
                      {isDrawingZone ? 'Cancel Zone' : 'Draw Zone'}
                    </button>
                    
                    {currentZone && (
                      <button
                        onClick={clearZone}
                        className="bg-gray-600 text-white px-3 py-1 rounded text-sm hover:scale-105 transition-transform duration-300"
                      >
                        Clear Zone
                      </button>
                    )}
                  </div>
                </div>
                
                <div className="text-center text-gray-400 text-sm">
                  {isDrawingZone ? 
                    'Click and drag to draw a restricted zone' : 
                    'Click "Draw Zone" to start defining a restricted area'
                  }
                </div>
              </div>
            )}
          </div>

          {/* Performance Stats */}
          {performance && (
            <div className="mt-10 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl">
              <h3 className="text-xl font-bold text-white mb-4">System Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-sm text-gray-400">FPS</p>
                  <p className="text-2xl font-bold text-green-400">{performance.fps}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-400">Faces Detected</p>
                  <p className="text-2xl font-bold text-blue-400">{performance.faces_detected}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-400">Objects Detected</p>
                  <p className="text-2xl font-bold text-purple-400">{performance.objects_detected}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-400">Zone Status</p>
                  <p className="text-2xl font-bold text-orange-400">{performance.zone_active ? 'Active' : 'Inactive'}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
} 