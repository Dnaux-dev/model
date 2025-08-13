import React, { useState, useEffect, useRef } from 'react';
import { useEventStore } from './store/history-store';
import { useNotificationStore } from './store/notification-store';
import Header from './Header';
import LogoLoader from './LogoLoader';
import CameraCard from './CameraCard';
import PopupModal from './PopupModal';
import useLoadingStore from './store/loading-store';

export default function Dashboard() {
  //LOAD BEFORE IT SHOWS DASHBOARD PAGE
  const [showMain, setShowMain] = useState(false)

  const {showLoading, hideLoading} = useLoadingStore();
  useEffect(() => {
    showLoading();
    const timer = setTimeout(() => {
      hideLoading();
      handleShowMain()
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  function handleShowMain () {
    setShowMain(!showMain)
  }

  const { addEvent } = useEventStore();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [showWebcam, setShowWebcam] = useState(false);
  const [stream, setStream] = useState(null);
  const videoRef = useRef(null);

  // Simple backend video feed - no camera store needed
  const backendVideoFeed = {
    id: 'backend-feed',
    cameraName: 'Backend Video Feed',
    zoneCategory: 'AI Security',
    date: new Date().toLocaleDateString(),
    time: new Date().toLocaleTimeString(),
    ipAddress: 'localhost:8000',
    threatLevel: 'Low',
    status: 'active',
    url: 'http://localhost:8000/video_feed'
  };

  const switchBackendSource = async (source) => {
    try {
      const body = source === 'webcam' ? { source: 'webcam' } : { source: 'file', path: 'scene 2.mp4' };
      await fetch('http://localhost:8000/set_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
    } catch (e) {
      console.error('Failed to switch backend source', e);
    }
  };

  const handleModalSave = (ipAddress, zone, cameraName, date, time) => {
    const newCamera = {
      id: Date.now().toString(),
      cameraName: cameraName,
      zoneCategory: zone,
      date,
      time,
      ipAddress,
      threatLevel: 'Low',
      status: 'active',
      url: `rtsp://${ipAddress}:554/stream`
    };
    // addToCameraStreams(newCamera); // This line is removed as per the edit hint
    addEvent({
      cameraName: cameraName,
      zoneCategory: zone,
      date,
      time,
      ipAddress,
      threatLevel: 'Low',
      timestamp: Date.now(),
    });
    setIsModalOpen(false);
  };

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play().catch(err => {
          console.error("Autoplay failed:", err);
        });
      };
    }
  }, [stream]);

  const handleWebcamAccess = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }, 
        audio: false 
      });
      setStream(mediaStream);
      setShowWebcam(true);
      
      // Ensure video element is properly updated
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(console.error);
        };
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
      alert('Unable to access webcam. Please check permissions and ensure no other application is using the camera.');
    }
  };

  const closeWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setShowWebcam(false);
  };

  // Simple filtered cameras - just the backend feed
  const filteredCameras = [backendVideoFeed].filter(camera => {
    if (!camera || !camera.cameraName) return false;
    
    const cameraName = camera.cameraName.toString().toLowerCase();
    const searchTermLower = searchTerm.toLowerCase();
    
    return cameraName.includes(searchTermLower);
  });

  const getSystemStats = () => {
    const totalCameras = 1; // Just the backend feed
    const activeCameras = 1; // Backend feed is always active
    const highThreats = 0; // Will be updated by alerts
    const totalEvents = 1; // Backend feed event
    
    // Get alerts from notification store
    const { notifications } = useNotificationStore.getState();
    const totalAlerts = notifications.length;
    const unreadAlerts = notifications.filter(n => !n.read).length;
    
    return { totalCameras, activeCameras, highThreats, totalEvents, totalAlerts, unreadAlerts };
  };

  const stats = getSystemStats();

  return (
    <>
      <Header />
      <LogoLoader />
      {isModalOpen && (
        <PopupModal
          onSave={handleModalSave}
          onCancel={() => setIsModalOpen(false)}
        />
      )}
      
      {showMain && (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
          {/* Animated Background Elements */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-cyan-400/10 to-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
            <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-blue-400/10 to-cyan-500/10 rounded-full blur-3xl animate-pulse animation-delay-1000"></div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-cyan-400/5 to-blue-500/5 rounded-full blur-3xl animate-ping"></div>
          </div>

          <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Enhanced Header Section */}
            <div className="mb-10">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="relative group">
                      <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg transition-all duration-300 group-hover:scale-110">
                        <i className="fa-solid fa-video text-white text-xl"></i>
                      </div>
                      <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    </div>
                    <div>
                      <h1 className="text-[24px] sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold bg-gradient-to-r from-white via-cyan-100 to-white bg-clip-text text-transparent">
                        Security Dashboard
                      </h1>
                      <p className="text-gray-400 mt-2 text-lg">
                        Real-time monitoring and control center for your security system
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                  <div className="flex items-center gap-3 bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-sm px-4 py-2 rounded-xl border border-slate-600/30">
                    <div className="w-3 h-3 bg-green-400 rounded-full shadow-lg shadow-green-400/50 animate-pulse"></div>
                    <span className="text-sm text-gray-300 font-medium">
                      System Online
                    </span>
                  </div>
                  
                  {/* Backend source controls */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => switchBackendSource('webcam')}
                      className="bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl border border-emerald-400/30"
                    >
                      Use Webcam
                    </button>
                    <button
                      onClick={() => switchBackendSource('file')}
                      className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl border border-cyan-400/30"
                    >
                      Use Test Video
                    </button>
                  </div>

                  <button
                    onClick={() => setIsModalOpen(true)}
                    className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/30 animate-bounce"
                  >
                    <i className="fa-solid fa-plus text-lg"></i>
                    <span className="font-semibold">Add Camera</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Enhanced Stats Dashboard */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Total Cameras</p>
                    <p className="text-3xl font-bold text-white mt-2 group-hover:text-cyan-400 transition-colors duration-300">{stats.totalCameras}</p>
                    <p className="text-xs text-gray-500 mt-1">Connected devices</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <i className="fa-solid fa-video text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Active Cameras</p>
                    <p className="text-3xl font-bold text-emerald-400 mt-2 group-hover:text-emerald-300 transition-colors duration-300">{stats.activeCameras}</p>
                    <p className="text-xs text-gray-500 mt-1">Live streams</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <i className="fa-solid fa-play text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Active Alerts</p>
                    <p className="text-3xl font-bold text-red-400 mt-2 group-hover:text-red-300 transition-colors duration-300">{stats.unreadAlerts}</p>
                    <p className="text-xs text-gray-500 mt-1">Unread alerts</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <i className="fa-solid fa-exclamation-triangle text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 group">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Total Alerts</p>
                    <p className="text-3xl font-bold text-orange-400 mt-2 group-hover:text-orange-300 transition-colors duration-300">{stats.totalAlerts}</p>
                    <p className="text-xs text-gray-500 mt-1">All time alerts</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-red-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <i className="fa-solid fa-bell text-white text-xl"></i>
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Search and Controls */}
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl mb-8">
              <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
                <div className="flex flex-wrap gap-3 flex-1">
                  <div className="relative flex-1 min-w-[280px]">
                    <i className="fa-solid fa-search absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                    <input
                      type="text"
                      placeholder="Search cameras by name..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white placeholder-gray-400 border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Camera Streams Section */}
            {filteredCameras.length === 0 ? (
              <div className="text-center py-20">
                <div className="w-32 h-32 bg-gradient-to-r from-slate-700 to-slate-800 rounded-full flex items-center justify-center mx-auto mb-8 shadow-2xl border border-slate-600/30">
                  <i className="fa-solid fa-video text-slate-400 text-5xl"></i>
                </div>
                <h2 className="text-3xl font-bold text-white mb-4">No Cameras Connected</h2>
                <p className="text-gray-400 text-lg max-w-md mx-auto mb-8">
                  Get started by adding your first security camera to begin monitoring your premises.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={() => setIsModalOpen(true)}
                    className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/30 flex items-center gap-3"
                  >
                    <i className="fa-solid fa-plus text-xl"></i>
                    Add Your First Camera
                  </button>
                  <button
                    onClick={handleWebcamAccess}
                    className="bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-emerald-400/30 flex items-center gap-3"
                  >
                    <i className="fa-solid fa-camera text-xl"></i>
                    Access Webcam
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Live Feed Header */}
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                      <i className="fa-solid fa-broadcast-tower text-white text-lg"></i>
                    </div>
                    <h2 className="text-2xl font-bold text-white">Live Camera Feeds</h2>
                    <span className="px-3 py-1 bg-gradient-to-r from-green-500 to-emerald-600 text-white text-sm font-semibold rounded-full shadow-lg">
                      {filteredCameras.length} Active
                    </span>
                  </div>
                  
                  <div className="flex gap-3">
                    <button
                      onClick={() => setIsModalOpen(true)}
                      className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/30 flex items-center gap-2"
                    >
                      <i className="fa-solid fa-plus text-lg"></i>
                      Add Camera
                    </button>
                    <button
                      onClick={handleWebcamAccess}
                      className="bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-emerald-400/30 flex items-center gap-2"
                    >
                      <i className="fa-solid fa-camera text-lg"></i>
                      Webcam
                    </button>
                  </div>
                </div>

                {/* Camera Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-3 gap-8 mb-10">
                  {filteredCameras.map((camera, index) => (
                    <div key={index} className="w-full">
                      <CameraCard {...camera} />
                    </div>
                  ))}
                </div>
              </>
            )}

            {/* Enhanced Webcam Section */}
            {showWebcam && (
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-xl mb-8 relative z-20">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
                      <i className="fa-solid fa-camera text-white text-lg"></i>
                    </div>
                    <h3 className="text-2xl font-bold text-white">Webcam Access</h3>
                  </div>
                  <button
                    onClick={closeWebcam}
                    className="bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-red-400/30"
                  >
                    <i className="fa-solid fa-times text-lg"></i>
                  </button>
                </div>
                
                <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-2xl p-6 border border-slate-600/30 shadow-2xl">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-64 md:h-80 lg:h-96 rounded-xl border border-slate-600/50 shadow-lg object-cover bg-slate-800"
                    style={{ minHeight: '256px' }}
                  />
                  {!stream && (
                    <div className="w-full h-64 md:h-80 lg:h-96 rounded-xl border border-slate-600/50 bg-slate-800 flex items-center justify-center">
                      <div className="text-center text-slate-400">
                        <i className="fa-solid fa-camera text-4xl mb-4"></i>
                        <p>Initializing webcam...</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}



