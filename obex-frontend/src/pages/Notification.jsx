import React, { useState, useEffect } from 'react';
import { useNavStore } from '../store/navigation-store';
import Header from '../Header';
import LogoLoader from '../LogoLoader';
import useLoadingStore from '../store/loading-store';

/*
 * NOTIFICATION SYSTEM - BACKEND INTEGRATION READY
 * 
 * This component is designed to work with backend notifications:
 * 
 * 1. REMOVED: Automatic notification logging out
 * 2. ADDED: Button to trigger sample notifications for testing
 * 3. PREPARED: Backend notification integration
 * 
 * BACKEND INTEGRATION APPROACH:
 * - Use WebSocket connection for real-time notifications
 * - Call receiveBackendNotification() when backend sends data
 * - Example WebSocket implementation:
 *   const ws = new WebSocket('ws://your-backend-url/notifications');
 *   ws.onmessage = (event) => {
 *     const notificationData = JSON.parse(event.data);
 *     receiveBackendNotification(notificationData);
 *   };
 * 
 * - Alternative: HTTP polling with setInterval
 * - Alternative: Server-Sent Events (SSE)
 * 
 * NOTIFICATION DATA STRUCTURE EXPECTED FROM BACKEND:
 * {
 *   type: 'threat' | 'system' | 'maintenance' | 'info',
 *   level: 'High' | 'Medium' | 'Low',
 *   title: 'Notification Title',
 *   message: 'Detailed message content',
 *   priority: 'urgent' | 'high' | 'normal'
 * }
 */

const Notification = () => {
  //LOAD BEFORE IT SHOWS NOTIFICATION PAGE
  const [showNotification, setShowNotification] = useState(false)

  const {showLoading, hideLoading} = useLoadingStore();
  useEffect(() => {
    showLoading();
    const timer = setTimeout(() => {
      hideLoading();
      handleShowNotification()
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  function handleShowNotification () {
    setShowNotification(!showNotification)
  }

  const { setActive } = useNavStore();
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      type: 'threat',
      level: 'High',
      title: 'Unauthorized Access Detected',
      message: 'Multiple failed login attempts detected from IP address 192.168.1.100',
      timestamp: '2025-01-15T14:30:00Z',
      read: false,
      priority: 'urgent'
    },
    {
      id: 2,
      type: 'system',
      level: 'Medium',
      title: 'Camera Offline Alert',
      message: 'Camera CAM-001 has been offline for more than 5 minutes',
      timestamp: '2025-01-15T14:25:00Z',
      read: false,
      priority: 'high'
    },
    {
      id: 3,
      type: 'maintenance',
      level: 'Low',
      title: 'System Maintenance Scheduled',
      message: 'Scheduled maintenance will begin at 2:00 AM tomorrow',
      timestamp: '2025-01-15T14:20:00Z',
      read: true,
      priority: 'normal'
    },
    {
      id: 4,
      type: 'threat',
      level: 'High',
      title: 'Perimeter Breach Alert',
      message: 'Motion detected in restricted zone A-12 during off-hours',
      timestamp: '2025-01-15T14:15:00Z',
      read: false,
      priority: 'urgent'
    },
    {
      id: 5,
      type: 'system',
      level: 'Medium',
      title: 'Storage Space Warning',
      message: 'Storage usage has reached 85% capacity',
      timestamp: '2025-01-15T14:10:00Z',
      read: true,
      priority: 'high'
    },
    {
      id: 6,
      type: 'info',
      level: 'Low',
      title: 'New User Registration',
      message: 'New security officer account created for John Smith',
      timestamp: '2025-01-15T14:05:00Z',
      read: true,
      priority: 'normal'
    }
  ]);

  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');

  useEffect(() => {
    setActive('notification');
  }, [setActive]);

  // Function to add sample notifications
  const addSampleNotification = () => {
    const sampleNotifications = [
      {
        id: Date.now() + Math.random(),
        type: 'threat',
        level: 'High',
        title: 'New Security Breach Detected',
        message: 'Unauthorized access attempt detected in Zone B-15 at the main entrance',
        timestamp: new Date().toISOString(),
        read: false,
        priority: 'urgent'
      },
      {
        id: Date.now() + Math.random() + 1,
        type: 'system',
        level: 'Medium',
        title: 'Network Connectivity Issue',
        message: 'Camera CAM-003 experiencing intermittent connection problems',
        timestamp: new Date().toISOString(),
        read: false,
        priority: 'high'
      },
      {
        id: Date.now() + Math.random() + 2,
        type: 'info',
        level: 'Low',
        title: 'System Update Available',
        message: 'New security patch v2.1.4 is ready for installation',
        timestamp: new Date().toISOString(),
        read: false,
        priority: 'normal'
      }
    ];

    setNotifications(prev => [...sampleNotifications, ...prev]);
  };

  // Function to receive notifications from backend (placeholder for future implementation)
  const receiveBackendNotification = (notificationData) => {
    const newNotification = {
      id: Date.now() + Math.random(),
      type: notificationData.type || 'info',
      level: notificationData.level || 'Medium',
      title: notificationData.title || 'New Notification',
      message: notificationData.message || 'Notification from backend system',
      timestamp: new Date().toISOString(),
      read: false,
      priority: notificationData.priority || 'normal'
    };

    setNotifications(prev => [newNotification, ...prev]);
  };

  // Function to simulate receiving notifications from backend (for testing)
  const simulateBackendNotification = () => {
    const backendNotification = {
      type: 'threat',
      level: 'High',
      title: 'Backend Security Alert',
      message: 'Suspicious activity detected by backend monitoring system',
      priority: 'urgent'
    };

    receiveBackendNotification(backendNotification);
  };

  const markAsRead = (id) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(notif => ({ ...notif, read: true }))
    );
  };

  const deleteNotification = (id) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'threat': return 'fa-solid fa-exclamation-triangle';
      case 'system': return 'fa-solid fa-cog';
      case 'maintenance': return 'fa-solid fa-wrench';
      case 'info': return 'fa-solid fa-info-circle';
      default: return 'fa-solid fa-bell';
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'threat': return 'from-red-500 to-pink-600';
      case 'system': return 'from-cyan-500 to-blue-600';
      case 'maintenance': return 'from-amber-500 to-orange-600';
      case 'info': return 'from-emerald-500 to-green-600';
      default: return 'from-slate-500 to-slate-600';
    }
  };

  const getLevelColor = (level) => {
    switch (level) {
      case 'High': return 'from-red-600 to-pink-600';
      case 'Medium': return 'from-amber-500 to-orange-500';
      case 'Low': return 'from-emerald-500 to-green-500';
      default: return 'from-slate-600 to-slate-700';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'urgent': return 'bg-red-500/20 border-red-500/30 text-red-300';
      case 'high': return 'bg-amber-500/20 border-amber-500/30 text-amber-300';
      case 'normal': return 'bg-slate-500/20 border-slate-500/30 text-slate-300';
      default: return 'bg-slate-500/20 border-slate-500/30 text-slate-300';
    }
  };

  const filteredNotifications = notifications.filter(notif => {
    const matchesFilter = filter === 'all' || notif.type === filter;
    const matchesSearch = notif.title.toLowerCase().includes(search.toLowerCase()) || 
                         notif.message.toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const unreadCount = notifications.filter(n => !n.read).length;
  const urgentCount = notifications.filter(n => n.priority === 'urgent').length;

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <>
      <Header />
      <LogoLoader />
      
      {showNotification && (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Enhanced Header Section */}
            <div className="mb-10">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                        <i className="fa-solid fa-bell text-white text-xl"></i>
                      </div>
                      <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    </div>
                    <div>
                      <h1 className="text-[24px] sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold bg-gradient-to-r from-white via-cyan-100 to-white bg-clip-text text-transparent">
                        Notifications
                      </h1>
                      <p className="text-gray-400 mt-2 text-lg">
                        Stay informed about security alerts and system updates
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                  <div className="flex items-center gap-3 bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-sm px-4 py-2 rounded-xl border border-slate-600/30">
                    <div className="w-3 h-3 bg-red-400 rounded-full shadow-lg shadow-red-400/50"></div>
                    <span className="text-sm text-gray-300 font-medium">
                      {unreadCount} unread, {urgentCount} urgent
                    </span>
                  </div>
                  
                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      onClick={addSampleNotification}
                      className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-blue-400/30"
                    >
                      <i className="fa-solid fa-plus text-lg"></i>
                      <span className="font-semibold">Add Sample</span>
                    </button>
                    
                    <button
                      onClick={simulateBackendNotification}
                      className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-purple-400/30"
                    >
                      <i className="fa-solid fa-server text-lg"></i>
                      <span className="font-semibold">Simulate Backend</span>
                    </button>
                    
                    <button
                      onClick={markAllAsRead}
                      className="bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-emerald-400/30"
                    >
                      <i className="fa-solid fa-check-double text-lg"></i>
                      <span className="font-semibold">Mark All Read</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Total</p>
                    <p className="text-3xl font-bold text-white mt-2">{notifications.length}</p>
                    <p className="text-xs text-gray-500 mt-1">All notifications</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-slate-500 to-slate-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-bell text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Unread</p>
                    <p className="text-3xl font-bold text-cyan-400 mt-2">{unreadCount}</p>
                    <p className="text-xs text-gray-500 mt-1">Require attention</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-envelope-open text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Urgent</p>
                    <p className="text-3xl font-bold text-red-400 mt-2">{urgentCount}</p>
                    <p className="text-xs text-gray-500 mt-1">Critical alerts</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-exclamation-triangle text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Today</p>
                    <p className="text-3xl font-bold text-emerald-400 mt-2">
                      {notifications.filter(n => {
                        const today = new Date().toDateString();
                        return new Date(n.timestamp).toDateString() === today;
                      }).length}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">New today</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-calendar-day text-white text-xl"></i>
                  </div>
                </div>
              </div>
            </div>

            {/* Backend Connection Status and System Info */}
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl mb-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Backend Connection Status */}
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-white flex items-center gap-3">
                    <i className="fa-solid fa-server text-cyan-400"></i>
                    Backend Connection Status
                  </h3>
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-400/50"></div>
                    <span className="text-green-400 font-medium">Ready for Backend Integration</span>
                  </div>
                  <p className="text-gray-400 text-sm">
                    The notification system is prepared to receive real-time updates from your backend server.
                    Use the "Simulate Backend" button to test the integration.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-3 py-1 bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 text-xs rounded-full">
                      WebSocket Ready
                    </span>
                    <span className="px-3 py-1 bg-purple-500/20 border border-purple-500/30 text-purple-300 text-xs rounded-full">
                      HTTP Polling Ready
                    </span>
                    <span className="px-3 py-1 bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs rounded-full">
                      SSE Ready
                    </span>
                  </div>
                </div>

                {/* Notification System Info */}
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-white flex items-center gap-3">
                    <i className="fa-solid fa-info-circle text-blue-400"></i>
                    System Information
                  </h3>
                  <div className="space-y-2 text-sm text-gray-300">
                    <div className="flex justify-between">
                      <span>Notification Types:</span>
                      <span className="text-cyan-400">4 Supported</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Priority Levels:</span>
                      <span className="text-amber-400">3 Levels</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Threat Levels:</span>
                      <span className="text-red-400">3 Levels</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Auto-cleanup:</span>
                      <span className="text-red-400">Disabled</span>
                    </div>
                  </div>
                  <p className="text-gray-400 text-sm">
                    Notifications are now persistent and will remain until manually deleted.
                    The system automatically generates unique IDs and timestamps for each notification.
                  </p>
                </div>
              </div>
            </div>

            {/* Enhanced Filters and Search */}
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl mb-8">
              <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
                <div className="flex flex-wrap gap-3">
                  <div className="relative">
                    <i className="fa-solid fa-search absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                    <input
                      type="text"
                      placeholder="Search notifications..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white placeholder-gray-400 border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm w-full sm:w-80"
                    />
                  </div>
                  
                  <select
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    className="px-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="threat">Threat Alerts</option>
                    <option value="system">System Notifications</option>
                    <option value="maintenance">Maintenance</option>
                    <option value="info">Information</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Enhanced Notifications List */}
            {filteredNotifications.length === 0 ? (
              <div className="text-center py-16">
                <div className="w-24 h-24 bg-gradient-to-r from-gray-600 to-gray-700 rounded-full flex items-center justify-center mx-auto mb-6 shadow-xl">
                  <i className="fa-solid fa-bell-slash text-gray-400 text-3xl"></i>
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">No notifications found</h3>
                <p className="text-gray-400 text-lg max-w-md mx-auto">
                  {search || filter !== 'all' 
                    ? 'Try adjusting your search or filter criteria'
                    : 'You\'re all caught up! No new notifications at the moment.'
                  }
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {filteredNotifications.map((notification) => (
                  <div 
                    key={notification.id} 
                    className={`bg-gradient-to-br from-slate-800/80 to-slate-700/80 rounded-2xl p-6 border border-slate-600/30 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 backdrop-blur-sm ${
                      !notification.read ? 'ring-2 ring-cyan-400/30' : ''
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      {/* Notification Icon */}
                      <div className={`w-12 h-12 bg-gradient-to-r ${getTypeColor(notification.type)} rounded-xl flex items-center justify-center shadow-lg flex-shrink-0`}>
                        <i className={`${getTypeIcon(notification.type)} text-white text-lg`}></i>
                      </div>
                      
                      {/* Notification Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <h3 className="text-lg font-bold text-white truncate">
                              {notification.title}
                            </h3>
                            {!notification.read && (
                              <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getPriorityColor(notification.priority)}`}>
                              {notification.priority.toUpperCase()}
                            </span>
                            <span className={`px-3 py-1 bg-gradient-to-r ${getLevelColor(notification.level)} rounded-full text-white text-xs font-semibold shadow-lg`}>
                              {notification.level}
                            </span>
                          </div>
                        </div>
                        
                        <p className="text-gray-300 mb-3 leading-relaxed">
                          {notification.message}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4 text-sm text-gray-400">
                            <span className="flex items-center gap-2">
                              <i className="fa-solid fa-clock text-slate-500"></i>
                              {formatTimestamp(notification.timestamp)}
                            </span>
                            <span className="flex items-center gap-2">
                              <i className="fa-solid fa-tag text-slate-500"></i>
                              {notification.type.charAt(0).toUpperCase() + notification.type.slice(1)}
                            </span>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            {!notification.read && (
                              <button
                                onClick={() => markAsRead(notification.id)}
                                className="px-3 py-1 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-lg text-sm font-medium transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
                              >
                                Mark Read
                              </button>
                            )}
                            <button
                              onClick={() => deleteNotification(notification.id)}
                              className="px-3 py-1 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white rounded-lg text-sm font-medium transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
                            >
                              <i className="fa-solid fa-trash text-sm"></i>
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default Notification;