// HistoryPage.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { useEventStore } from '../store/history-store';
import VideoModal from '../VideoModal';
import Header from '../Header';
import LogoLoader from '../LogoLoader';
import useLoadingStore from '../store/loading-store';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

export default function History() {
  //LOAD BEFORE IT SHOWS HISTORY PAGE
  const [showHistory, setShowHistory] = useState(false)

  const {showLoading, hideLoading} = useLoadingStore();
  useEffect(() => {
    showLoading();
    const timer = setTimeout(() => {
      hideLoading();
      handleShowHistory()
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  function handleShowHistory () {
    setShowHistory(!showHistory)
  }

  const { events } = useEventStore();
  const [search, setSearch] = useState('');
  const [filterThreat, setFilterThreat] = useState('');
  const [filterDate, setFilterDate] = useState('');

  const formatTimestamp = (ts) => new Date(ts).toLocaleString();

  const filteredEvents = events.filter((e) =>
    (!search || String(e.cameraName || '').toLowerCase().includes(search.toLowerCase())) &&
    (!filterThreat || (e.threatLevel || '') === filterThreat) &&
    (!filterDate || (e.date || '') === filterDate)
  );
  
  const exportToPDF = () => {
    const doc = new jsPDF();
    doc.text('Camera Event History', 14, 15);

    const tableData = filteredEvents.map((e) => [
      e.cameraName,
      e.zoneCategory,
      e.date,
      e.time,
      e.ipAddress,
      e.threatLevel,
      e.type || 'ADDED',
      formatTimestamp(e.timestamp),
    ]);

    autoTable(doc, {
      startY: 20,
      head: [[
        'Camera', 'Zone', 'Date', 'Time', 'IP', 'Threat', 'Action', 'Timestamp'
      ]],
      body: tableData,
    });

    doc.save('camera_history.pdf');
  };

  const getThreatColor = (threatLevel) => {
    switch (threatLevel) {
      case 'High': return 'from-red-600 to-pink-600';
      case 'Medium': return 'from-amber-500 to-orange-500';
      case 'Low': return 'from-emerald-500 to-green-500';
      default: return 'from-slate-600 to-slate-700';
    }
  };

  const getThreatIcon = (threatLevel) => {
    switch (threatLevel) {
      case 'High': return 'fa-solid fa-exclamation-triangle';
      case 'Medium': return 'fa-solid fa-exclamation-circle';
      case 'Low': return 'fa-solid fa-info-circle';
      default: return 'fa-solid fa-circle';
    }
  };

  return (
    <>
      <Header />
      <LogoLoader />

      {showHistory && (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Enhanced Header Section */}
            <div className="mb-10">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                        <i className="fa-solid fa-history text-white text-xl"></i>
                      </div>
                      <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    </div>
                    <div>
                      <h1 className="text-[24px] sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold bg-gradient-to-r from-white via-cyan-100 to-white bg-clip-text text-transparent">
                        Camera Event History
                      </h1>
                      <p className="text-gray-400 mt-2 text-lg">
                        Complete audit trail of all camera activities and security events
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                  <div className="flex items-center gap-3 bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-sm px-4 py-2 rounded-xl border border-slate-600/30">
                    <div className="w-3 h-3 bg-green-400 rounded-full shadow-lg shadow-green-400/50"></div>
                    <span className="text-sm text-gray-300 font-medium">
                      {filteredEvents.length} events found
                    </span>
                  </div>
                  
                  <button
                    onClick={exportToPDF}
                    className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/30"
                  >
                    <i className="fa-solid fa-file-pdf text-lg"></i>
                    <span className="font-semibold">Export PDF</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Enhanced Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Total Events</p>
                    <p className="text-3xl font-bold text-white mt-2">{events.length}</p>
                    <p className="text-xs text-gray-500 mt-1">All time</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-calendar text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">High Threat</p>
                    <p className="text-3xl font-bold text-red-400 mt-2">
                      {events.filter(e => e.threatLevel === 'High').length}
                    </p>
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
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Medium Threat</p>
                    <p className="text-3xl font-bold text-amber-400 mt-2">
                      {events.filter(e => e.threatLevel === 'Medium').length}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Warning alerts</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-amber-500 to-orange-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-exclamation-circle text-white text-xl"></i>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Low Threat</p>
                    <p className="text-3xl font-bold text-emerald-400 mt-2">
                      {events.filter(e => e.threatLevel === 'Low').length}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">Info alerts</p>
                  </div>
                  <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-info-circle text-white text-xl"></i>
                  </div>
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
                      placeholder="Search by camera name..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white placeholder-gray-400 border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm w-full sm:w-80"
                    />
                  </div>
                  
                  <select
                    value={filterThreat}
                    onChange={(e) => setFilterThreat(e.target.value)}
                    className="px-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                  >
                    <option value="">All Threat Levels</option>
                    <option value="High">High Threat</option>
                    <option value="Medium">Medium Threat</option>
                    <option value="Low">Low Threat</option>
                  </select>
                  
                  <input
                    type="date"
                    value={filterDate}
                    onChange={(e) => setFilterDate(e.target.value)}
                    className="px-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                  />
                </div>
              </div>
            </div>

            {/* Enhanced Events Grid */}
            {filteredEvents.length === 0 ? (
              <div className="text-center py-16">
                <div className="w-24 h-24 bg-gradient-to-r from-gray-600 to-gray-700 rounded-full flex items-center justify-center mx-auto mb-6 shadow-xl">
                  <i className="fa-solid fa-history text-gray-400 text-3xl"></i>
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">No events found</h3>
                <p className="text-gray-400 text-lg max-w-md mx-auto">
                  {search || filterThreat || filterDate 
                    ? 'Try adjusting your search or filter criteria'
                    : 'No camera events recorded yet. Events will appear here once cameras are added or activities are detected.'
                  }
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredEvents.map((event, index) => (
                  <div 
                    key={index} 
                    className="bg-gradient-to-br from-slate-800/80 to-slate-700/80 rounded-2xl p-6 border border-slate-600/30 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 backdrop-blur-sm group"
                  >
                    {/* Threat Level Header */}
                    <div className={`bg-gradient-to-r ${getThreatColor(event.threatLevel)} rounded-xl p-3 mb-4 shadow-lg`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <i className={`${getThreatIcon(event.threatLevel)} text-white text-lg`}></i>
                          <span className="text-white font-bold text-sm uppercase">
                            {event.threatLevel} Threat
                          </span>
                        </div>
                        <span className="text-white/80 text-xs font-medium">
                          {event.type || 'ADDED'}
                        </span>
                      </div>
                    </div>

                    {/* Event Details */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <i className="fa-solid fa-video text-cyan-400 text-lg"></i>
                        <h3 className="text-lg font-bold text-white truncate">
                          {String(event.cameraName)}
                        </h3>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="flex items-center gap-2">
                          <i className="fa-solid fa-map-marker-alt text-slate-400"></i>
                          <span className="text-gray-300">
                            Zone: <span className="text-white font-medium">{String(event.zone || 'N/A')}</span>
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <i className="fa-solid fa-calendar text-slate-400"></i>
                          <span className="text-gray-300">
                            Date: <span className="text-white font-medium">{String(event.date)}</span>
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <i className="fa-solid fa-clock text-slate-400"></i>
                          <span className="text-gray-300">
                            Time: <span className="text-white font-medium">{String(event.time)}</span>
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <i className="fa-solid fa-network-wired text-slate-400"></i>
                          <span className="text-gray-300">
                            IP: <span className="text-white font-medium truncate">{String(event.ipAddress)}</span>
                          </span>
                        </div>
                      </div>
                      
                      <div className="pt-3 border-t border-slate-600/30">
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <i className="fa-solid fa-clock text-slate-500"></i>
                          <span className="italic">
                            {formatTimestamp(event.timestamp)}
                          </span>
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
}







//THE HISTORY CODE I MET HERE//

// const History = () => {
//   const { setActive } = useNavStore();
//   const [detections, setDetections] = useState([]);
//   const [filter, setFilter] = useState('');

//   useEffect(() => {
//     setActive('history');
//     const fetchDetections = async () => {
//       try {
//         const response = await fetch('http://localhost:8000/api/detections', {
//           headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
//         });
//         if (!response.ok) throw new Error('Failed to fetch detections');
//         const data = await response.json();
//         setDetections(data);
//       } catch (error) {
//         console.error('Error fetching detections:', error);
//       }
//     };
//     fetchDetections();
//   }, [setActive]);

//   const filteredDetections = detections.filter(
//     (d) =>
//       d.type.toLowerCase().includes(filter.toLowerCase()) ||
//       d.timestamp.toLowerCase().includes(filter.toLowerCase())
//   );

//   return (
//     <div className="min-h-screen bg-gray-900 text-white p-4 sm:p-6 lg:p-10">
//       <h1 className="text-3xl font-bold text-cyan-200 mb-6">Detection History</h1>
//       <div className="mb-4">
//         <input
//           type="text"
//           placeholder="Filter by type or date..."
//           value={filter}
//           onChange={(e) => setFilter(e.target.value)}
//           className="w-full max-w-md px-4 py-2 rounded-md bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-cyan-400"
//         />
//       </div>
//       <div className="overflow-x-auto">
//         <table className="w-full text-left border-collapse">
//           <thead>
//             <tr className="bg-gray-800">
//               <th className="p-3 text-sm font-medium text-cyan-200">Timestamp</th>
//               <th className="p-3 text-sm font-medium text-cyan-200">Type</th>
//               <th className="p-3 text-sm font-medium text-cyan-200">Camera</th>
//               <th className="p-3 text-sm font-medium text-cyan-200">Details</th>
//             </tr>
//           </thead>
//           <tbody>
//             {filteredDetections.length > 0 ? (
//               filteredDetections.map((detection) => (
//                 <tr key={detection.id} className="border-b border-gray-700 hover:bg-gray-700">
//                   <td className="p-3">{new Date(detection.timestamp).toLocaleString()}</td>
//                   <td className="p-3">{detection.type}</td>
//                   <td className="p-3">{detection.cameraId}</td>
//                   <td className="p-3">{detection.details}</td>
//                 </tr>
//               ))
//             ) : (
//               <tr>
//                 <td colSpan={4} className="p-3 text-center">No detections found</td>
//               </tr>
//             )}
//           </tbody>
//         </table>
//       </div>
//     </div>
//   );
// };

// export default History;
