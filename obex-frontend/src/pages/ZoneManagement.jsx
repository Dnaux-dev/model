// src/pages/ZoneManagement.jsx
import React, { useState, useEffect } from 'react';
import Header from '../Header';
import { useZoneStore } from '../store/zone-store';
import { Dialog } from '@headlessui/react';
import { Plus, Trash2, Pencil, MapPin, Camera, Shield, Activity, BarChart3, Settings, Users, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area, PieChart, Pie, Cell } from 'recharts';
import useLoadingStore from '../store/loading-store';
import LogoLoader from '../LogoLoader';

const defaultZone = {
  id: null,
  name: '',
  camera: '',
  sensitivity: 'Medium',
  description: '',
  status: 'active'
};

const ZoneManagement = () => {
//LOAD BEFORE DISPLAYING PAGE//
    const [showZoneMgt, setShowZoneMgt] = useState(false)

  const {showLoading, hideLoading} = useLoadingStore();
  useEffect(() => {
    showLoading();
    const timer = setTimeout(() => {
      hideLoading();
      handleShowZoneMgt()
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  function handleShowZoneMgt () {
    setShowZoneMgt(!showZoneMgt)
  }

  const { zones, addZone, editZone, deleteZone } = useZoneStore();
  const [isOpen, setIsOpen] = useState(false);
  const [zoneData, setZoneData] = useState(defaultZone);
  const [isEdit, setIsEdit] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isEdit) {
      editZone(zoneData.id, zoneData);
    } else {
      addZone({ ...zoneData, id: Date.now() });
    }
    setZoneData(defaultZone);
    setIsOpen(false);
    setIsEdit(false);
  };

  const handleEdit = (zone) => {
    setZoneData(zone);
    setIsEdit(true);
    setIsOpen(true);
  };

  const heatmapZones = [
    { x: 1, y: 10, zone: 'Zone A' },
    { x: 2, y: 30, zone: 'Zone B' },
    { x: 3, y: 50, zone: 'Zone C' },
    { x: 4, y: 40, zone: 'Zone D' },
    { x: 5, y: 80, zone: 'Zone E' },
    { x: 6, y: 60, zone: 'Zone F' },
  ];

  const zoneStats = [
    { label: 'Total Zones', value: zones.length, icon: MapPin, color: 'from-blue-500 to-cyan-500' },
    { label: 'Active Cameras', value: zones.filter(z => z.camera).length, icon: Camera, color: 'from-green-500 to-emerald-500' },
    { label: 'High Sensitivity', value: zones.filter(z => z.sensitivity === 'High').length, icon: AlertTriangle, color: 'from-red-500 to-pink-500' },
    { label: 'System Status', value: 'Online', icon: CheckCircle, color: 'from-purple-500 to-indigo-500' },
  ];

  const pieData = [
    { name: 'Low', value: zones.filter(z => z.sensitivity === 'Low').length, color: '#10b981' },
    { name: 'Medium', value: zones.filter(z => z.sensitivity === 'Medium').length, color: '#f59e0b' },
    { name: 'High', value: zones.filter(z => z.sensitivity === 'High').length, color: '#ef4444' },
  ];

  return (
    <>
    <Header />
    <LogoLoader />
    {showZoneMgt && (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header Section */}
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
              Zone Management
            </h1>
            <p className="text-slate-300 text-lg max-w-2xl mx-auto">
              Monitor and manage security zones with real-time analytics and intelligent threat detection
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {zoneStats.map((stat, index) => (
              <div key={index} className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/30 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm font-medium">{stat.label}</p>
                    <p className="text-2xl font-bold text-white mt-1">{stat.value}</p>
                  </div>
                  <div className={`w-12 h-12 bg-gradient-to-r ${stat.color} rounded-xl flex items-center justify-center shadow-lg`}>
                    <stat.icon className="w-6 h-6 text-white" />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Action Button */}
          <div className="flex justify-center">
            <button
              onClick={() => {
                setIsOpen(true);
                setZoneData(defaultZone);
                setIsEdit(false);
              }}
              className="group bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white px-8 py-4 rounded-2xl font-semibold text-lg shadow-2xl hover:shadow-cyan-500/30 transition-all duration-300 transform hover:scale-105 flex items-center gap-3"
            >
              <Plus size={24} className="group-hover:rotate-90 transition-transform duration-300" />
              Add New Zone
            </button>
          </div>

          {/* Zones Table */}
          <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-xl rounded-3xl border border-slate-600/30 shadow-2xl overflow-hidden">
            <div className="p-6 border-b border-slate-600/30">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Shield className="w-6 h-6 text-cyan-400" />
                Security Zones
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gradient-to-r from-slate-700/50 to-slate-600/50">
                  <tr>
                    <th className="p-6 text-left text-slate-300 font-semibold">Zone Name</th>
                    <th className="p-6 text-left text-slate-300 font-semibold">Assigned Camera</th>
                    <th className="p-6 text-left text-slate-300 font-semibold">Sensitivity</th>
                    <th className="p-6 text-left text-slate-300 font-semibold">Status</th>
                    <th className="p-6 text-right text-slate-300 font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-600/30">
                  {zones.map((zone, index) => (
                    <tr key={zone.id} className="hover:bg-slate-700/30 transition-colors duration-200">
                      <td className="p-6">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
                            <MapPin className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <p className="font-bold text-white text-lg">{zone.name}</p>
                            <p className="text-slate-400 text-sm">Zone ID: {zone.id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="p-6">
                        <div className="flex items-center gap-2">
                          <Camera className="w-5 h-5 text-cyan-400" />
                          <span className="text-white">{zone.camera || 'Unassigned'}</span>
                        </div>
                      </td>
                      <td className="p-6">
                        <span className={`px-4 py-2 text-sm font-semibold rounded-full text-white ${
                          zone.sensitivity === 'High'
                            ? 'bg-gradient-to-r from-red-500 to-pink-600 shadow-lg shadow-red-500/30'
                            : zone.sensitivity === 'Medium'
                            ? 'bg-gradient-to-r from-amber-500 to-orange-600 shadow-lg shadow-amber-500/30'
                            : 'bg-gradient-to-r from-green-500 to-emerald-600 shadow-lg shadow-green-500/30'
                        }`}>
                          {zone.sensitivity}
                        </span>
                      </td>
                      <td className="p-6">
                        <span className="flex items-center gap-2">
                          <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                          <span className="text-green-400 font-medium">Active</span>
                        </span>
                      </td>
                      <td className="p-6 text-right">
                        <div className="flex items-center justify-end gap-3">
                          <button
                            onClick={() => handleEdit(zone)}
                            className="p-3 bg-gradient-to-r from-blue-500 to-cyan-600 hover:from-blue-400 hover:to-cyan-500 text-white rounded-xl shadow-lg hover:shadow-blue-500/30 transition-all duration-300 transform hover:scale-110"
                          >
                            <Pencil size={18} />
                          </button>
                          <button
                            onClick={() => deleteZone(zone.id)}
                            className="p-3 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-400 hover:to-pink-500 text-white rounded-xl shadow-lg hover:shadow-red-500/30 transition-all duration-300 transform hover:scale-110"
                          >
                            <Trash2 size={18} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Analytics Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Heatmap Chart */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-xl rounded-3xl p-6 border border-slate-600/30 shadow-2xl">
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                <Activity className="w-6 h-6 text-cyan-400" />
                Zone Activity Heatmap
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={heatmapZones}>
                  <defs>
                    <linearGradient id="colorActivity" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                  <XAxis dataKey="x" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: '12px',
                      color: '#f1f5f9'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="y" 
                    stroke="#06b6d4" 
                    strokeWidth={3}
                    fillOpacity={1}
                    fill="url(#colorActivity)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Sensitivity Distribution */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-xl rounded-3xl p-6 border border-slate-600/30 shadow-2xl">
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                <BarChart3 className="w-6 h-6 text-cyan-400" />
                Sensitivity Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: '12px',
                      color: '#f1f5f9'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-6 mt-4">
                {pieData.map((item, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full" style={{ backgroundColor: item.color }}></div>
                    <span className="text-slate-300 text-sm">{item.name}: {item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Modal */}
        <Dialog open={isOpen} onClose={() => setIsOpen(false)} className="relative z-50">
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" aria-hidden="true" />
          <div className="fixed inset-0 flex items-center justify-center p-4">
            <Dialog.Panel className="w-full max-w-md bg-gradient-to-br from-slate-800 to-slate-700 rounded-3xl border border-slate-600/30 shadow-2xl">
              <div className="p-6">
                <Dialog.Title className="text-2xl font-bold text-white mb-6 text-center">
                  {isEdit ? 'Edit Zone' : 'Add New Zone'}
                </Dialog.Title>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
                    <label className="block mb-2 text-sm font-medium text-slate-300">Zone Name</label>
                    <input
                      type="text"
                      className="w-full bg-slate-700/50 border border-slate-600/30 px-4 py-3 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-300"
                      placeholder="Enter zone name"
                      value={zoneData.name}
                      onChange={(e) => setZoneData({ ...zoneData, name: e.target.value })}
                      required
                    />
                  </div>
                  <div>
                    <label className="block mb-2 text-sm font-medium text-slate-300">Assigned Camera</label>
                    <input
                      type="text"
                      className="w-full bg-slate-700/50 border border-slate-600/30 px-4 py-3 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-300"
                      placeholder="Enter camera ID"
                      value={zoneData.camera}
                      onChange={(e) => setZoneData({ ...zoneData, camera: e.target.value })}
                      required
                    />
                  </div>
                  <div>
                    <label className="block mb-2 text-sm font-medium text-slate-300">Sensitivity Level</label>
                    <select
                      className="w-full bg-slate-700/50 border border-slate-600/30 px-4 py-3 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-300"
                      value={zoneData.sensitivity}
                      onChange={(e) => setZoneData({ ...zoneData, sensitivity: e.target.value })}
                    >
                      <option value="Low">Low</option>
                      <option value="Medium">Medium</option>
                      <option value="High">High</option>
                    </select>
                  </div>
                  <div className="flex justify-end gap-4 pt-6">
                    <button
                      type="button"
                      className="px-6 py-3 border border-slate-600/30 text-slate-300 rounded-xl hover:bg-slate-700/50 transition-all duration-300"
                      onClick={() => setIsOpen(false)}
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-cyan-500/30 transition-all duration-300 transform hover:scale-105"
                    >
                      {isEdit ? 'Update Zone' : 'Add Zone'}
                    </button>
                  </div>
                </form>
              </div>
            </Dialog.Panel>
          </div>
        </Dialog>
      </div>
    )}
    </>
  );
};

export default ZoneManagement;
