import { useState, useEffect } from "react";
import { useEventStore } from "./store/history-store";
import { useCameraStore } from "./store/camera-store";
import { Camera, Globe, MapPin, Calendar, Clock, X } from "lucide-react";

export default function PopupModal({ onSave, onCancel }) {
  const [ipAddress, setIpAddress] = useState("");
  const [zone, setZone] = useState("");
  const [cameraName, setCameraName] = useState("");
  const [currentDate, setCurrentDate] = useState("");
  const [currentTime, setCurrentTime] = useState("");
  const addEvent = useEventStore((state) => state.addEvent);

  // Update current date and time when component mounts and every minute
  useEffect(() => {
    const updateDateTime = () => {
      const now = new Date();
      setCurrentDate(now.toISOString().split('T')[0]);
      setCurrentTime(now.toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      }));
    };

    updateDateTime();
    const interval = setInterval(updateDateTime, 60000);
    return () => clearInterval(interval);
  }, []);

  function handleSaveClick() {
    if (!ipAddress || !zone || !cameraName) {
      alert("Please fill in all required fields");
      return;
    }

    addEvent({
      cameraName,
      zone,
      date: currentDate,
      time: currentTime,
      ipAddress,
      type: "ADDED",
      timestamp: new Date().toISOString(),
      description: `Added camera "${cameraName}" in ${zone} zone with IP: ${ipAddress}`,
    });

    onSave(ipAddress, zone, cameraName, currentDate, currentTime);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      {/* Animated background patterns */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-blue-500/5 to-purple-500/10"></div>
      
      <div className="relative w-[90vw] max-w-lg rounded-3xl bg-gradient-to-br from-white via-gray-50/80 to-white p-8 shadow-2xl backdrop-blur-xl border border-white/20 dark:from-gray-900 dark:via-gray-800/90 dark:to-gray-900 dark:border-gray-700/30">
        {/* Header with close button */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl shadow-lg">
              <Camera className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
                Add New Camera
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Configure your camera settings
              </p>
            </div>
          </div>
          <button
            onClick={onCancel}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-xl transition-all duration-200 hover:scale-110 group"
          >
            <X className="w-5 h-5 text-gray-500 group-hover:text-gray-700 dark:text-gray-400 dark:group-hover:text-gray-200" />
          </button>
        </div>

        {/* Form fields with enhanced styling */}
        <div className="space-y-6">
          {/* IP Address Field */}
          <div className="group">
            <label className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              <Globe className="w-4 h-4 text-cyan-600" />
              IP Address *
            </label>
            <input
              type="text"
              value={ipAddress}
              onChange={(e) => setIpAddress(e.target.value)}
              placeholder="192.168.1.100"
              className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 bg-white/50 backdrop-blur-sm text-gray-700 placeholder-gray-400 transition-all duration-200 focus:border-cyan-500 focus:ring-4 focus:ring-cyan-500/20 focus:outline-none hover:border-gray-300 dark:bg-gray-800/50 dark:border-gray-600 dark:text-white dark:placeholder-gray-500 dark:focus:border-cyan-400 dark:focus:ring-cyan-400/20"
              required
            />
          </div>

          {/* Zone Category Field */}
          <div className="group">
            <label className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              <MapPin className="w-4 h-4 text-blue-600" />
              Zone Category *
            </label>
            <select
              value={zone}
              onChange={(e) => setZone(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 bg-white/50 backdrop-blur-sm text-gray-700 transition-all duration-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 focus:outline-none hover:border-gray-300 dark:bg-gray-800/50 dark:border-gray-600 dark:text-white dark:focus:border-blue-400 dark:focus:ring-blue-400/20"
              required
            >
              <option value="">-- Select Zone Category --</option>
              <option value="public">Public</option>
              <option value="private">Private</option>
              <option value="closure">Closure</option>
              <option value="vault">Vault</option>
            </select>
          </div>

          {/* Camera Name Field */}
          <div className="group">
            <label className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
              <Camera className="w-4 h-4 text-purple-600" />
              Camera Name *
            </label>
            <input
              type="text"
              value={cameraName}
              onChange={(e) => setCameraName(e.target.value)}
              placeholder="e.g. Backyard Surveillance"
              className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 bg-white/50 backdrop-blur-sm text-gray-700 placeholder-gray-400 transition-all duration-200 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/20 focus:outline-none hover:border-gray-300 dark:bg-gray-800/50 dark:border-gray-600 dark:text-white dark:placeholder-gray-500 dark:focus:border-purple-400 dark:focus:ring-purple-400/20"
              required
            />
          </div>

          {/* Auto-updated Date and Time Fields */}
          <div className="grid grid-cols-2 gap-4">
            {/* <div className="group">
              <label className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                <Calendar className="w-4 h-4 text-green-600" />
                Date Added
              </label>
              <div className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 bg-gray-50/50 backdrop-blur-sm text-gray-700 dark:bg-gray-700/50 dark:border-gray-600 dark:text-gray-300">
                {currentDate}
              </div>
            </div> */}
            {/* <div className="group">
              <label className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-700 dark:text-gray-300">
                <Clock className="w-4 h-4 text-orange-600" />
                Time Added
              </label>
              <div className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 bg-gray-50/50 backdrop-blur-sm text-gray-700 dark:bg-gray-700/50 dark:border-gray-600 dark:text-gray-300">
                {currentTime}
              </div>
            </div> */}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end gap-4 mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onCancel}
            className="px-6 py-3 rounded-xl border-2 border-gray-300 bg-white/80 backdrop-blur-sm text-gray-700 font-medium transition-all duration-200 hover:bg-gray-50 hover:border-gray-400 hover:scale-105 hover:shadow-lg dark:bg-gray-800/80 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:border-gray-500"
          >
            Cancel
          </button>
          <button
            onClick={handleSaveClick}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold shadow-lg transition-all duration-200 hover:from-cyan-600 hover:to-blue-700 hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/25"
          >
            Save Camera
          </button>
        </div>

        {/* Decorative elements */}
        <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-cyan-400/20 to-blue-500/20 rounded-full blur-2xl"></div>
        <div className="absolute -bottom-4 -left-4 w-20 h-20 bg-gradient-to-tr from-purple-400/20 to-pink-500/20 rounded-full blur-2xl"></div>
      </div>
    </div>
  );
}