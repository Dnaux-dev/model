import React, { useState, useEffect } from 'react';
import { useNavStore } from '../store/navigation-store';
import Header from '../Header';
import LogoLoader from '../LogoLoader';
import useLoadingStore from '../store/loading-store';

const Profile = () => {
  const { setActive } = useNavStore();
  const [profile, setProfile] = useState({ 
    name: 'John Doe', 
    email: 'john.doe@obex.com', 
    role: 'Security Administrator',
    avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
    phone: '+1 (555) 123-4567',
    department: 'Security Operations',
    location: 'New York, NY',
    joinDate: '2023-01-15',
    lastLogin: '2025-01-15 14:30',
    status: 'Active',
    bio: 'Experienced security professional with expertise in surveillance systems, access control, and incident response. Dedicated to maintaining the highest standards of security and safety.',
    skills: ['Security Management', 'CCTV Systems', 'Access Control', 'Incident Response', 'Risk Assessment', 'Emergency Planning']
  });

  const [isEditing, setIsEditing] = useState(false);
  const [showLoading, setShowLoading] = useState(false);

  useEffect(() => {
    setActive('profile');
    const fetchProfile = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/auth/profile', {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        });
        if (!response.ok) throw new Error('Failed to fetch profile');
        const data = await response.json();
        setProfile(data);
      } catch (error) {
        console.error('Error fetching profile:', error);
      }
    };
    fetchProfile();
  }, [setActive]);

  const handleSave = async () => {
    setShowLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/auth/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(profile),
      });
      if (!response.ok) throw new Error('Failed to update profile');
      alert('Profile updated successfully');
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating profile:', error);
      alert('Failed to update profile');
    } finally {
      setShowLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Active': return 'from-green-500 to-emerald-600';
      case 'Inactive': return 'from-red-500 to-pink-600';
      case 'Pending': return 'from-yellow-500 to-orange-600';
      default: return 'from-slate-500 to-slate-600';
    }
  };

  return (
    <>
      <Header />
      <LogoLoader />
      
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(6,182,212,0.1),transparent_50%)] opacity-60"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(59,130,246,0.1),transparent_50%)] opacity-60"></div>
        
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Enhanced Header Section */}
          <div className="mb-12">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8">
              <div className="space-y-4">
                <div className="flex items-center gap-6">
                  <div className="relative group">
                    <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-2xl shadow-cyan-500/25">
                      <i className="fa-solid fa-user text-white text-2xl"></i>
                    </div>
                    <div className="absolute inset-0 bg-cyan-400/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  </div>
                  <div>
                    <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-white via-cyan-100 to-white bg-clip-text text-transparent leading-tight">
                      User Profile
                    </h1>
                    <p className="text-gray-300 mt-3 text-lg lg:text-xl max-w-2xl">
                      Manage your account information, security preferences, and professional details
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
                <div className="flex items-center gap-4 bg-gradient-to-r from-slate-800/60 to-slate-700/60 backdrop-blur-xl px-6 py-3 rounded-2xl border border-slate-600/40 shadow-xl">
                  <div className={`w-4 h-4 bg-gradient-to-r ${getStatusColor(profile.status)} rounded-full shadow-lg animate-pulse`}></div>
                  <span className="text-gray-200 font-semibold">
                    Status: {profile.status}
                  </span>
                </div>
                
                <div className="flex gap-4">
                  {isEditing ? (
                    <>
                      <button
                        onClick={() => setIsEditing(false)}
                        className="px-8 py-4 bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-500 hover:to-slate-600 text-white rounded-2xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-slate-500/40 text-lg"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleSave}
                        disabled={showLoading}
                        className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-2xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/40 disabled:opacity-50 disabled:cursor-not-allowed text-lg"
                      >
                        {showLoading ? (
                          <div className="flex items-center gap-3">
                            <i className="fa-solid fa-spinner fa-spin"></i>
                            Saving...
                          </div>
                        ) : (
                          <div className="flex items-center gap-3">
                            <i className="fa-solid fa-save"></i>
                            Save Changes
                          </div>
                        )}
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => setIsEditing(true)}
                      className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-2xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/40 text-lg"
                    >
                      <i className="fa-solid fa-edit mr-3"></i>
                      Edit Profile
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Profile Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
            {/* Profile Card */}
            <div className="lg:col-span-1">
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl p-10 border border-white/20 shadow-2xl hover:shadow-cyan-500/10 transition-all duration-500 group">
                {/* Avatar Section */}
                <div className="text-center mb-10">
                  <div className="relative inline-block group">
                    <img
                      src={profile.avatar}
                      alt="Profile Avatar"
                      className="w-40 h-40 rounded-full border-4 border-cyan-400/40 shadow-2xl shadow-cyan-500/25 transition-all duration-300 group-hover:border-cyan-400/60 group-hover:scale-105"
                    />
                    <div className="absolute -bottom-3 -right-3 w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full flex items-center justify-center shadow-xl shadow-cyan-500/50 cursor-pointer hover:scale-110 transition-all duration-300">
                      <i className="fa-solid fa-camera text-white text-lg"></i>
                    </div>
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/20 to-blue-400/20 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-xl"></div>
                  </div>
                  <h2 className="text-3xl font-bold text-white mt-6 mb-2">{profile.name}</h2>
                  <p className="text-cyan-300 font-semibold text-lg mb-4">{profile.role}</p>
                  <div className={`inline-block px-6 py-3 bg-gradient-to-r ${getStatusColor(profile.status)} rounded-full text-white text-sm font-bold shadow-lg shadow-cyan-500/25`}>
                    {profile.status}
                  </div>
                </div>

                {/* Quick Stats */}
                <div className="space-y-5 mb-10">
                  <div className="bg-gradient-to-r from-slate-800/60 to-slate-700/60 rounded-2xl p-5 border border-slate-600/40 hover:border-slate-500/60 transition-all duration-300">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm font-medium">Department</span>
                      <span className="text-white font-semibold">{profile.department}</span>
                    </div>
                  </div>
                  <div className="bg-gradient-to-r from-slate-800/60 to-slate-700/60 rounded-2xl p-5 border border-slate-600/40 hover:border-slate-500/60 transition-all duration-300">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm font-medium">Location</span>
                      <span className="text-white font-semibold">{profile.location}</span>
                    </div>
                  </div>
                  <div className="bg-gradient-to-r from-slate-800/60 to-slate-700/60 rounded-2xl p-5 border border-slate-600/40 hover:border-slate-500/60 transition-all duration-300">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm font-medium">Member Since</span>
                      <span className="text-white font-semibold">{profile.joinDate}</span>
                    </div>
                  </div>
                </div>

                {/* Last Login */}
                <div className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-2xl p-5 border border-cyan-500/40 hover:border-cyan-400/60 transition-all duration-300">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                      <i className="fa-solid fa-clock text-white text-xl"></i>
                    </div>
                    <div>
                      <p className="text-cyan-200 text-sm font-semibold">Last Login</p>
                      <p className="text-white font-medium">{profile.lastLogin}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Profile Details */}
            <div className="lg:col-span-2">
              <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl p-10 border border-white/20 shadow-2xl hover:shadow-cyan-500/10 transition-all duration-500">
                <div className="flex items-center gap-4 mb-8">
                  <div className="w-14 h-14 bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl flex items-center justify-center shadow-xl shadow-emerald-500/25">
                    <i className="fa-solid fa-user-edit text-white text-2xl"></i>
                  </div>
                  <h2 className="text-3xl font-bold text-white">Profile Information</h2>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {[
                    { key: 'name', label: 'Full Name', type: 'text', icon: 'fa-solid fa-user', required: true },
                    { key: 'email', label: 'Email Address', type: 'email', icon: 'fa-solid fa-envelope', required: true },
                    { key: 'phone', label: 'Phone Number', type: 'tel', icon: 'fa-solid fa-phone', required: false },
                    { key: 'department', label: 'Department', type: 'text', icon: 'fa-solid fa-building', required: false },
                    { key: 'location', label: 'Location', type: 'text', icon: 'fa-solid fa-map-marker-alt', required: false },
                    { key: 'role', label: 'User Role', type: 'text', icon: 'fa-solid fa-shield-alt', required: true, disabled: true },
                  ].map(({ key, label, type, icon, required, disabled }) => (
                    <div key={key} className={key === 'role' ? 'md:col-span-2' : ''}>
                      <label className="block text-sm text-gray-300 mb-3 font-semibold">
                        {label} {required && <span className="text-red-400">*</span>}
                      </label>
                      <div className="relative group">
                        <i className={`${icon} absolute left-5 top-1/2 transform -translate-y-1/2 text-slate-400 text-lg group-hover:text-cyan-400 transition-colors duration-300`}></i>
                        <input
                          type={type}
                          value={profile[key]}
                          onChange={(e) => setProfile({ ...profile, [key]: e.target.value })}
                          disabled={disabled || !isEditing}
                          className={`w-full pl-14 pr-5 py-4 bg-gradient-to-r from-slate-700/80 to-slate-800/80 text-white border border-slate-600/50 focus:border-cyan-400/60 focus:ring-4 focus:ring-cyan-400/20 rounded-2xl focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm text-lg ${
                            disabled || !isEditing ? 'opacity-50 cursor-not-allowed' : 'hover:border-slate-500/60 hover:shadow-lg hover:shadow-slate-500/20'
                          }`}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Additional Information */}
                <div className="mt-12 pt-10 border-t border-slate-600/40">
                  <h3 className="text-2xl font-bold text-white mb-6">Additional Information</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                      <label className="block text-sm text-gray-300 mb-3 font-semibold">Bio</label>
                      <textarea
                        rows={5}
                        value={profile.bio}
                        onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
                        placeholder="Tell us about yourself..."
                        className="w-full pl-5 pr-5 py-4 bg-gradient-to-r from-slate-700/80 to-slate-800/80 text-white border border-slate-600/50 focus:border-cyan-400/60 rounded-2xl focus:ring-4 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm resize-none text-lg hover:border-slate-500/60 hover:shadow-lg hover:shadow-slate-500/20"
                        disabled={!isEditing}
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm text-gray-300 mb-3 font-semibold">Skills & Expertise</label>
                      <div className="flex flex-wrap gap-3">
                        {profile.skills.map((skill, index) => (
                          <span
                            key={index}
                            className="px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-200 text-sm rounded-xl border border-cyan-500/40 hover:border-cyan-400/60 hover:bg-gradient-to-r hover:from-cyan-500/30 hover:to-blue-500/30 transition-all duration-300 cursor-pointer hover:scale-105"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Profile;
