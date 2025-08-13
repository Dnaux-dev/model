import React, { useState, useEffect } from 'react';
import { useNavStore } from '../store/navigation-store';
import Header from '../Header';
import LogoLoader from '../LogoLoader';
import useLoadingStore from '../store/loading-store';

const Settings = () => {
  //LOAD BEFORE IT SHOWS SETTINGS PAGE
  const [showSettings, setShowSettings] = useState(false)

  const {showLoading, hideLoading} = useLoadingStore();
  useEffect(() => {
    showLoading();
    const timer = setTimeout(() => {
      hideLoading();
      handleShowSettings()
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  function handleShowSettings () {
    setShowSettings(!showSettings)
  }

  const { setActive } = useNavStore();
  const [notificationSettings, setNotificationSettings] = useState({
    email: false,
    sms: false,
    whatsapp: false,
    push: true,
    desktop: true,
    sound: true,
  });

  const [accountSettings, setAccountSettings] = useState({
    username: 'AdminUser',
    email: 'admin@obex.com',
    role: 'Administrator',
    timezone: 'UTC',
    language: 'English'
  });

  const [securitySettings, setSecuritySettings] = useState({
    twoFactor: false,
    sessionTimeout: 30,
    passwordExpiry: 90,
    loginAttempts: 5
  });

  useEffect(() => {
    setActive('settings');
    const fetchSettings = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/notifications/settings', {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        });
        if (!response.ok) throw new Error('Failed to fetch settings');
        const data = await response.json();
        setNotificationSettings(data);
      } catch (error) {
        console.error('Error fetching settings:', error);
      }
    };
    fetchSettings();
  }, [setActive]);

  const handleSave = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/notifications/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(notificationSettings),
      });
      if (!response.ok) throw new Error('Failed to save settings');
      alert('Settings saved successfully');
    } catch (error) {
      console.error('Error saving settings:', error);
      alert('Failed to save settings');
    }
  };

  return (
    <>
      <Header />
      <LogoLoader />
      {showSettings && (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Enhanced Header Section */}
            <div className="mb-10">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                        <i className="fa-solid fa-cog text-white text-xl"></i>
                      </div>
                      <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    </div>
                    <div>
                      <h1 className="text-[24px] sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold bg-gradient-to-r from-white via-cyan-100 to-white bg-clip-text text-transparent">
                        System Settings
                      </h1>
                      <p className="text-gray-400 mt-2 text-lg">
                        Configure your security system preferences and account settings
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                  <div className="flex items-center gap-3 bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-sm px-4 py-2 rounded-xl border border-slate-600/30">
                    <div className="w-3 h-3 bg-green-400 rounded-full shadow-lg shadow-green-400/50"></div>
                    <span className="text-sm text-gray-300 font-medium">
                      Settings Active
                    </span>
                  </div>
                  
                  <button
                    onClick={handleSave}
                    className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 border border-cyan-400/30"
                  >
                    <i className="fa-solid fa-save text-lg"></i>
                    <span className="font-semibold">Save All Settings</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Settings Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Notification Preferences */}
              <section className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-bell text-white text-lg"></i>
                  </div>
                  <h2 className="text-2xl font-bold text-white">Notification Preferences</h2>
                </div>
                
                <div className="space-y-6">
                  {[
                    { key: 'email', label: 'Email Notifications', icon: 'fa-solid fa-envelope', desc: 'Receive alerts via email' },
                    { key: 'sms', label: 'SMS Notifications', icon: 'fa-solid fa-mobile-alt', desc: 'Get SMS alerts for critical events' },
                    { key: 'whatsapp', label: 'WhatsApp Notifications', icon: 'fa-brands fa-whatsapp', desc: 'Instant messaging alerts' },
                    { key: 'push', label: 'Push Notifications', icon: 'fa-solid fa-mobile-screen', desc: 'Mobile push notifications' },
                    { key: 'desktop', label: 'Desktop Notifications', icon: 'fa-solid fa-desktop', desc: 'Browser desktop notifications' },
                    { key: 'sound', label: 'Sound Alerts', icon: 'fa-solid fa-volume-up', desc: 'Audio alerts for threats' },
                  ].map(({ key, label, icon, desc }) => (
                    <div key={key} className="flex items-center justify-between p-4 bg-gradient-to-r from-slate-800/50 to-slate-700/50 rounded-xl border border-slate-600/30">
                      <div className="flex items-center gap-3">
                        <i className={`${icon} text-cyan-400 text-lg`}></i>
                        <div>
                          <span className="text-white font-semibold">{label}</span>
                          <p className="text-gray-400 text-sm">{desc}</p>
                        </div>
                      </div>
                      <div className="relative">
                        <input
                          type="checkbox"
                          checked={notificationSettings[key]}
                          onChange={(e) =>
                            setNotificationSettings((prev) => ({ ...prev, [key]: e.target.checked }))
                          }
                          className="sr-only peer"
                        />
                        <div className="w-12 h-6 bg-slate-600 rounded-full peer-checked:bg-cyan-500 transition-all duration-300 shadow-lg"></div>
                        <div className="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full peer-checked:translate-x-6 transition-transform duration-300 shadow-md"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Account Settings */}
              <section className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-user text-white text-lg"></i>
                  </div>
                  <h2 className="text-2xl font-bold text-white">Account Settings</h2>
                </div>
                
                <div className="space-y-4">
                  {[
                    { key: 'username', label: 'Username', type: 'text', icon: 'fa-solid fa-user', disabled: false },
                    { key: 'email', label: 'Email Address', type: 'email', icon: 'fa-solid fa-envelope', disabled: false },
                    { key: 'role', label: 'User Role', type: 'text', icon: 'fa-solid fa-shield-alt', disabled: true },
                    { key: 'timezone', label: 'Timezone', type: 'text', icon: 'fa-solid fa-clock', disabled: false },
                    { key: 'language', label: 'Language', type: 'text', icon: 'fa-solid fa-globe', disabled: false },
                  ].map(({ key, label, type, icon, disabled }) => (
                    <div key={key}>
                      <label className="block text-sm text-gray-400 mb-2 font-medium">{label}</label>
                      <div className="relative">
                        <i className={`${icon} absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400`}></i>
                        <input
                          type={type}
                          value={accountSettings[key]}
                          onChange={(e) => setAccountSettings({ ...accountSettings, [key]: e.target.value })}
                          disabled={disabled}
                          className={`w-full pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm ${
                            disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-slate-500/50'
                          }`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Security Settings */}
              <section className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-shield-alt text-white text-lg"></i>
                  </div>
                  <h2 className="text-2xl font-bold text-white">Security Settings</h2>
                </div>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between p-4 bg-gradient-to-r from-slate-800/50 to-slate-700/50 rounded-xl border border-slate-600/30">
                    <div className="flex items-center gap-3">
                      <i className="fa-solid fa-key text-red-400 text-lg"></i>
                      <div>
                        <span className="text-white font-semibold">Two-Factor Authentication</span>
                        <p className="text-gray-400 text-sm">Enhanced account security</p>
                      </div>
                    </div>
                    <div className="relative">
                      <input
                        type="checkbox"
                        checked={securitySettings.twoFactor}
                        onChange={(e) => setSecuritySettings({ ...securitySettings, twoFactor: e.target.checked })}
                        className="sr-only peer"
                      />
                      <div className="w-12 h-6 bg-slate-600 rounded-full peer-checked:bg-red-500 transition-all duration-300 shadow-lg"></div>
                      <div className="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full peer-checked:translate-x-6 transition-transform duration-300 shadow-md"></div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2 font-medium">Session Timeout (minutes)</label>
                      <div className="relative">
                        <i className="fa-solid fa-clock absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400"></i>
                        <input
                          type="number"
                          value={securitySettings.sessionTimeout}
                          onChange={(e) => setSecuritySettings({ ...securitySettings, sessionTimeout: parseInt(e.target.value) })}
                          className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm text-gray-400 mb-2 font-medium">Password Expiry (days)</label>
                      <div className="relative">
                        <i className="fa-solid fa-calendar-alt absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400"></i>
                        <input
                          type="number"
                          value={securitySettings.passwordExpiry}
                          onChange={(e) => setSecuritySettings({ ...securitySettings, passwordExpiry: parseInt(e.target.value) })}
                          className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm text-gray-400 mb-2 font-medium">Max Login Attempts</label>
                      <div className="relative">
                        <i className="fa-solid fa-lock absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400"></i>
                        <input
                          type="number"
                          value={securitySettings.loginAttempts}
                          onChange={(e) => setSecuritySettings({ ...securitySettings, loginAttempts: parseInt(e.target.value) })}
                          className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white border border-slate-600/50 focus:border-cyan-400/50 rounded-xl focus:ring-2 focus:ring-cyan-400/20 transition-all duration-300 backdrop-blur-sm"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              {/* System Information */}
              <section className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-xl hover:shadow-2xl transition-all duration-300">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
                    <i className="fa-solid fa-info-circle text-white text-lg"></i>
                  </div>
                  <h2 className="text-2xl font-bold text-white">System Information</h2>
                </div>
                
                <div className="space-y-4">
                  {[
                    { label: 'System Version', value: 'OBEX v2.1.0', icon: 'fa-solid fa-code-branch' },
                    { label: 'Last Updated', value: '2025-01-15', icon: 'fa-solid fa-calendar-check' },
                    { label: 'Database Status', value: 'Connected', icon: 'fa-solid fa-database' },
                    { label: 'API Status', value: 'Active', icon: 'fa-solid fa-plug' },
                    { label: 'Storage Used', value: '2.4 GB / 10 GB', icon: 'fa-solid fa-hdd' },
                    { label: 'Uptime', value: '99.9%', icon: 'fa-solid fa-server' },
                  ].map(({ label, value, icon }) => (
                    <div key={label} className="flex items-center justify-between p-3 bg-gradient-to-r from-slate-800/30 to-slate-700/30 rounded-lg border border-slate-600/20">
                      <div className="flex items-center gap-3">
                        <i className={`${icon} text-purple-400 text-sm`}></i>
                        <span className="text-gray-300 text-sm">{label}</span>
                      </div>
                      <span className="text-white font-medium text-sm">{value}</span>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Settings;
