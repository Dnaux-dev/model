import { useState } from "react";
import primusLogo from './obex-logo.png';
import { useNavStore } from "./store/navigation-store";
import { Link } from "react-router-dom";

export default function Header({ addCameraStream }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const { active, setActive } = useNavStore();

  function handleDropdownDashboard() {
    setIsDashboardOpen(!isDashboardOpen);
  }

  function handleDropdownSettings() {
    setIsSettingsOpen(!isSettingsOpen);
  }

  return (
    <header className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 backdrop-blur-xl shadow-2xl sticky top-0 z-50 border-b border-slate-600/30">
      <nav className="mx-4 sm:mx-6 lg:mx-10">
        <section className="flex items-center justify-between h-20 xl:h-24">
          {/* Logo */}
          <Link to="/" className="flex items-center group">
            <div className="relative">
              <img
                src={primusLogo}
                alt="primus logo"
                className="w-12 h-12 md:w-14 md:h-14 lg:w-16 lg:h-16 xl:w-18 xl:h-18 cursor-pointer transition-all duration-300 group-hover:scale-110 group-hover:rotate-3"
              />
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-300 blur-xl"></div>
            </div>
            <span className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-white cursor-pointer ml-3 md:text-[24px] lg:text-[28px] xl:text-[32px] tracking-wider">
              OBEX
              <span className="text-cyan-400 font-black ml-1">.</span>
            </span>
          </Link>

          {/* NOTIFICATION AND ALERT ICON FOR MOBILE */}
          <div className="md:hidden flex items-center">
            <button className="relative group inline-block text-white p-3 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 transition-all duration-300 transform hover:scale-105 shadow-lg">
              <span><i className="fa-regular fa-bell text-white text-[18px]"></i></span>
              <div className="absolute top-full left-1/2 -translate-x-1/2 hidden group-hover:block bg-slate-800 text-white whitespace-nowrap z-10 px-3 py-2 rounded-lg shadow-xl border border-slate-600 mt-2">
                <span className="text-sm font-medium">Intrusion Alert</span>
              </div>
            </button>
          </div>

          {/* Hamburger (Mobile) */}
          <div className="md:hidden">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="text-white focus:outline-none p-2 rounded-lg hover:bg-slate-700/50 transition-all duration-300"
              aria-label="Toggle Menu"
            >
              <svg
                className="w-7 h-7 text-cyan-400 cursor-pointer transition-all duration-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                {menuOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2.5"
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2.5"
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </button>
          </div>

          {/* Nav Links (Desktop) */}
          <article className="hidden md:flex space-x-2 items-center">
            <Link to= '/dashboard'
              key="dashboard"
              onClick={() => { handleDropdownDashboard(); setActive('dashboard'); }}
              className={`relative text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 md:text-[12px] lg:text-[14px] xl:text-[16px] group ${active === 'dashboard' ? 'bg-gradient-to-r from-slate-700 to-slate-600 ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <article className="flex items-center gap-3 group">
                <i className="fa-solid fa-table-columns text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i>
                <span className="group-hover:text-cyan-100 transition-colors duration-300">Dashboard</span>
                <span className="transition-transform duration-300 group-hover:rotate-180">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="xl:w-5 xl:h-4 md:w-4 md:h-3 text-cyan-400"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </span>
              </article>
              {isDashboardOpen && (
                <article className="w-[240px] h-[180px] bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl absolute flex justify-around flex-col p-6 mt-4 transition-all duration-300 ease-out shadow-2xl border border-slate-600/30 backdrop-blur-xl">
                  <h3 className="group flex items-center gap-3 p-3 rounded-xl hover:bg-slate-700/50 transition-all duration-300">
                    <span><i className="fa-solid fa-camera text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Security Cameras</span>
                  </h3>
                  <h3 className="group flex items-center gap-3 p-3 rounded-xl hover:bg-slate-700/50 transition-all duration-300">
                    <span><i className="fa-solid fa-video text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Heatmap/Analytics</span>
                  </h3>
                  <h3 className="group flex items-center gap-3 p-3 rounded-xl hover:bg-slate-700/50 transition-all duration-300">
                    <span><i className="fa-solid fa-chart-line text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Active Alerts</span>
                  </h3>
                </article>
              )}
            </Link>

            <Link to="/zone-management"
              key="zones"
              onClick={() => setActive('zones')}
              className={`text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px] ${active === 'zones' ? 'bg-gradient-to-r from-slate-700 to-slate-600 ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"><i className="fa-solid fa-money-bill-trend-up"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">Zones</span>
            </Link>

            <Link to="/history"
              key="history"
              onClick={() => setActive('history')}
              className={`text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px] ${active === 'history' ? 'bg-gradient-to-r from-slate-700 to-slate-600 ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"><i className="fa-solid fa-clock-rotate-left"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">History</span>
            </Link>

            <Link
              to="/settings"
              key="settings"
              onClick={() => { handleDropdownSettings(); setActive('settings'); }}
              className={`text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 md:text-[12px] lg:text-[14px] xl:text-[16px] ${active === 'settings' ? 'bg-gradient-to-r from-slate-700 to-slate-600 ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <article className="flex items-center gap-3 group">
                <i className="fas fa-gear text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i>
                <span className="group-hover:text-cyan-100 transition-colors duration-300">Settings</span>
                <span className="transition-transform duration-300 group-hover:rotate-180">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="xl:w-5 xl:h-4 md:w-4 md:h-3 text-cyan-400"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </span>
              </article>
              {isSettingsOpen && (
                <article className="w-[260px] h-[180px] bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl absolute flex justify-around flex-col p-6 mt-4 transition-all duration-300 ease-out shadow-2xl border border-slate-600/30 backdrop-blur-xl">
                  <h3 className="group flex items-center gap-3 p-3 rounded-xl hover:bg-slate-700/50 transition-all duration-300">
                    <span><i className="fa-solid fa-camera text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">System Settings</span>
                  </h3>
                  <h3 className="group flex items-center gap-3 p-3 rounded-xl hover:bg-slate-700/50 transition-all duration-300">
                    <span><i className="fa-solid fa-video text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Device Management</span>
                  </h3>
                </article>
              )}
            </Link>
          </article>

          <div className="hidden md:flex space-x-2 items-center lg:gap-2">
            <Link to="/profile"
              onClick={() => setActive('profile')}
              className={`text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 bg-gradient-to-r from-slate-800 to-slate-700 lg:text-[14px] xl:text-[16px] flex items-center gap-3 group ${active === 'profile' ? 'ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <div className="flex flex-row justify-center items-center gap-3">
                <span><i className="fa-regular fa-user text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                <span className="hidden lg:inline group-hover:text-cyan-100 transition-colors duration-300">Profile</span>
              </div>
            </Link>
            <Link to="/notifications"
              onClick={() => setActive('notification')}
              className={`text-white md:px-4 md:py-3 lg:px-4 lg:py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 bg-gradient-to-r from-slate-800 to-slate-700 lg:text-[14px] xl:text-[16px] flex items-center gap-3 group ${active === 'notification' ? 'ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <div className="flex flex-row justify-center items-center gap-3">
                <span><i className="fa-regular fa-bell text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                <span className="hidden lg:inline group-hover:text-cyan-100 transition-colors duration-300">Notifications</span>
              </div>
            </Link>

            <Link to="/alerts"
              onClick={() => setActive('alert')}
              className={`relative group inline-block text-white md:px-4 md:py-3 lg:px-4 lg:py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 ${active === 'alert' ? 'ring-2 ring-cyan-400/50 shadow-lg' : 'hover:shadow-lg'}`}
            >
              <span className="flex items-center gap-3">
                <i className="fa-solid fa-exclamation-triangle text-amber-400 text-[18px] md:text-[16px] group-hover:text-amber-300 transition-colors duration-300"></i>
                <span className="hidden lg:inline group-hover:text-cyan-100 transition-colors duration-300">Alerts</span>
              </span>
              <div className="absolute top-full left-1/2 -translate-x-1/2 hidden group-hover:block bg-slate-800 text-white whitespace-nowrap z-10 px-3 py-2 rounded-lg shadow-xl border border-slate-600 mt-2">
                <span className="text-sm font-medium">Security Alerts</span>
              </div>
            </Link>

            <Link to='/'
              onClick={() => setActive('logout')}
              className={`text-white md:px-4 md:py-3 lg:px-4 lg:py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-cyan-600 hover:to-blue-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 bg-gradient-to-r from-cyan-600 to-blue-600 md:text-[12px] xl:text-[16px] shadow-lg hover:shadow-xl transform hover:scale-105 ${active === 'logout' ? 'ring-2 ring-cyan-400/50' : ''}`}
            >
              <span className="flex items-center gap-2">
                <i className="fa-solid fa-sign-out-alt mr-1"></i>
                Logout
              </span>
            </Link>
          </div>
        </section>

        {/* Nav Links (Mobile dropdown) */}
        {menuOpen && (
          <div className="flex flex-col md:hidden mt-4 space-y-2 pb-6 bg-gradient-to-b from-slate-800/50 to-transparent rounded-2xl p-4 backdrop-blur-xl border border-slate-600/30">
            <Link to='/'
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 md:text-[12px] xl:text-[16px] relative lg:text-[14px] group"
              onClick={() => setActive('home')}
            >
              <article className="flex items-center gap-3 group">
                <i className="fa-solid fa-home text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i>
                <span className="group-hover:text-cyan-100 transition-colors duration-300">Home</span>
              </article>
            </Link>

            <Link to='/dashboard'
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 md:text-[12px] xl:text-[16px] relative lg:text-[14px] group"
              onClick={handleDropdownDashboard}
            >
              <article className="flex items-center gap-3 group">
                <i className="fa-solid fa-table-columns text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i>
                <span className="group-hover:text-cyan-100 transition-colors duration-300">Dashboard</span>
                <i className="fa-solid fa-caret-down text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i>
              </article>
              {isDashboardOpen && (
                <article className="w-full h-[180px] bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl flex justify-around flex-col p-4 mt-3 transition-all duration-300 ease-out shadow-lg border border-slate-600/30">
                  <h3 className="group flex items-center gap-3 p-3 rounded-lg hover:bg-slate-600/50 transition-all duration-300">
                    <span><i className="fa-solid fa-camera text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Security Cameras</span>
                  </h3>
                  <h3 className="group flex items-center gap-3 p-3 rounded-lg hover:bg-slate-600/50 transition-all duration-300">
                    <span><i className="fa-solid fa-video text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Heatmap/Analytics</span>
                  </h3>
                  <h3 className="group flex items-center gap-3 p-3 rounded-lg hover:bg-slate-600/50 transition-all duration-300">
                    <span><i className="fa-solid fa-chart-line text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Active Alerts</span>
                  </h3>
                </article>
              )}
            </Link>

            <Link to="/zone-management"
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px]"
              onClick={() => setActive('zones')}
            >
              <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"><i className="fa-solid fa-money-bill-trend-up"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">Zone Management</span>
            </Link>

            <Link to='/history'
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px]"
              onClick={() => setActive('history')}
            >
              <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"><i className="fa-solid fa-clock-rotate-left"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">History</span>
            </Link>

            <Link to='/notifications'
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px]"
              onClick={() => setActive('notification')}
            >
              <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"><i className="fa-regular fa-bell"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">Notifications</span>
            </Link>

            <Link to='/alerts'
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px]"
              onClick={() => setActive('alert')}
            >
              <span className="text-amber-400 group-hover:text-amber-300 transition-colors duration-300"><i className="fa-solid fa-exclamation-triangle"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">Alerts</span>
            </Link>

            <Link to="/profile"
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px]"
              onClick={() => setActive('profile')}
            >
              <span className="text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"><i className="fa-regular fa-user"></i></span>
              <span className="group-hover:text-cyan-100 transition-colors duration-300">Profile</span>
            </Link>

            <Link
              to="/settings"
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-slate-700 hover:to-slate-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 md:text-[12px] lg:text-[14px] xl:text-[16px]"
              onClick={handleDropdownSettings}
            >
              <article className="flex items-center gap-3 group">
                <i className="fas fa-gear text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i>
                <span className="group-hover:text-cyan-100 transition-colors duration-300">Settings</span>
                <span><i className="fa-solid fa-caret-down text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
              </article>
              {isSettingsOpen && (
                <article className="w-full h-[180px] bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl flex justify-around flex-col p-4 mt-3 transition-all duration-300 ease-out shadow-lg border border-slate-600/30">
                  <h3 className="group flex items-center gap-3 p-3 rounded-lg hover:bg-slate-600/50 transition-all duration-300">
                    <span><i className="fa-solid fa-camera text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">System Settings</span>
                  </h3>
                  <h3 className="group flex items-center gap-3 p-3 rounded-lg hover:bg-slate-600/50 transition-all duration-300">
                    <span><i className="fa-solid fa-video text-cyan-400 group-hover:text-cyan-300 transition-colors duration-300"></i></span>
                    <span className="text-slate-300 group-hover:text-white transition-colors duration-300 font-medium">Device Management</span>
                  </h3>
                </article>
              )}
            </Link>

            <Link to='/login'
              className="text-white px-4 py-3 rounded-xl text-sm font-semibold hover:bg-gradient-to-r hover:from-cyan-600 hover:to-blue-600 cursor-pointer focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300 xl:text-[16px] flex items-center gap-3 group md:text-[12px] xl:text-[16px] lg:text-[14px] bg-gradient-to-r from-cyan-600 to-blue-600 w-40 mt-5 shadow-lg hover:shadow-xl transform hover:scale-105"
              onClick={() => setActive('logout')}
            >
              <span className="group-hover:text-white transition-colors duration-300 flex items-center gap-2">
                <i className="fa-solid fa-sign-out-alt"></i>
                Logout
              </span>
            </Link>
          </div>
        )}
      </nav>
    </header>
  );
}
