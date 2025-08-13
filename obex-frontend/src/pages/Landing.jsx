import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Camera, Bell, ArrowRight, Play, Star, Users, MapPin, Zap } from 'lucide-react';

const LandingPage = () => {
  const navigate = useNavigate();
  return (
    <div className="relative min-h-screen w-full text-white font-sans overflow-hidden">
      {/* Video Background - Fixed to cover properly */}
      <div className="fixed inset-0 w-full h-full z-0">
        <video
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        >
          <source src="/background.mp4" type="video/mp4" />
          {/* Fallback background if video fails */}
          <div className="w-full h-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900"></div>
        </video>
      </div>

      {/* Dark overlay for better text readability */}
      <div className="absolute inset-0 bg-black/40 z-10"></div>

      {/* Main Content */}
      <div className="relative z-20 flex flex-col min-h-screen">
        {/* Header/Navigation */}
        <header className="flex justify-between items-center p-6 lg:p-8">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
              OBEX
            </span>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <button className="text-slate-200 hover:text-cyan-400 transition-colors duration-200">
              Features
            </button>
            <button className="text-slate-200 hover:text-cyan-400 transition-colors duration-200">
              About
            </button>
            <button className="text-slate-200 hover:text-cyan-400 transition-colors duration-200">
              Contact
            </button>
          </div>
        </header>

        {/* Hero Section */}
        <main className="flex-1 flex flex-col items-center justify-center text-center px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            {/* Main Heading */}
            <div className="mb-12 lg:mb-16">
              <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-cyan-200 via-cyan-400 to-blue-500 drop-shadow-[0_4px_8px_rgba(0,255,255,0.3)] tracking-tight leading-tight">
                The Zero Crime
                <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">
                  Mission
                </span>
              </h1>
              
              <p className="text-lg sm:text-xl md:text-2xl lg:text-3xl text-slate-200 mb-12 leading-relaxed max-w-4xl mx-auto font-medium">
                Stay one step ahead of danger. OBEX delivers real-time alerts, smart surveillance, 
                and crystal-clear monitoring—all in one powerful, easy-to-use platform.
              </p>
              
              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-6 lg:space-x-8">
                <button
                  className="group px-8 py-4 lg:px-12 lg:py-5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold rounded-2xl shadow-[0_8px_32px_rgba(0,255,255,0.3)] transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-cyan-200/50 text-lg lg:text-xl"
                  onClick={() => navigate('/signup')}
                >
                  Get Started
                  <ArrowRight className="inline-block ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform duration-200" />
                </button>
                
                <button className="px-8 py-4 lg:px-12 lg:py-5 border-2 border-cyan-300 text-cyan-100 hover:bg-cyan-900/40 hover:text-cyan-50 rounded-2xl font-bold shadow-[0_4px_16px_rgba(0,255,255,0.2)] transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-cyan-200/50 text-lg lg:text-xl">
                  <Play className="inline-block mr-2 w-5 h-5" />
                  Live Demo
                </button>
              </div>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-10 mb-20 lg:mb-32">
              <div className="group bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-md p-8 lg:p-10 rounded-3xl border border-white/20 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-500 transform hover:scale-105 hover:border-cyan-400/30">
                <div className="w-20 h-20 bg-gradient-to-r from-red-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <Bell className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Real-time Alerts</h3>
                <p className="text-slate-300 text-base leading-relaxed">
                  Instant notifications for any security threats detected with AI-powered analysis
                </p>
              </div>
              
              <div className="group bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-md p-8 lg:p-10 rounded-3xl border border-white/20 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-500 transform hover:scale-105 hover:border-cyan-400/30">
                <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <Camera className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Smart Surveillance</h3>
                <p className="text-slate-300 text-base leading-relaxed">
                  AI-powered monitoring with crystal-clear video quality and intelligent threat detection
                </p>
              </div>
              
              <div className="group bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-md p-8 lg:p-10 rounded-3xl border border-white/20 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-500 transform hover:scale-105 hover:border-cyan-400/30">
                <div className="w-20 h-20 bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <Shield className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Complete Protection</h3>
                <p className="text-slate-300 text-base leading-relaxed">
                  From your front door to your business, comprehensive security coverage 24/7
                </p>
              </div>
            </div>

            {/* Stats Section */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 lg:gap-12 mb-20 lg:mb-32">
              <div className="text-center group">
                <div className="text-4xl lg:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-3 group-hover:scale-110 transition-transform duration-300">
                  99.9%
                </div>
                <div className="text-slate-300 text-base font-medium">Uptime</div>
              </div>
              <div className="text-center group">
                <div className="text-4xl lg:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-3 group-hover:scale-110 transition-transform duration-300">
                  24/7
                </div>
                <div className="text-slate-300 text-base font-medium">Monitoring</div>
              </div>
              <div className="text-center group">
                <div className="text-4xl lg:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-3 group-hover:scale-110 transition-transform duration-300">
                  10K+
                </div>
                <div className="text-slate-300 text-base font-medium">Users</div>
              </div>
              <div className="text-center group">
                <div className="text-4xl lg:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-3 group-hover:scale-110 transition-transform duration-300">
                  50+
                </div>
                <div className="text-slate-300 text-base font-medium">Countries</div>
              </div>
            </div>

            {/* Additional CTA */}
            <div className="text-center">
              <button
                className="px-10 py-5 bg-gradient-to-r from-slate-700/50 to-slate-800/50 backdrop-blur-md border border-white/20 text-white font-semibold rounded-2xl hover:bg-slate-600/50 hover:border-cyan-400/50 transition-all duration-300 transform hover:scale-105 shadow-xl"
                onClick={() => navigate('/login')}
              >
                Already have an account? Sign In
              </button>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="text-center py-8 text-slate-300 text-sm lg:text-base font-light">
          © {new Date().getFullYear()} OBEX Security Systems. All rights reserved.
        </footer>
      </div>
    </div>
  );
};

export default LandingPage;
