import React, { useState } from "react";
import axios from "axios";
import { Mail, Lock, LogIn, CheckCircle, XCircle, ArrowRight } from 'lucide-react';
import { useNavigate, Link } from 'react-router-dom';

const Login = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const [message, setMessage] = useState(null);
  const [messageType, setMessageType] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    setMessageType(null);

    try {
      const res = await axios.post("https://primus-lite.onrender.com/api/users/login", formData);
      setMessage(res.data.message || "Login successful!");
      setMessageType('success');

      localStorage.setItem('primusLiteToken', res.data.token);
      navigate('/dashboard');

    } catch (err) {
      setMessage(err.response?.data?.message || "Login failed. Please check your credentials.");
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex items-center justify-center p-4 font-sans relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-cyan-400/20 to-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-blue-400/20 to-cyan-500/20 rounded-full blur-3xl animate-pulse animation-delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-cyan-400/10 to-blue-500/10 rounded-full blur-3xl animate-ping"></div>
      </div>

      {/* Floating particles */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-20 w-1 h-1 bg-cyan-400 rounded-full animate-ping opacity-60"></div>
        <div className="absolute top-40 right-32 w-1 h-1 bg-blue-500 rounded-full animate-ping animation-delay-300 opacity-60"></div>
        <div className="absolute bottom-32 left-32 w-1 h-1 bg-cyan-400 rounded-full animate-ping animation-delay-600 opacity-60"></div>
        <div className="absolute bottom-20 right-20 w-1 h-1 bg-blue-500 rounded-full animate-ping animation-delay-900 opacity-60"></div>
      </div>

      {/* Main form container */}
      <div className="relative z-10 w-full max-w-md">
        {/* Glowing effect behind form */}
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/20 to-blue-500/20 rounded-3xl blur-2xl opacity-30 animate-pulse"></div>
        
        <form
          onSubmit={handleSubmit}
          className="relative bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm p-8 md:p-10 rounded-3xl border border-white/20 shadow-2xl shadow-cyan-500/20 space-y-6 transform transition-all duration-300 hover:scale-[1.02]"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-2 tracking-wide">
              Welcome Back
            </h2>
            <p className="text-slate-300 text-sm">Sign in to your OBEX account</p>
          </div>

          {/* Email Input */}
          <div className="relative group">
            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-cyan-400 group-focus-within:text-cyan-300 transition-colors duration-200" size={20} />
            <input
              type="email"
              name="email"
              placeholder="Email"
              className="w-full pl-10 pr-3 py-4 rounded-xl bg-white/10 text-white placeholder-slate-300 border border-white/20 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:border-cyan-400/50 transition-all duration-200 backdrop-blur-sm"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>

          {/* Password Input */}
          <div className="relative group">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-cyan-400 group-focus-within:text-cyan-300 transition-colors duration-200" size={20} />
            <input
              type="password"
              name="password"
              placeholder="Password"
              className="w-full pl-10 pr-3 py-4 rounded-xl bg-white/10 text-white placeholder-slate-300 border border-white/20 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:border-cyan-400/50 transition-all duration-200 backdrop-blur-sm"
              value={formData.password}
              onChange={handleChange}
              required
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-lg rounded-xl shadow-lg shadow-cyan-500/40 hover:shadow-cyan-500/60 transition-all duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center transform hover:scale-[1.02] focus:outline-none focus:ring-4 focus:ring-cyan-400/30"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 mr-3 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Signing in...
              </>
            ) : (
              <>
                <LogIn size={20} className="mr-2" /> Sign In
                <ArrowRight size={20} className="ml-2 transition-transform group-hover:translate-x-1" />
              </>
            )}
          </button>

          {/* Message Display */}
          {message && (
            <div className={`p-4 rounded-xl border ${
              messageType === 'success' 
                ? 'bg-green-500/20 border-green-500/30 text-green-300' 
                : 'bg-red-500/20 border-red-500/30 text-red-300'
            } flex items-center justify-center`}>
              {messageType === 'success' ? <CheckCircle size={16} className="mr-2" /> : <XCircle size={16} className="mr-2" />}
              {message}
            </div>
          )}

          {/* Signup Link */}
          <div className="text-center pt-4">
            <span className="text-slate-300">Don't have an account? </span>
            <Link
              to="/signup"
              className="text-cyan-400 font-semibold hover:text-cyan-300 transition duration-200 hover:underline"
            >
              Sign up here
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
