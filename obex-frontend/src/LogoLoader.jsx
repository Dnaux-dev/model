import React from 'react'
import useLoadingStore from './store/loading-store'
import primusLogo from './obex-logo.png'

export default function LogoLoader() {
    const loading = useLoadingStore((state) => state.loading);

    if (!loading) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            {/* Animated background elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-cyan-400/20 to-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-blue-400/20 to-cyan-500/20 rounded-full blur-3xl animate-pulse animation-delay-1000"></div>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-cyan-400/10 to-blue-500/10 rounded-full blur-3xl animate-ping"></div>
            </div>
            
            {/* Main logo container */}
            <div className="relative z-10">
                {/* Glowing effect behind logo */}
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full blur-2xl opacity-30 animate-pulse"></div>
                
                {/* Logo with enhanced animations */}
                <div className="relative">
                    <img 
                        src={primusLogo} 
                        alt="Loading..." 
                        className="w-40 h-20 xl:w-60 xl:h-30 animate-bounce drop-shadow-2xl filter brightness-110"
                    />
                    
                    {/* Ripple effect */}
                    <div className="absolute inset-0 rounded-full border-2 border-cyan-400/50 animate-ping"></div>
                    <div className="absolute inset-0 rounded-full border-2 border-blue-500/50 animate-ping animation-delay-500"></div>
                </div>
                
                {/* Loading text */}
                <div className="mt-8 text-center">
                    <h2 className="text-2xl xl:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 animate-pulse">
                        OBEX
                    </h2>
                    <p className="text-slate-400 text-sm xl:text-base mt-2 font-medium animate-pulse animation-delay-700">
                        Loading Security System...
                    </p>
                </div>
                
                {/* Loading dots */}
                <div className="flex justify-center mt-6 space-x-2">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce animation-delay-200"></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce animation-delay-400"></div>
                </div>
            </div>
            
            {/* Floating particles */}
            <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-20 left-20 w-1 h-1 bg-cyan-400 rounded-full animate-ping opacity-60"></div>
                <div className="absolute top-40 right-32 w-1 h-1 bg-blue-500 rounded-full animate-ping animation-delay-300 opacity-60"></div>
                <div className="absolute bottom-32 left-32 w-1 h-1 bg-cyan-400 rounded-full animate-ping animation-delay-600 opacity-60"></div>
                <div className="absolute bottom-20 right-20 w-1 h-1 bg-blue-500 rounded-full animate-ping animation-delay-900 opacity-60"></div>
            </div>
        </div>
    )
}
