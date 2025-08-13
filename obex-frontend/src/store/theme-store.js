import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useThemeStore = create(
  persist(
    (set, get) => ({
      darkMode: true,
      toggleDarkMode: () => {
        const currentMode = get().darkMode;
        set({ darkMode: !currentMode });
        
        // Apply theme to document root
        const root = document.documentElement;
        if (!currentMode) {
          // Switching to dark mode
          root.classList.add('dark');
          root.classList.remove('light');
        } else {
          // Switching to light mode
          root.classList.add('light');
          root.classList.remove('dark');
        }
      },
      setDarkMode: (isDark) => {
        set({ darkMode: isDark });
        
        // Apply theme to document root
        const root = document.documentElement;
        if (isDark) {
          root.classList.add('dark');
          root.classList.remove('light');
        } else {
          root.classList.add('light');
          root.classList.remove('dark');
        }
      },
      initializeTheme: () => {
        const { darkMode } = get();
        const root = document.documentElement;
        if (darkMode) {
          root.classList.add('dark');
          root.classList.remove('light');
        } else {
          root.classList.add('light');
          root.classList.remove('dark');
        }
      }
    }),
    {
      name: 'theme-storage',
    }
  )
);

export default useThemeStore;
