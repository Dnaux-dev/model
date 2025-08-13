import { create } from 'zustand';

const useAppStore = create((set) => ({
  visitedDashboard: false,
  setVisitedDashboard: (value) => {
    set({ visitedDashboard: value });
    localStorage.setItem("visitedDashboard", JSON.stringify(value));
  },
  restoreState: () => {
    const saved = JSON.parse(localStorage.getItem("visitedDashboard"));
    if (saved) set({ visitedDashboard: saved });
  },
}));

export default useAppStore;
