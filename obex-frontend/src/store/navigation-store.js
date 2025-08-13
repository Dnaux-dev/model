import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useNavStore = create(
  persist(
    (set) => ({
      active: 'dashboard', // default
      setActive: (id) => set({ active: id }),
    }),
    {
      name: 'active-nav', // key in localStorage
    }
  )
);
