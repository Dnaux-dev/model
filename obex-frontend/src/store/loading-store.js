import { create } from "zustand";

const useLoadingStore = create((set) => ({
    loading: false,
    showLoading: () => set({ loading: true}),
    hideLoading: () => set({ loading: false}),
}));

export default useLoadingStore;