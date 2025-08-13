import { create } from "zustand";
import { persist } from "zustand/middleware";

export const useCameraStore = create(
  persist(
    (set) => ({
      CameraStreams: [],
      addToCameraStreams: (newCameraStream) =>
        set((state) => ({
          CameraStreams: [...state.CameraStreams, newCameraStream],
        })),
      removeFromCameraStreams: (idToDelete) =>
        set((state) => ({
          CameraStreams: state.CameraStreams.filter(
            (item) => item.id !== idToDelete
          ),
        })),
      clearCameraStreams: () => set({ CameraStreams: [] }),
    }),
    {
      name: "camera-storage", // 🔐 Key used in localStorage
    }
  )
);
