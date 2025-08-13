import { create } from "zustand";
import { persist } from "zustand/middleware";

export const useNotificationStore = create(
  persist(
    (set, get) => ({
      notifications: [],
      unreadCount: 0,

      // Add a new threat notification
      addNotification: (notification) => {
        const newNotification = {
          id: Date.now() + Math.random(),
          timestamp: new Date().toISOString(),
          read: false,
          ...notification,
        };
        
        set((state) => ({
          notifications: [newNotification, ...state.notifications],
          unreadCount: state.unreadCount + 1,
        }));
      },

      // Mark notification as read
      markAsRead: (id) => {
        set((state) => ({
          notifications: state.notifications.map((notification) =>
            notification.id === id ? { ...notification, read: true } : notification
          ),
          unreadCount: Math.max(0, state.unreadCount - 1),
        }));
      },

      // Mark all notifications as read
      markAllAsRead: () => {
        set((state) => ({
          notifications: state.notifications.map((notification) => ({
            ...notification,
            read: true,
          })),
          unreadCount: 0,
        }));
      },

      // Delete a notification
      deleteNotification: (id) => {
        set((state) => {
          const notification = state.notifications.find(n => n.id === id);
          return {
            notifications: state.notifications.filter((n) => n.id !== id),
            unreadCount: notification && !notification.read 
              ? Math.max(0, state.unreadCount - 1) 
              : state.unreadCount,
          };
        });
      },

      // Clear all notifications
      clearAllNotifications: () => {
        set({ notifications: [], unreadCount: 0 });
      },

      // Get notifications by type
      getNotificationsByType: (type) => {
        const state = get();
        return state.notifications.filter((notification) => notification.type === type);
      },

      // Get unread notifications
      getUnreadNotifications: () => {
        const state = get();
        return state.notifications.filter((notification) => !notification.read);
      },

      // Get recent notifications (last 24 hours)
      getRecentNotifications: () => {
        const state = get();
        const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        return state.notifications.filter((notification) => 
          new Date(notification.timestamp) > twentyFourHoursAgo
        );
      },
    }),
    {
      name: "notification-storage",
    }
  )
);
