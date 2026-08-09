import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  roles: string[];
  tenantId?: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  register: (data: RegisterData) => Promise<boolean>;
  clearError: () => void;
  hasPermission: (permission: string) => boolean;
  hasRole: (role: string) => boolean;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  firstName: string;
  lastName: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

            login: async (username: string, password: string) => {
              set({ isLoading: true, error: null });

              // Try the real backend first. Demo mode is only a fallback for
              // 'admin'/'demo' when the backend is UNREACHABLE (network error),
              // so that UI demos without a server still work. Credentials are
              // never short-circuited while a backend is available.
              let backendUnreachable = false;
              try {
                const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ username, password }),
                });

                if (!response.ok) {
                  const errorData = await response.json();
                  throw new Error(errorData.detail || 'Login failed');
                }

                const data = await response.json();

                set({
                  user: data.user,
                  token: data.session?.token ?? data.access_token ?? data.accessToken,
                  isAuthenticated: true,
                  isLoading: false,
                  error: null,
                });

                return true;
              } catch (error) {
                backendUnreachable = error instanceof TypeError; // fetch network failure
                if (!backendUnreachable || (username !== 'admin' && username !== 'demo')) {
                  set({
                    isLoading: false,
                    error: error instanceof Error ? error.message : 'Login failed',
                  });
                  return false;
                }
              }

              // Offline demo fallback (backend unreachable + demo username).
              set({
                user: {
                  id: '1',
                  username: username,
                  email: `${username}@mineralvision.ai`,
                  firstName: username === 'admin' ? 'Admin' : 'Demo',
                  lastName: 'User',
                  roles: ['admin', 'resource_geologist'],
                  tenantId: 'demo-tenant',
                },
                token: 'demo-token-' + Date.now(),
                isAuthenticated: true,
                isLoading: false,
                error: null,
              });
              return true;
            },

      logout: () => {
        const { token } = get();
        
        if (token) {
          fetch(`${API_BASE_URL}/api/auth/logout`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          }).catch(() => {});
        }
        
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
      },

      register: async (data: RegisterData) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Registration failed');
          }

          set({ isLoading: false });
          return true;
        } catch (error) {
          set({
            isLoading: false,
            error: error instanceof Error ? error.message : 'Registration failed',
          });
          return false;
        }
      },

      clearError: () => set({ error: null }),

      hasPermission: (permission: string) => {
        const { user } = get();
        if (!user) return false;
        
        if (user.roles.includes('admin')) return true;
        
        const rolePermissions: Record<string, string[]> = {
          admin: ['read', 'write', 'delete', 'admin', 'approve', 'export'],
          resource_geologist: ['read', 'write', 'execute', 'export'],
          geologist: ['read', 'write', 'execute'],
          viewer: ['read'],
        };
        
        for (const role of user.roles) {
          if (rolePermissions[role]?.includes(permission)) {
            return true;
          }
        }
        
        return false;
      },

      hasRole: (role: string) => {
        const { user } = get();
        return user?.roles.includes(role) || false;
      },
    }),
    {
      name: 'mineralvision-auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
