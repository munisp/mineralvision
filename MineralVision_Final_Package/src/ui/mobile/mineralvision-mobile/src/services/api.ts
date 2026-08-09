import axios, { AxiosInstance, AxiosError } from 'axios';
import * as SecureStore from 'expo-secure-store';

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  async (config) => {
    const token = await SecureStore.getItemAsync('mineralvision_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    if (error.response?.status === 401) {
      await SecureStore.deleteItemAsync('mineralvision_token');
      await SecureStore.deleteItemAsync('mineralvision_user');
    }
    return Promise.reject(error);
  }
);

export interface Project {
  id: string;
  name: string;
  description: string;
  location: string;
  commodities: string[];
  status: 'active' | 'completed' | 'on-hold';
  createdAt: string;
  updatedAt: string;
}

export interface Drillhole {
  id: string;
  holeId: string;
  projectId: string;
  collar: { x: number; y: number; z: number };
  totalDepth: number;
  azimuth: number;
  dip: number;
  status: 'completed' | 'in-progress' | 'planned';
}

export interface Sample {
  id: string;
  sampleId: string;
  drillholeId: string;
  fromDepth: number;
  toDepth: number;
  sampleType: string;
  assays: Record<string, number>;
}

export const projectsApi = {
  getAll: () => api.get<Project[]>('/api/projects'),
  getById: (id: string) => api.get<Project>(`/api/projects/${id}`),
  create: (data: Partial<Project>) => api.post<Project>('/api/projects', data),
  update: (id: string, data: Partial<Project>) => api.put<Project>(`/api/projects/${id}`, data),
  delete: (id: string) => api.delete(`/api/projects/${id}`),
};

export const drillholesApi = {
  getAll: (projectId?: string) => api.get<Drillhole[]>('/api/drillholes', { params: { projectId } }),
  getById: (id: string) => api.get<Drillhole>(`/api/drillholes/${id}`),
  create: (data: Partial<Drillhole>) => api.post<Drillhole>('/api/drillholes', data),
  update: (id: string, data: Partial<Drillhole>) => api.put<Drillhole>(`/api/drillholes/${id}`, data),
  delete: (id: string) => api.delete(`/api/drillholes/${id}`),
};

export const samplesApi = {
  getAll: (drillholeId?: string) => api.get<Sample[]>('/api/samples', { params: { drillholeId } }),
  getById: (id: string) => api.get<Sample>(`/api/samples/${id}`),
  create: (data: Partial<Sample>) => api.post<Sample>('/api/samples', data),
  update: (id: string, data: Partial<Sample>) => api.put<Sample>(`/api/samples/${id}`, data),
  delete: (id: string) => api.delete(`/api/samples/${id}`),
};

export const uploadApi = {
  uploadFile: async (uri: string, type: string, projectId: string) => {
    const formData = new FormData();
    const filename = uri.split('/').pop() || 'file';
    
    formData.append('file', {
      uri,
      name: filename,
      type: type || 'application/octet-stream',
    } as any);
    formData.append('projectId', projectId);
    
    return api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

export default api;
