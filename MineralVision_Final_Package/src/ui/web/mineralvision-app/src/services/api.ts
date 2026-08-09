import axios, { AxiosInstance, AxiosError } from 'axios';
import { useAuthStore } from '../store/authStore';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
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
  status: string;
  createdAt: string;
  updatedAt: string;
}

export interface Drillhole {
  id: string;
  holeId: string;
  projectId: string;
  collar: {
    x: number;
    y: number;
    z: number;
  };
  totalDepth: number;
  status: string;
  assayCount: number;
}

export interface QAQCResult {
  id: string;
  type: string;
  status: string;
  value: number;
  expectedValue: number;
  deviation: number;
  timestamp: string;
}

export interface BlockModel {
  id: string;
  name: string;
  projectId: string;
  cellCount: number;
  tonnage: number;
  grade: number;
  classification: string;
}

export const projectsApi = {
  list: () => api.get<Project[]>('/api/projects'),
  get: (id: string) => api.get<Project>(`/api/projects/${id}`),
  create: (data: Partial<Project>) => api.post<Project>('/api/projects', data),
  update: (id: string, data: Partial<Project>) => api.put<Project>(`/api/projects/${id}`, data),
  delete: (id: string) => api.delete(`/api/projects/${id}`),
};

export const drillholesApi = {
  list: (projectId?: string) => api.get<Drillhole[]>('/api/drillholes', { params: { projectId } }),
  get: (id: string) => api.get<Drillhole>(`/api/drillholes/${id}`),
  create: (data: Partial<Drillhole>) => api.post<Drillhole>('/api/drillholes', data),
  upload: (file: File, projectId: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('projectId', projectId);
    return api.post('/api/drillholes/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  composite: (id: string, params: { length: number; method: string }) =>
    api.post(`/api/drillholes/${id}/composite`, params),
  desurvey: (id: string, method: string) =>
    api.post(`/api/drillholes/${id}/desurvey`, { method }),
};

export const qaqcApi = {
  list: (projectId?: string) => api.get<QAQCResult[]>('/api/qaqc', { params: { projectId } }),
  analyze: (projectId: string, type: string) =>
    api.post('/api/qaqc/analyze', { projectId, type }),
  getControlChart: (projectId: string, standardId: string) =>
    api.get(`/api/qaqc/control-chart/${projectId}/${standardId}`),
};

export const geostatisticsApi = {
  calculateVariogram: (params: {
    projectId: string;
    variable: string;
    lagDistance: number;
    numLags: number;
    directions: number[];
  }) => api.post('/api/geostatistics/variogram', params),
  
  fitVariogramModel: (params: {
    experimentalVariogram: any;
    modelType: string;
  }) => api.post('/api/geostatistics/variogram/fit', params),
  
  runKriging: (params: {
    projectId: string;
    variogramModel: any;
    krigingType: string;
    searchRadius: number;
    minSamples: number;
    maxSamples: number;
  }) => api.post('/api/geostatistics/kriging', params),
  
  createBlockModel: (params: {
    projectId: string;
    origin: { x: number; y: number; z: number };
    cellSize: { x: number; y: number; z: number };
    dimensions: { nx: number; ny: number; nz: number };
  }) => api.post('/api/geostatistics/block-model', params),
  
  classifyResources: (blockModelId: string, params: {
    cutoffGrade: number;
    varianceThresholds: { measured: number; indicated: number };
  }) => api.post(`/api/geostatistics/block-model/${blockModelId}/classify`, params),
};

export const visualizationApi = {
  getDrillholeScene: (projectId: string) =>
    api.get(`/api/visualization/drillholes/${projectId}`),
  getBlockModelScene: (blockModelId: string) =>
    api.get(`/api/visualization/block-model/${blockModelId}`),
  getSurfaceScene: (surfaceId: string) =>
    api.get(`/api/visualization/surface/${surfaceId}`),
  exportImage: (sceneId: string, format: string) =>
    api.post(`/api/visualization/export/${sceneId}`, { format }, { responseType: 'blob' }),
};

export const inversionApi = {
  createMesh: (params: {
    origin: { x: number; y: number; z: number };
    cellSize: { x: number; y: number; z: number };
    dimensions: { nx: number; ny: number; nz: number };
  }) => api.post('/api/inversion/mesh', params),
  
  uploadSurvey: (file: File, type: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);
    return api.post('/api/inversion/survey', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  runInversion: (params: {
    meshId: string;
    surveyId: string;
    inversionType: string;
    maxIterations: number;
    targetMisfit: number;
  }) => api.post('/api/inversion/run', params),
  
  getResult: (inversionId: string) =>
    api.get(`/api/inversion/result/${inversionId}`),
};

export const reportsApi = {
  list: (projectId?: string) => api.get('/api/reports', { params: { projectId } }),
  
  generate: (params: {
    projectId: string;
    standard: string;
    qualifiedPerson: any;
    resourceStatement: any;
  }) => api.post('/api/reports/generate', params),
  
  download: (reportId: string, format: string) =>
    api.get(`/api/reports/${reportId}/download`, {
      params: { format },
      responseType: 'blob',
    }),
};

export const sensorFusionApi = {
  uploadData: (file: File, sensorType: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('sensorType', sensorType);
    return api.post('/api/sensor-fusion/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  fuse: (params: {
    dataIds: string[];
    algorithm: string;
    parameters: Record<string, any>;
  }) => api.post('/api/sensor-fusion/fuse', params),
  
  getResults: () => api.get('/api/sensor-fusion/results'),
};

export const usersApi = {
  list: () => api.get('/api/users'),
  get: (id: string) => api.get(`/api/users/${id}`),
  create: (data: any) => api.post('/api/users', data),
  update: (id: string, data: any) => api.put(`/api/users/${id}`, data),
  delete: (id: string) => api.delete(`/api/users/${id}`),
  updateRoles: (id: string, roles: string[]) =>
    api.put(`/api/users/${id}/roles`, { roles }),
};

export default api;
