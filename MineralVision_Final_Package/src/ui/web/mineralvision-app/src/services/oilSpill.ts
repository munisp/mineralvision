import api from './api';

export type ReviewStatus = 'pending_review' | 'confirmed' | 'false_positive' | 'needs_resurvey';
export type Severity = 'unknown' | 'low' | 'medium' | 'high' | 'critical';

export interface OilSpillIncident {
  incident_id: string;
  source: string;
  model_id: string;
  model_version: string;
  review_status: ReviewStatus;
  severity: Severity;
  oil_pixel_count: number;
  oil_fraction: number;
  oil_area_m2: number | null;
  oil_area_hectares: number | null;
  confidence: number | null;
  quality_flags: string[];
  geometry_geojson: Record<string, unknown> | null;
  mask_dimensions: [number, number];
  observed_at: string | null;
  created_at: string;
}

export interface OperationsSummary {
  total_incidents: number;
  pending_review: number;
  confirmed: number;
  needs_resurvey: number;
  high_or_critical: number;
  approved_models: number;
  candidate_models: number;
}

export interface CoveragePlan {
  incident_id: string;
  recommended_search_area_m2: number | null;
  priority_cells: Array<Record<string, unknown>>;
  notes: string[];
}

export interface OfflineMaskEvidence {
  id: string;
  createdAt: string;
  payload: Record<string, unknown>;
}

const OFFLINE_QUEUE_KEY = 'mineralvision:oil-spill:mask-evidence-queue';

function readQueue(): OfflineMaskEvidence[] {
  try {
    return JSON.parse(localStorage.getItem(OFFLINE_QUEUE_KEY) || '[]') as OfflineMaskEvidence[];
  } catch {
    return [];
  }
}

function writeQueue(queue: OfflineMaskEvidence[]): void {
  localStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(queue));
}

export const oilSpillApi = {
  summary: () => api.get<OperationsSummary>('/api/oil-spill/operations/summary'),
  listIncidents: (reviewStatus?: ReviewStatus) =>
    api.get<OilSpillIncident[]>('/api/oil-spill/incidents', { params: reviewStatus ? { review_status: reviewStatus } : {} }),
  getIncident: (incidentId: string) => api.get<OilSpillIncident>(`/api/oil-spill/incidents/${incidentId}`),
  review: (incidentId: string, data: { status: ReviewStatus; reviewer: string; note?: string }) =>
    api.patch<OilSpillIncident>(`/api/oil-spill/incidents/${incidentId}/review`, data),
  coveragePlan: (incidentId: string, data: { cell_size_m: number; drone_count: number; buffer_m: number }) =>
    api.post<CoveragePlan>(`/api/oil-spill/incidents/${incidentId}/coverage-plan`, data),
  analyzeMask: (payload: Record<string, unknown>) => api.post<OilSpillIncident>('/api/oil-spill/analyze/mask', payload),
  exportGeoJson: (incidentId: string) => api.get(`/api/oil-spill/incidents/${incidentId}/export.geojson`),

  queueMaskEvidence: (payload: Record<string, unknown>): OfflineMaskEvidence[] => {
    const item: OfflineMaskEvidence = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      payload,
    };
    const queue = [...readQueue(), item];
    writeQueue(queue);
    return queue;
  },
  getOfflineQueue: readQueue,
  async syncOfflineQueue(): Promise<{ synced: number; remaining: number }> {
    const queue = readQueue();
    let synced = 0;
    const pending: OfflineMaskEvidence[] = [];
    for (const item of queue) {
      try {
        await this.analyzeMask(item.payload);
        synced += 1;
      } catch {
        pending.push(item);
      }
    }
    writeQueue(pending);
    return { synced, remaining: pending.length };
  },
};
