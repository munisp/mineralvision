import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  CloudOff,
  Download,
  MapPinned,
  RefreshCw,
  ScanSearch,
  Send,
  ShieldCheck,
  Signal,
  UploadCloud,
} from 'lucide-react';
import {
  CoveragePlan,
  OilSpillIncident,
  OperationsSummary,
  ReviewStatus,
  Severity,
  oilSpillApi,
} from '../../services/oilSpill';
import { useAuthStore } from '../../store/authStore';

const reviewOptions: Array<{ value: ReviewStatus; label: string }> = [
  { value: 'pending_review', label: 'Pending review' },
  { value: 'confirmed', label: 'Confirmed' },
  { value: 'needs_resurvey', label: 'Needs resurvey' },
  { value: 'false_positive', label: 'False positive' },
];

const severityTone: Record<Severity, string> = {
  unknown: 'bg-slate-500/15 text-slate-700 dark:text-slate-200',
  low: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  medium: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  high: 'bg-orange-500/15 text-orange-700 dark:text-orange-300',
  critical: 'bg-red-500/15 text-red-700 dark:text-red-300',
};

function formatArea(area: number | null): string {
  if (area === null) return 'Georeference required';
  if (area >= 1_000_000) return `${(area / 1_000_000).toFixed(2)} km²`;
  if (area >= 10_000) return `${(area / 10_000).toFixed(2)} ha`;
  return `${area.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²`;
}

function StatCard({ label, value, icon: Icon, accent }: { label: string; value: number; icon: typeof ScanSearch; accent: string }) {
  return (
    <article className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{label}</p>
          <p className="mt-2 text-3xl font-bold tracking-tight text-foreground">{value}</p>
        </div>
        <div className={`rounded-xl p-2.5 ${accent}`}><Icon className="h-5 w-5" /></div>
      </div>
    </article>
  );
}

export default function OilSpillOperationsPage() {
  const { user } = useAuthStore();
  const reviewer = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || user?.email || 'authenticated_operator';
  const [summary, setSummary] = useState<OperationsSummary | null>(null);
  const [incidents, setIncidents] = useState<OilSpillIncident[]>([]);
  const [selected, setSelected] = useState<OilSpillIncident | null>(null);
  const [filter, setFilter] = useState<ReviewStatus | 'all'>('pending_review');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string>('');
  const [coverage, setCoverage] = useState<CoveragePlan | null>(null);
  const [queueCount, setQueueCount] = useState(oilSpillApi.getOfflineQueue().length);
  const [reviewStatus, setReviewStatus] = useState<ReviewStatus>('confirmed');
  const [reviewNote, setReviewNote] = useState('');
  const [captureOpen, setCaptureOpen] = useState(() => new URLSearchParams(window.location.search).get('capture') === '1');
  const [capture, setCapture] = useState({
    mask_base64: '', image_width_px: '512', image_height_px: '512', source: 'drone_rgb',
    model_id: 'field-model', model_version: 'candidate', ground_sampling_distance_m: '',
  });

  const filteredCount = useMemo(() => incidents.length, [incidents]);

  async function loadData(): Promise<void> {
    setLoading(true);
    try {
      const [summaryResponse, incidentsResponse] = await Promise.all([
        oilSpillApi.summary(),
        oilSpillApi.listIncidents(filter === 'all' ? undefined : filter),
      ]);
      setSummary(summaryResponse.data);
      setIncidents(incidentsResponse.data);
      setSelected((current) => incidentsResponse.data.find((incident) => incident.incident_id === current?.incident_id) || incidentsResponse.data[0] || null);
      setMessage('');
    } catch {
      setMessage('Unable to reach the operations service. Saved compact mask evidence can still be queued on this device.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void loadData(); }, [filter]);

  async function submitReview(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!selected) return;
    setBusy(true);
    try {
      const response = await oilSpillApi.review(selected.incident_id, { status: reviewStatus, reviewer, note: reviewNote || undefined });
      setSelected(response.data);
      setReviewNote('');
      setMessage('Review state recorded. No external notification, flight, or cleanup action was initiated.');
      await loadData();
    } catch {
      setMessage('Review could not be saved. Check the secure API connection and try again.');
    } finally {
      setBusy(false);
    }
  }

  async function createCoveragePlan(): Promise<void> {
    if (!selected) return;
    setBusy(true);
    try {
      const response = await oilSpillApi.coveragePlan(selected.incident_id, { cell_size_m: 50, drone_count: 2, buffer_m: 100 });
      setCoverage(response.data);
      setMessage('Advisory coverage grid prepared. Obtain operational authorization before any flight activity.');
    } catch {
      setMessage('A coverage grid needs a valid georeferenced incident footprint.');
    } finally {
      setBusy(false);
    }
  }

  async function downloadGeoJson(): Promise<void> {
    if (!selected) return;
    try {
      const response = await oilSpillApi.exportGeoJson(selected.incident_id);
      const file = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/geo+json' });
      const url = URL.createObjectURL(file);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `oil-spill-${selected.incident_id}.geojson`;
      anchor.click();
      URL.revokeObjectURL(url);
      setMessage('GeoJSON evidence exported to this device.');
    } catch {
      setMessage('GeoJSON export requires a georeferenced incident footprint.');
    }
  }

  async function submitCapture(event: FormEvent): Promise<void> {
    event.preventDefault();
    const payload = {
      mask_base64: capture.mask_base64,
      image_width_px: Number(capture.image_width_px),
      image_height_px: Number(capture.image_height_px),
      source: capture.source,
      model_id: capture.model_id,
      model_version: capture.model_version,
      ground_sampling_distance_m: capture.ground_sampling_distance_m ? Number(capture.ground_sampling_distance_m) : undefined,
      metadata: { capture_mode: 'pwa_field_workspace', captured_by: reviewer },
    };
    if (!navigator.onLine) {
      setQueueCount(oilSpillApi.queueMaskEvidence(payload).length);
      setMessage('Offline: compact mask evidence is encrypted by neither the browser nor this app—use a managed device and sync through the secure API when online.');
      setCaptureOpen(false);
      return;
    }
    setBusy(true);
    try {
      const response = await oilSpillApi.analyzeMask(payload);
      setSelected(response.data);
      setCaptureOpen(false);
      setMessage('New evidence assessed and queued for human review.');
      await loadData();
    } catch {
      setQueueCount(oilSpillApi.queueMaskEvidence(payload).length);
      setMessage('Submission failed, so compact mask evidence was queued locally. Review device controls before synchronizing.');
    } finally {
      setBusy(false);
    }
  }

  async function syncQueue(): Promise<void> {
    if (!navigator.onLine) {
      setMessage('This device is offline. Connect to the secured API before synchronizing queued evidence.');
      return;
    }
    setBusy(true);
    const result = await oilSpillApi.syncOfflineQueue();
    setQueueCount(result.remaining);
    setMessage(`${result.synced} queued item(s) synchronized; ${result.remaining} remain on this device.`);
    setBusy(false);
    await loadData();
  }

  return (
    <section className="mx-auto max-w-7xl space-y-6 pb-8">
      <div className="flex flex-col gap-4 rounded-3xl bg-slate-950 p-6 text-white shadow-xl lg:flex-row lg:items-center lg:justify-between">
        <div className="max-w-2xl">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-cyan-200"><ShieldCheck className="h-4 w-4" /> Response decision support</div>
          <h1 className="text-3xl font-bold tracking-tight">Oil-spill operations workspace</h1>
          <p className="mt-2 text-sm leading-6 text-slate-300">Triage versioned evidence, preserve review accountability, prepare an advisory coverage grid, and export GIS-ready footprints. The workspace never triggers response actions automatically.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button onClick={() => setCaptureOpen(!captureOpen)} className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300"><UploadCloud className="h-4 w-4" /> Capture evidence</button>
          <button onClick={() => void syncQueue()} disabled={busy} className="inline-flex items-center gap-2 rounded-xl border border-slate-600 px-4 py-2.5 text-sm font-semibold hover:bg-slate-800"><CloudOff className="h-4 w-4" /> Sync queue ({queueCount})</button>
        </div>
      </div>

      {message && <div role="status" className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-4 py-3 text-sm text-foreground">{message}</div>}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Pending review" value={summary?.pending_review || 0} icon={ScanSearch} accent="bg-amber-400/15 text-amber-600" />
        <StatCard label="High / critical" value={summary?.high_or_critical || 0} icon={AlertTriangle} accent="bg-red-500/15 text-red-600" />
        <StatCard label="Confirmed" value={summary?.confirmed || 0} icon={CheckCircle2} accent="bg-emerald-500/15 text-emerald-600" />
        <StatCard label="Approved models" value={summary?.approved_models || 0} icon={ShieldCheck} accent="bg-cyan-500/15 text-cyan-600" />
      </div>

      {captureOpen && (
        <form onSubmit={(event) => void submitCapture(event)} className="rounded-2xl border border-border bg-card p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2"><UploadCloud className="h-5 w-5 text-primary" /><h2 className="font-semibold text-foreground">Field evidence capture</h2></div>
          <p className="mb-4 text-sm text-muted-foreground">Paste a compact PNG probability-mask data URL or Base64 payload generated by a trusted model. Raw aerial imagery is intentionally handled by the controlled model-upload endpoint, not stored offline in this PWA queue.</p>
          <div className="grid gap-3 md:grid-cols-3">
            <label className="text-sm font-medium">Width (px)<input required value={capture.image_width_px} onChange={(event) => setCapture({ ...capture, image_width_px: event.target.value })} inputMode="numeric" className="mt-1 w-full rounded-lg border border-input bg-background p-2" /></label>
            <label className="text-sm font-medium">Height (px)<input required value={capture.image_height_px} onChange={(event) => setCapture({ ...capture, image_height_px: event.target.value })} inputMode="numeric" className="mt-1 w-full rounded-lg border border-input bg-background p-2" /></label>
            <label className="text-sm font-medium">Ground sample distance (m)<input value={capture.ground_sampling_distance_m} onChange={(event) => setCapture({ ...capture, ground_sampling_distance_m: event.target.value })} inputMode="decimal" className="mt-1 w-full rounded-lg border border-input bg-background p-2" /></label>
            <label className="text-sm font-medium">Source<select value={capture.source} onChange={(event) => setCapture({ ...capture, source: event.target.value })} className="mt-1 w-full rounded-lg border border-input bg-background p-2"><option value="drone_rgb">Drone RGB</option><option value="satellite_rgb">Satellite RGB</option><option value="fluorosensor">Fluorosensor</option><option value="manual_annotation">Manual annotation</option></select></label>
            <label className="text-sm font-medium">Model ID<input required value={capture.model_id} onChange={(event) => setCapture({ ...capture, model_id: event.target.value })} className="mt-1 w-full rounded-lg border border-input bg-background p-2" /></label>
            <label className="text-sm font-medium">Model version<input required value={capture.model_version} onChange={(event) => setCapture({ ...capture, model_version: event.target.value })} className="mt-1 w-full rounded-lg border border-input bg-background p-2" /></label>
          </div>
          <label className="mt-3 block text-sm font-medium">Probability mask evidence<textarea required value={capture.mask_base64} onChange={(event) => setCapture({ ...capture, mask_base64: event.target.value })} rows={4} className="mt-1 w-full rounded-lg border border-input bg-background p-2 font-mono text-xs" placeholder="data:image/png;base64,..." /></label>
          <div className="mt-4 flex gap-2"><button disabled={busy} className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground"><Send className="h-4 w-4" /> {navigator.onLine ? 'Submit for review' : 'Queue evidence'}</button><button type="button" onClick={() => setCaptureOpen(false)} className="rounded-lg border border-input px-4 py-2 text-sm font-semibold">Cancel</button></div>
        </form>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.9fr)]">
        <article className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
          <div className="flex flex-col gap-3 border-b border-border p-4 sm:flex-row sm:items-center sm:justify-between">
            <div><h2 className="font-semibold text-foreground">Incident triage</h2><p className="text-sm text-muted-foreground">{filteredCount} visible record(s)</p></div>
            <div className="flex gap-2"><select value={filter} onChange={(event) => setFilter(event.target.value as ReviewStatus | 'all')} className="rounded-lg border border-input bg-background px-3 py-2 text-sm"><option value="all">All review states</option>{reviewOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select><button onClick={() => void loadData()} aria-label="Refresh incidents" className="rounded-lg border border-input p-2 hover:bg-muted"><RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /></button></div>
          </div>
          <div className="max-h-[580px] overflow-auto">
            {loading ? <p className="p-6 text-sm text-muted-foreground">Loading operational evidence…</p> : incidents.length === 0 ? <p className="p-6 text-sm text-muted-foreground">No incidents match this review filter.</p> : incidents.map((incident) => (
              <button key={incident.incident_id} onClick={() => { setSelected(incident); setCoverage(null); setReviewStatus(incident.review_status); }} className={`w-full border-b border-border px-4 py-4 text-left transition-colors hover:bg-muted/60 ${selected?.incident_id === incident.incident_id ? 'bg-primary/5' : ''}`}>
                <div className="flex items-start justify-between gap-3"><div><p className="font-semibold text-foreground">{incident.source.replaceAll('_', ' ')} evidence</p><p className="mt-1 text-xs text-muted-foreground">{new Date(incident.observed_at || incident.created_at).toLocaleString()} · {incident.model_id} {incident.model_version}</p></div><span className={`rounded-full px-2.5 py-1 text-xs font-semibold capitalize ${severityTone[incident.severity]}`}>{incident.severity}</span></div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-sm"><span><strong>{formatArea(incident.oil_area_m2)}</strong><br /><small className="text-muted-foreground">extent</small></span><span><strong>{incident.confidence === null ? '—' : `${Math.round(incident.confidence * 100)}%`}</strong><br /><small className="text-muted-foreground">confidence</small></span><span className="capitalize"><strong>{incident.review_status.replaceAll('_', ' ')}</strong><br /><small className="text-muted-foreground">review state</small></span></div>
              </button>
            ))}
          </div>
        </article>

        <aside className="rounded-2xl border border-border bg-card p-5 shadow-sm">
          {!selected ? <div className="py-10 text-center text-sm text-muted-foreground"><MapPinned className="mx-auto mb-3 h-8 w-8" />Select an incident to inspect its evidence and response-safe controls.</div> : <>
            <div className="flex items-start justify-between gap-3"><div><p className="text-sm font-medium text-muted-foreground">Selected incident</p><h2 className="mt-1 break-all font-semibold text-foreground">{selected.incident_id}</h2></div><Signal className="h-5 w-5 text-primary" /></div>
            <div className="mt-4 rounded-xl bg-muted/60 p-3 text-sm"><p><strong>Extent:</strong> {formatArea(selected.oil_area_m2)}</p><p className="mt-1"><strong>Evidence:</strong> {selected.mask_dimensions[0]} × {selected.mask_dimensions[1]} px</p><p className="mt-1"><strong>Flags:</strong> {selected.quality_flags.length ? selected.quality_flags.join(', ') : 'None'}</p></div>
            <form onSubmit={(event) => void submitReview(event)} className="mt-5 space-y-3 border-t border-border pt-5"><h3 className="font-semibold text-foreground">Human review</h3><label className="block text-sm font-medium">Decision<select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value as ReviewStatus)} className="mt-1 w-full rounded-lg border border-input bg-background p-2">{reviewOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><label className="block text-sm font-medium">Review note<textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} rows={3} className="mt-1 w-full rounded-lg border border-input bg-background p-2" placeholder="What was verified, uncertain, or requires a resurvey?" /></label><button disabled={busy} className="w-full rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground">Record review</button></form>
            <div className="mt-5 grid gap-2 border-t border-border pt-5"><button onClick={() => void createCoveragePlan()} disabled={busy} className="inline-flex items-center justify-center gap-2 rounded-lg border border-input px-4 py-2 text-sm font-semibold hover:bg-muted"><MapPinned className="h-4 w-4" /> Prepare advisory coverage grid</button><button onClick={() => void downloadGeoJson()} className="inline-flex items-center justify-center gap-2 rounded-lg border border-input px-4 py-2 text-sm font-semibold hover:bg-muted"><Download className="h-4 w-4" /> Export GeoJSON evidence</button></div>
            {coverage && <div className="mt-4 rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3 text-sm"><p className="font-semibold">Coverage-plan preview</p><p className="mt-1">{coverage.priority_cells.length} priority cell(s) across {formatArea(coverage.recommended_search_area_m2)}.</p><p className="mt-2 text-xs text-muted-foreground">{coverage.notes[0]}</p></div>}
          </>}
        </aside>
      </div>
    </section>
  );
}
