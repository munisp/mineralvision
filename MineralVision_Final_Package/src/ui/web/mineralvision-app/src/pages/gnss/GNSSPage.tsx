import { useEffect, useState } from 'react';
import { Satellite, SignalZero, WifiOff, RefreshCw } from 'lucide-react';
import api from '../../services/api';

type ProbeState = 'checking' | 'unavailable';

/**
 * Enhanced GNSS page.
 *
 * Runtime QA finding (docs/RUNTIME_QA.md follow-up): the backend exposes NO
 * GNSS HTTP API — `src/api/gnss/` modules are not mounted as routers
 * (verified against /openapi.json). The previous version of this page showed
 * hardcoded satellites, an "RTK Fixed / 2.5 cm" position and a Math.random()
 * accuracy history under a pulsing green "Live" badge. All of that was
 * fabricated and has been removed.
 *
 * This page now honestly reports that no live GNSS feed is available. If a
 * GNSS endpoint is added to the backend later, probe it here and render the
 * real data.
 */
export default function GNSSPage() {
  const [state, setState] = useState<ProbeState>('checking');

  const probe = async () => {
    setState('checking');
    try {
      // The API catalogue is the source of truth for whether a GNSS feed exists.
      const response = await api.get('/openapi.json', {
        validateStatus: (s) => s < 500,
      });
      // No GNSS routes exist in the backend today, so this always ends in the
      // honest 'unavailable' state; the probe keeps the Retry button meaningful
      // and future-proofs the page for when a GNSS router is mounted.
      void response;
      setState('unavailable');
    } catch {
      setState('unavailable');
    }
  };

  useEffect(() => {
    void probe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Satellite className="h-6 w-6" />
            Enhanced GNSS
          </h1>
          <p className="text-muted-foreground">
            RTK positioning, satellite tracking and accuracy monitoring
          </p>
        </div>
        <button
          onClick={() => void probe()}
          className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>

      <div className="bg-card border border-border rounded-xl p-10 flex flex-col items-center text-center gap-4">
        {state === 'checking' ? (
          <>
            <Satellite className="h-12 w-12 text-muted-foreground animate-pulse" />
            <p className="text-foreground font-medium">Checking for a GNSS service…</p>
          </>
        ) : (
          <>
            <div className="flex items-center gap-3 text-muted-foreground">
              <WifiOff className="h-10 w-10" />
              <SignalZero className="h-10 w-10" />
            </div>
            <p className="text-foreground font-medium text-lg">No live GNSS feed available</p>
            <p className="text-sm text-muted-foreground max-w-lg">
              The MineralVision backend does not currently expose a GNSS API, so there is no real
              position, satellite or accuracy data to display. This page intentionally shows no
              positioning data rather than simulated values.
            </p>
            <p className="text-xs text-muted-foreground max-w-lg">
              To enable this page, mount the GNSS ingestion pipeline
              (<code>api/ingestion/gnss_ingestion.py</code>) as an HTTP router and expose receiver
              status, satellite and fix endpoints.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
