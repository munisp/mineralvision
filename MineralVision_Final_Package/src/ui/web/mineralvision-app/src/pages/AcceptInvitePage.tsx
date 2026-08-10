import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AxiosError } from 'axios';
import { onboardingApi } from '../services/api';

interface InviteInfo {
  email: string;
  role: string;
  org: { id: number; name: string; slug: string } | null;
  expires_at: string;
}

export default function AcceptInvitePage() {
  const { token } = useParams<{ token: string }>();
  const navigate = useNavigate();

  const [invite, setInvite] = useState<InviteInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!token) {
      setLoadError('Missing invitation token.');
      return;
    }
    onboardingApi
      .validateInvite(token)
      .then((r) => setInvite(r.data as InviteInfo))
      .catch((err: AxiosError) => {
        const detail = (err.response?.data as { detail?: string } | undefined)?.detail;
        setLoadError(detail || 'This invitation is invalid or has expired.');
      });
  }, [token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    setIsSubmitting(true);
    try {
      await onboardingApi.acceptInvite(token!, {
        password,
        full_name: fullName || undefined,
      });
      setDone(true);
      setTimeout(() => navigate('/login'), 2000);
    } catch (err) {
      const detail = ((err as AxiosError).response?.data as { detail?: string } | undefined)?.detail;
      setError(detail || 'Failed to accept the invitation.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-md p-6 space-y-4">
        <h1 className="text-2xl font-semibold text-foreground">Accept Invitation</h1>

        {loadError && (
          <div className="text-sm px-3 py-2 rounded-lg bg-red-500/10 text-red-500">{loadError}</div>
        )}

        {!loadError && !invite && <p className="text-muted-foreground">Validating invitation…</p>}

        {invite && !done && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <p className="text-sm text-muted-foreground">
              You are joining <span className="font-medium text-foreground">{invite.org?.name ?? 'an organization'}</span> as{' '}
              <span className="font-medium text-foreground">{invite.role.replace(/_/g, ' ')}</span> ({invite.email}).
            </p>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Full Name</label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Confirm Password</label>
              <input
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
              />
            </div>
            {error && (
              <div className="text-sm px-3 py-2 rounded-lg bg-red-500/10 text-red-500">{error}</div>
            )}
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
            >
              {isSubmitting ? 'Creating account…' : 'Create Account'}
            </button>
          </form>
        )}

        {done && (
          <div className="text-sm px-3 py-2 rounded-lg bg-green-500/10 text-green-500">
            Account created! Redirecting you to login…
          </div>
        )}
      </div>
    </div>
  );
}
