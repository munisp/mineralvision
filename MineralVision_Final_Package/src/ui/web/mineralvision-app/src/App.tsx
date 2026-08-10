import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import Layout from './components/layout/Layout';
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';
import AcceptInvitePage from './pages/AcceptInvitePage';
import DashboardPage from './pages/dashboard/DashboardPage';
import ProjectsPage from './pages/projects/ProjectsPage';
import ProjectDetailPage from './pages/projects/ProjectDetailPage';
import DrillholesPage from './pages/geology/DrillholesPage';
import QAQCPage from './pages/geology/QAQCPage';
import CrossSectionsPage from './pages/geology/CrossSectionsPage';
import VariographyPage from './pages/geostatistics/VariographyPage';
import KrigingPage from './pages/geostatistics/KrigingPage';
import BlockModelPage from './pages/geostatistics/BlockModelPage';
import Visualization3DPage from './pages/visualization/Visualization3DPage';
import InversionPage from './pages/geophysics/InversionPage';
import ReportsPage from './pages/reporting/ReportsPage';
import SensorFusionPage from './pages/sensors/SensorFusionPage';
import GNSSPage from './pages/gnss/GNSSPage';
import AIInsightsPage from './pages/ai-insights/AIInsightsPage';
import Molmo2Page from './pages/molmo2/Molmo2Page';
import SettingsPage from './pages/settings/SettingsPage';
import UsersPage from './pages/admin/UsersPage';
import JourneysPage from './pages/journeys/JourneysPage';
import MapExplorerPage from './pages/geology/MapExplorerPage';
import MineralMonitoringPage from './pages/mineral-monitoring/MineralMonitoringPage';
import { Toaster } from './components/ui/toaster';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
}

function App() {
  return (
    <>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/accept-invite/:token" element={<AcceptInvitePage />} />
        
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="projects" element={<ProjectsPage />} />
          <Route path="projects/:id" element={<ProjectDetailPage />} />
          <Route path="drillholes" element={<DrillholesPage />} />
          <Route path="qaqc" element={<QAQCPage />} />
          <Route path="cross-sections" element={<CrossSectionsPage />} />
          <Route path="variography" element={<VariographyPage />} />
          <Route path="kriging" element={<KrigingPage />} />
          <Route path="block-model" element={<BlockModelPage />} />
          <Route path="visualization" element={<Visualization3DPage />} />
          <Route path="inversion" element={<InversionPage />} />
          <Route path="reports" element={<ReportsPage />} />
                                        <Route path="sensor-fusion" element={<SensorFusionPage />} />
                                        <Route path="gnss" element={<GNSSPage />} />
                                                                                <Route path="ai-insights" element={<AIInsightsPage />} />
                                                                                          <Route path="molmo2" element={<Molmo2Page />} />
                                        <Route path="map-explorer" element={<MapExplorerPage />} />
                                        <Route path="journeys" element={<JourneysPage />} />
                                                                                <Route path="mineral-monitoring" element={<MineralMonitoringPage />} />
                                                                                <Route path="settings" element={<SettingsPage />} />
                    <Route path="admin/users" element={<UsersPage />} />
        </Route>
      </Routes>
      <Toaster />
    </>
  );
}

export default App;
