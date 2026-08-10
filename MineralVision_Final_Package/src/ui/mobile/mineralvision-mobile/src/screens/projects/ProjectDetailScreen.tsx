import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, RouteProp, useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/RootNavigator';
import { projectsApi, drillholesApi, Project, Drillhole } from '../../services/api';

type RouteProps = RouteProp<RootStackParamList, 'ProjectDetail'>;
type NavigationProp = NativeStackNavigationProp<RootStackParamList>;



export default function ProjectDetailScreen() {
  const route = useRoute<RouteProps>();
  const navigation = useNavigation<NavigationProp>();
  const { projectId } = route.params;

  const [project, setProject] = useState<Project | null>(null);
  const [holes, setHoles] = useState<Drillhole[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Live data: the project record + its drillholes from the backend.
  const load = useCallback(async () => {
    setError(null);
    try {
      const [projectResp, holesResp] = await Promise.all([
        projectsApi.getById(projectId),
        drillholesApi.getAll(projectId),
      ]);
      setProject(projectResp.data);
      setHoles(holesResp.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load project');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <SafeAreaView style={styles.container} edges={['bottom']}>
        <Text style={styles.description}>Loading project…</Text>
      </SafeAreaView>
    );
  }

  if (error || !project) {
    return (
      <SafeAreaView style={styles.container} edges={['bottom']}>
        <Text style={styles.description}>
          {error ? `Could not load project: ${error}` : 'Project not found'}
        </Text>
      </SafeAreaView>
    );
  }

  const totalMeters = holes.reduce((sum, h) => sum + (h.totalDepth ?? 0), 0);

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.projectName}>{project.name}</Text>
          <View style={styles.locationRow}>
            <Ionicons name="location-outline" size={16} color="#6b7280" />
            <Text style={styles.locationText}>{project.location}</Text>
          </View>
          <Text style={styles.description}>{project.description}</Text>
          <View style={styles.commoditiesRow}>
            {(project.commodities ?? []).map((commodity) => (
              <View key={commodity} style={styles.commodityBadge}>
                <Text style={styles.commodityText}>{commodity}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Ionicons name="layers-outline" size={24} color="#3b82f6" />
            <Text style={styles.statValue}>{holes.length}</Text>
            <Text style={styles.statLabel}>Drillholes</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="resize-outline" size={24} color="#10b981" />
            <Text style={styles.statValue}>{totalMeters >= 1000 ? `${(totalMeters / 1000).toFixed(1)}K` : `${totalMeters}`}</Text>
            <Text style={styles.statLabel}>Meters</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="flask-outline" size={24} color="#f59e0b" />
            <Text style={styles.statValue}>{'—'}</Text>
            <Text style={styles.statLabel}>Samples</Text>
          </View>
          <View style={styles.statCard}>
            <Ionicons name="cube-outline" size={24} color="#8b5cf6" />
            <Text style={styles.statValue}>{'—'}</Text>
            <Text style={styles.statLabel}>Models</Text>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Quick Actions</Text>
          </View>
          <View style={styles.actionsRow}>
            <TouchableOpacity style={styles.actionButton}>
              <Ionicons name="add-circle-outline" size={20} color="#3b82f6" />
              <Text style={styles.actionText}>Add Drillhole</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.actionButton}
              onPress={() => navigation.navigate('Camera', { mode: 'photo' })}
            >
              <Ionicons name="camera-outline" size={20} color="#3b82f6" />
              <Text style={styles.actionText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionButton}>
              <Ionicons name="cloud-upload-outline" size={20} color="#3b82f6" />
              <Text style={styles.actionText}>Upload Data</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Recent Drillholes</Text>
            <TouchableOpacity>
              <Text style={styles.seeAll}>See All</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.drillholesList}>
            {holes.length === 0 && (
              <Text style={styles.description}>No drillholes registered for this project.</Text>
            )}
            {holes.slice(0, 10).map((hole) => (
              <TouchableOpacity
                key={hole.id}
                style={styles.drillholeCard}
                onPress={() => navigation.navigate('Samples', { drillholeId: hole.id })}
              >
                <View style={styles.drillholeInfo}>
                  <Text style={styles.drillholeId}>{hole.holeId}</Text>
                  <Text style={styles.drillholeDepth}>{hole.totalDepth}m depth</Text>
                </View>
                <View style={styles.drillholeRight}>
                  <Text style={styles.drillholeGrade}>{hole.status}</Text>
                  <Ionicons name="chevron-forward" size={20} color="#6b7280" />
                </View>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  header: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#374151',
  },
  projectName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 12,
  },
  locationText: {
    fontSize: 14,
    color: '#6b7280',
  },
  description: {
    fontSize: 14,
    color: '#9ca3af',
    lineHeight: 20,
    marginBottom: 12,
  },
  commoditiesRow: {
    flexDirection: 'row',
    gap: 8,
  },
  commodityBadge: {
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  commodityText: {
    fontSize: 14,
    color: '#3b82f6',
    fontWeight: '500',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 24,
  },
  statCard: {
    width: '48%',
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#374151',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  seeAll: {
    fontSize: 14,
    color: '#3b82f6',
  },
  actionsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: '#374151',
  },
  actionText: {
    fontSize: 12,
    color: '#9ca3af',
  },
  drillholesList: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#374151',
    overflow: 'hidden',
  },
  drillholeCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#374151',
  },
  drillholeInfo: {
    flex: 1,
  },
  drillholeId: {
    fontSize: 16,
    fontWeight: '500',
    color: '#fff',
  },
  drillholeDepth: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 2,
  },
  drillholeRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  drillholeGrade: {
    fontSize: 16,
    fontWeight: '600',
    color: '#10b981',
  },
  drillholeInProgress: {
    fontSize: 14,
    color: '#f59e0b',
  },
});
