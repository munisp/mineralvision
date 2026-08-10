import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/RootNavigator';
import { projectsApi, drillholesApi, Project as ApiProject } from '../../services/api';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

interface Project {
  id: string;
  name: string;
  description: string;
  location: string;
  commodities: string[];
  status: 'active' | 'completed' | 'on-hold';
  drillholes: number;
  lastUpdated: string;
}


const statusColors = {
  active: '#10b981',
  completed: '#3b82f6',
  'on-hold': '#f59e0b',
};

const commodityColors: Record<string, string> = {
  Gold: '#f59e0b',
  Copper: '#f97316',
  Lithium: '#8b5cf6',
  Iron: '#ef4444',
};

function ProjectCard({ project, onPress }: { project: Project; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.projectCard} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.projectHeader}>
        <View style={styles.projectTitleRow}>
          <Text style={styles.projectName} numberOfLines={1}>{project.name}</Text>
          <View style={[styles.statusBadge, { backgroundColor: `${statusColors[project.status]}20` }]}>
            <Text style={[styles.statusText, { color: statusColors[project.status] }]}>
              {project.status}
            </Text>
          </View>
        </View>
        <Text style={styles.projectDescription} numberOfLines={2}>{project.description}</Text>
      </View>

      <View style={styles.commoditiesRow}>
        {project.commodities.map((commodity) => (
          <View
            key={commodity}
            style={[styles.commodityBadge, { backgroundColor: `${commodityColors[commodity] || '#6b7280'}20` }]}
          >
            <Text style={[styles.commodityText, { color: commodityColors[commodity] || '#6b7280' }]}>
              {commodity}
            </Text>
          </View>
        ))}
      </View>

      <View style={styles.projectFooter}>
        <View style={styles.footerItem}>
          <Ionicons name="location-outline" size={14} color="#6b7280" />
          <Text style={styles.footerText}>{project.location}</Text>
        </View>
        <View style={styles.footerItem}>
          <Ionicons name="layers-outline" size={14} color="#6b7280" />
          <Text style={styles.footerText}>{project.drillholes} holes</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

export default function ProjectsScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Live data: projects + per-project drillhole counts from the register.
  const load = useCallback(async () => {
    setError(null);
    try {
      const [projectsResp, holesResp] = await Promise.all([
        projectsApi.getAll(),
        drillholesApi.getAll(),
      ]);
      const counts = new Map<string, number>();
      for (const h of holesResp.data) {
        counts.set(h.projectId, (counts.get(h.projectId) ?? 0) + 1);
      }
      setProjects(
        projectsResp.data.map((p: ApiProject) => ({
          id: p.id,
          name: p.name,
          description: p.description,
          location: p.location,
          commodities: p.commodities ?? [],
          status: p.status,
          drillholes: counts.get(p.id) ?? 0,
          lastUpdated: p.updatedAt,
        })),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const filteredProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    project.location.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    load().finally(() => setRefreshing(false));
  }, [load]);

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.searchContainer}>
        <View style={styles.searchInputContainer}>
          <Ionicons name="search" size={20} color="#6b7280" />
          <TextInput
            style={styles.searchInput}
            placeholder="Search projects..."
            placeholderTextColor="#6b7280"
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery('')}>
              <Ionicons name="close-circle" size={20} color="#6b7280" />
            </TouchableOpacity>
          )}
        </View>
        <TouchableOpacity style={styles.filterButton}>
          <Ionicons name="filter" size={20} color="#fff" />
        </TouchableOpacity>
      </View>

      <FlatList
        data={filteredProjects}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <ProjectCard
            project={item}
            onPress={() => navigation.navigate('ProjectDetail', { projectId: item.id })}
          />
        )}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3b82f6" />
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons name="folder-open-outline" size={48} color="#6b7280" />
            <Text style={styles.emptyText}>
              {loading
                ? 'Loading projects…'
                : error
                  ? `Could not load projects: ${error}`
                  : 'No projects registered'}
            </Text>
          </View>
        }
      />

      <TouchableOpacity style={styles.fab}>
        <Ionicons name="add" size={28} color="#fff" />
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  searchContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 12,
  },
  searchInputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1f2937',
    borderRadius: 12,
    paddingHorizontal: 12,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    height: 44,
    color: '#fff',
    fontSize: 16,
  },
  filterButton: {
    width: 44,
    height: 44,
    backgroundColor: '#1f2937',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContent: {
    padding: 16,
    paddingTop: 0,
  },
  projectCard: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#374151',
  },
  projectHeader: {
    marginBottom: 12,
  },
  projectTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  projectName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    flex: 1,
    marginRight: 8,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  projectDescription: {
    fontSize: 14,
    color: '#9ca3af',
  },
  commoditiesRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
  },
  commodityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
  },
  commodityText: {
    fontSize: 12,
    fontWeight: '500',
  },
  projectFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#374151',
  },
  footerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  footerText: {
    fontSize: 12,
    color: '#6b7280',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 48,
  },
  emptyText: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 12,
  },
  fab: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#3b82f6',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
});
