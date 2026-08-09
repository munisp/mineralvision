import React, { useState } from 'react';
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

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

interface Drillhole {
  id: string;
  holeId: string;
  project: string;
  depth: number;
  azimuth: number;
  dip: number;
  status: 'completed' | 'in-progress' | 'planned';
  assayCount: number;
  avgGrade: number | null;
}

const mockDrillholes: Drillhole[] = [
  { id: '1', holeId: 'DDH-2024-156', project: 'Copper Ridge', depth: 450, azimuth: 45, dip: -60, status: 'completed', assayCount: 45, avgGrade: 0.85 },
  { id: '2', holeId: 'DDH-2024-155', project: 'Copper Ridge', depth: 380, azimuth: 45, dip: -60, status: 'completed', assayCount: 38, avgGrade: 1.12 },
  { id: '3', holeId: 'DDH-2024-154', project: 'Copper Ridge', depth: 520, azimuth: 90, dip: -55, status: 'completed', assayCount: 52, avgGrade: 0.72 },
  { id: '4', holeId: 'DDH-2024-153', project: 'Copper Ridge', depth: 290, azimuth: 45, dip: -60, status: 'in-progress', assayCount: 15, avgGrade: null },
  { id: '5', holeId: 'GV-2024-089', project: 'Golden Valley', depth: 320, azimuth: 0, dip: -90, status: 'completed', assayCount: 64, avgGrade: 2.45 },
  { id: '6', holeId: 'GV-2024-088', project: 'Golden Valley', depth: 280, azimuth: 0, dip: -90, status: 'completed', assayCount: 56, avgGrade: 3.12 },
  { id: '7', holeId: 'LF-2024-012', project: 'Lithium Flats', depth: 150, azimuth: 0, dip: -90, status: 'completed', assayCount: 30, avgGrade: 850 },
];

const statusColors = {
  completed: '#10b981',
  'in-progress': '#f59e0b',
  planned: '#3b82f6',
};

function DrillholeCard({ drillhole, onPress }: { drillhole: Drillhole; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.cardHeader}>
        <View>
          <Text style={styles.holeId}>{drillhole.holeId}</Text>
          <Text style={styles.project}>{drillhole.project}</Text>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: `${statusColors[drillhole.status]}20` }]}>
          <Text style={[styles.statusText, { color: statusColors[drillhole.status] }]}>
            {drillhole.status}
          </Text>
        </View>
      </View>

      <View style={styles.cardBody}>
        <View style={styles.infoRow}>
          <View style={styles.infoItem}>
            <Ionicons name="resize-outline" size={16} color="#6b7280" />
            <Text style={styles.infoText}>{drillhole.depth}m</Text>
          </View>
          <View style={styles.infoItem}>
            <Ionicons name="compass-outline" size={16} color="#6b7280" />
            <Text style={styles.infoText}>{drillhole.azimuth}/{drillhole.dip}</Text>
          </View>
          <View style={styles.infoItem}>
            <Ionicons name="flask-outline" size={16} color="#6b7280" />
            <Text style={styles.infoText}>{drillhole.assayCount} assays</Text>
          </View>
        </View>
      </View>

      <View style={styles.cardFooter}>
        <Text style={styles.gradeLabel}>Avg Grade:</Text>
        {drillhole.avgGrade !== null ? (
          <Text style={styles.gradeValue}>{drillhole.avgGrade} g/t</Text>
        ) : (
          <Text style={styles.gradePending}>Pending</Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

export default function DrillholesScreen() {
  const navigation = useNavigation<NavigationProp>();
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  const filteredDrillholes = mockDrillholes.filter((hole) =>
    hole.holeId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    hole.project.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.searchContainer}>
        <View style={styles.searchInputContainer}>
          <Ionicons name="search" size={20} color="#6b7280" />
          <TextInput
            style={styles.searchInput}
            placeholder="Search drillholes..."
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

      <View style={styles.summaryRow}>
        <View style={styles.summaryItem}>
          <Text style={styles.summaryValue}>{mockDrillholes.length}</Text>
          <Text style={styles.summaryLabel}>Total</Text>
        </View>
        <View style={styles.summaryItem}>
          <Text style={[styles.summaryValue, { color: '#10b981' }]}>
            {mockDrillholes.filter((h) => h.status === 'completed').length}
          </Text>
          <Text style={styles.summaryLabel}>Completed</Text>
        </View>
        <View style={styles.summaryItem}>
          <Text style={[styles.summaryValue, { color: '#f59e0b' }]}>
            {mockDrillholes.filter((h) => h.status === 'in-progress').length}
          </Text>
          <Text style={styles.summaryLabel}>In Progress</Text>
        </View>
      </View>

      <FlatList
        data={filteredDrillholes}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <DrillholeCard
            drillhole={item}
            onPress={() => navigation.navigate('Samples', { drillholeId: item.id })}
          />
        )}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3b82f6" />
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons name="layers-outline" size={48} color="#6b7280" />
            <Text style={styles.emptyText}>No drillholes found</Text>
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
  summaryRow: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    marginBottom: 12,
    gap: 12,
  },
  summaryItem: {
    flex: 1,
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#374151',
  },
  summaryValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  summaryLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  listContent: {
    padding: 16,
    paddingTop: 0,
  },
  card: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#374151',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  holeId: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  project: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 2,
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
  cardBody: {
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: 'row',
    gap: 16,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  infoText: {
    fontSize: 14,
    color: '#9ca3af',
  },
  cardFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#374151',
  },
  gradeLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginRight: 8,
  },
  gradeValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#10b981',
  },
  gradePending: {
    fontSize: 14,
    color: '#f59e0b',
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
