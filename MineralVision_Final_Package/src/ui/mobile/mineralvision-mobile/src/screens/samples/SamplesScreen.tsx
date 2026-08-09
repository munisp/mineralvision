import React, { useState } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../navigation/RootNavigator';

type RouteProps = RouteProp<RootStackParamList, 'Samples'>;

interface Sample {
  id: string;
  sampleId: string;
  fromDepth: number;
  toDepth: number;
  sampleType: string;
  assays: {
    Au?: number;
    Cu?: number;
    Ag?: number;
  };
  status: 'assayed' | 'pending' | 'submitted';
}

const mockSamples: Sample[] = [
  { id: '1', sampleId: 'DDH-156-001', fromDepth: 0, toDepth: 2, sampleType: 'Core', assays: { Au: 0.12, Cu: 0.05 }, status: 'assayed' },
  { id: '2', sampleId: 'DDH-156-002', fromDepth: 2, toDepth: 4, sampleType: 'Core', assays: { Au: 0.45, Cu: 0.12 }, status: 'assayed' },
  { id: '3', sampleId: 'DDH-156-003', fromDepth: 4, toDepth: 6, sampleType: 'Core', assays: { Au: 1.25, Cu: 0.35 }, status: 'assayed' },
  { id: '4', sampleId: 'DDH-156-004', fromDepth: 6, toDepth: 8, sampleType: 'Core', assays: { Au: 2.85, Cu: 0.82 }, status: 'assayed' },
  { id: '5', sampleId: 'DDH-156-005', fromDepth: 8, toDepth: 10, sampleType: 'Core', assays: { Au: 1.92, Cu: 0.65 }, status: 'assayed' },
  { id: '6', sampleId: 'DDH-156-006', fromDepth: 10, toDepth: 12, sampleType: 'Core', assays: {}, status: 'submitted' },
  { id: '7', sampleId: 'DDH-156-007', fromDepth: 12, toDepth: 14, sampleType: 'Core', assays: {}, status: 'pending' },
];

const statusColors = {
  assayed: '#10b981',
  submitted: '#3b82f6',
  pending: '#f59e0b',
};

function SampleCard({ sample }: { sample: Sample }) {
  const hasAssays = Object.keys(sample.assays).length > 0;

  return (
    <TouchableOpacity style={styles.card} activeOpacity={0.7}>
      <View style={styles.cardHeader}>
        <View>
          <Text style={styles.sampleId}>{sample.sampleId}</Text>
          <Text style={styles.interval}>
            {sample.fromDepth}m - {sample.toDepth}m ({sample.toDepth - sample.fromDepth}m)
          </Text>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: `${statusColors[sample.status]}20` }]}>
          <Text style={[styles.statusText, { color: statusColors[sample.status] }]}>
            {sample.status}
          </Text>
        </View>
      </View>

      <View style={styles.cardBody}>
        <View style={styles.typeRow}>
          <Ionicons name="cube-outline" size={16} color="#6b7280" />
          <Text style={styles.typeText}>{sample.sampleType}</Text>
        </View>

        {hasAssays ? (
          <View style={styles.assaysRow}>
            {sample.assays.Au !== undefined && (
              <View style={styles.assayItem}>
                <Text style={styles.assayLabel}>Au</Text>
                <Text style={[styles.assayValue, sample.assays.Au > 1 && styles.highGrade]}>
                  {sample.assays.Au} g/t
                </Text>
              </View>
            )}
            {sample.assays.Cu !== undefined && (
              <View style={styles.assayItem}>
                <Text style={styles.assayLabel}>Cu</Text>
                <Text style={styles.assayValue}>{sample.assays.Cu}%</Text>
              </View>
            )}
            {sample.assays.Ag !== undefined && (
              <View style={styles.assayItem}>
                <Text style={styles.assayLabel}>Ag</Text>
                <Text style={styles.assayValue}>{sample.assays.Ag} g/t</Text>
              </View>
            )}
          </View>
        ) : (
          <Text style={styles.noAssays}>Awaiting assay results</Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

export default function SamplesScreen() {
  const route = useRoute<RouteProps>();
  const { drillholeId } = route.params;
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  const assayedCount = mockSamples.filter((s) => s.status === 'assayed').length;
  const pendingCount = mockSamples.filter((s) => s.status !== 'assayed').length;

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.summaryRow}>
        <View style={styles.summaryCard}>
          <Text style={styles.summaryValue}>{mockSamples.length}</Text>
          <Text style={styles.summaryLabel}>Total Samples</Text>
        </View>
        <View style={styles.summaryCard}>
          <Text style={[styles.summaryValue, { color: '#10b981' }]}>{assayedCount}</Text>
          <Text style={styles.summaryLabel}>Assayed</Text>
        </View>
        <View style={styles.summaryCard}>
          <Text style={[styles.summaryValue, { color: '#f59e0b' }]}>{pendingCount}</Text>
          <Text style={styles.summaryLabel}>Pending</Text>
        </View>
      </View>

      <FlatList
        data={mockSamples}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => <SampleCard sample={item} />}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3b82f6" />
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons name="flask-outline" size={48} color="#6b7280" />
            <Text style={styles.emptyText}>No samples found</Text>
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
  summaryRow: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  summaryCard: {
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
    fontSize: 11,
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
  sampleId: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  interval: {
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
    gap: 12,
  },
  typeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  typeText: {
    fontSize: 14,
    color: '#9ca3af',
  },
  assaysRow: {
    flexDirection: 'row',
    gap: 16,
  },
  assayItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  assayLabel: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '500',
  },
  assayValue: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '600',
  },
  highGrade: {
    color: '#10b981',
  },
  noAssays: {
    fontSize: 14,
    color: '#6b7280',
    fontStyle: 'italic',
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
