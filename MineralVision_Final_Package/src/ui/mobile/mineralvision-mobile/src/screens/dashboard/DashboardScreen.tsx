import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../navigation/RootNavigator';
import { useAuthStore } from '../../store/authStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

interface StatCardProps {
  title: string;
  value: string | number;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  onPress?: () => void;
}

function StatCard({ title, value, icon, color, onPress }: StatCardProps) {
  return (
    <TouchableOpacity style={styles.statCard} onPress={onPress} activeOpacity={0.7}>
      <View style={[styles.statIcon, { backgroundColor: `${color}20` }]}>
        <Ionicons name={icon} size={24} color={color} />
      </View>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statTitle}>{title}</Text>
    </TouchableOpacity>
  );
}

interface ActivityItemProps {
  icon: keyof typeof Ionicons.glyphMap;
  message: string;
  time: string;
  status: 'success' | 'warning' | 'info';
}

function ActivityItem({ icon, message, time, status }: ActivityItemProps) {
  const statusColors = {
    success: '#10b981',
    warning: '#f59e0b',
    info: '#3b82f6',
  };

  return (
    <View style={styles.activityItem}>
      <View style={[styles.activityIcon, { backgroundColor: `${statusColors[status]}20` }]}>
        <Ionicons name={icon} size={16} color={statusColors[status]} />
      </View>
      <View style={styles.activityContent}>
        <Text style={styles.activityMessage} numberOfLines={2}>{message}</Text>
        <Text style={styles.activityTime}>{time}</Text>
      </View>
    </View>
  );
}

export default function DashboardScreen() {
  const navigation = useNavigation<NavigationProp>();
  const { user } = useAuthStore();
  const [refreshing, setRefreshing] = React.useState(false);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3b82f6" />
        }
      >
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Welcome back,</Text>
            <Text style={styles.userName}>{user?.firstName || 'User'}</Text>
          </View>
          <TouchableOpacity style={styles.notificationButton}>
            <Ionicons name="notifications-outline" size={24} color="#fff" />
            <View style={styles.notificationBadge} />
          </TouchableOpacity>
        </View>

        <View style={styles.statsGrid}>
          <StatCard
            title="Projects"
            value={12}
            icon="folder"
            color="#3b82f6"
            onPress={() => navigation.navigate('Main')}
          />
          <StatCard
            title="Drillholes"
            value={847}
            icon="layers"
            color="#10b981"
          />
          <StatCard
            title="Samples"
            value="12.4K"
            icon="flask"
            color="#f59e0b"
          />
          <StatCard
            title="Reports"
            value={24}
            icon="document-text"
            color="#8b5cf6"
          />
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Quick Actions</Text>
          </View>
          <View style={styles.quickActions}>
            <TouchableOpacity style={styles.quickAction}>
              <View style={[styles.quickActionIcon, { backgroundColor: '#3b82f620' }]}>
                <Ionicons name="add-circle" size={24} color="#3b82f6" />
              </View>
              <Text style={styles.quickActionText}>New Sample</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.quickAction}
              onPress={() => navigation.navigate('Camera', { mode: 'photo' })}
            >
              <View style={[styles.quickActionIcon, { backgroundColor: '#10b98120' }]}>
                <Ionicons name="camera" size={24} color="#10b981" />
              </View>
              <Text style={styles.quickActionText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.quickAction}
              onPress={() => navigation.navigate('Camera', { mode: 'waldo' })}
            >
              <View style={[styles.quickActionIcon, { backgroundColor: '#f59e0b20' }]}>
                <Ionicons name="scan" size={24} color="#f59e0b" />
              </View>
              <Text style={styles.quickActionText}>WALDO Scan</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickAction}>
              <View style={[styles.quickActionIcon, { backgroundColor: '#8b5cf620' }]}>
                <Ionicons name="cloud-upload" size={24} color="#8b5cf6" />
              </View>
              <Text style={styles.quickActionText}>Upload Data</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Recent Activity</Text>
            <TouchableOpacity>
              <Text style={styles.seeAll}>See All</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.activityList}>
            <ActivityItem
              icon="checkmark-circle"
              message="DDH-2024-156 added to Copper Ridge Project"
              time="2 hours ago"
              status="success"
            />
            <ActivityItem
              icon="warning"
              message="QA/QC alert: Standard deviation exceeded for CRM-Au-01"
              time="4 hours ago"
              status="warning"
            />
            <ActivityItem
              icon="cube"
              message="Block model estimation completed for Zone A"
              time="6 hours ago"
              status="success"
            />
            <ActivityItem
              icon="document"
              message="NI 43-101 report draft generated"
              time="1 day ago"
              status="info"
            />
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Resource Summary</Text>
          </View>
          <View style={styles.resourceCard}>
            <View style={styles.resourceRow}>
              <Text style={styles.resourceLabel}>Measured</Text>
              <Text style={styles.resourceValue}>2.5 Mt @ 1.82 g/t</Text>
            </View>
            <View style={styles.resourceRow}>
              <Text style={styles.resourceLabel}>Indicated</Text>
              <Text style={styles.resourceValue}>5.8 Mt @ 1.45 g/t</Text>
            </View>
            <View style={styles.resourceRow}>
              <Text style={styles.resourceLabel}>Inferred</Text>
              <Text style={styles.resourceValue}>8.2 Mt @ 1.12 g/t</Text>
            </View>
            <View style={[styles.resourceRow, styles.resourceTotal]}>
              <Text style={styles.resourceTotalLabel}>Total</Text>
              <Text style={styles.resourceTotalValue}>16.5 Mt @ 1.32 g/t</Text>
            </View>
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  greeting: {
    fontSize: 14,
    color: '#6b7280',
  },
  userName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  notificationButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#1f2937',
    justifyContent: 'center',
    alignItems: 'center',
  },
  notificationBadge: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ef4444',
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
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#374151',
  },
  statIcon: {
    width: 44,
    height: 44,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statTitle: {
    fontSize: 14,
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
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  quickAction: {
    alignItems: 'center',
    width: '23%',
  },
  quickActionIcon: {
    width: 56,
    height: 56,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  quickActionText: {
    fontSize: 12,
    color: '#9ca3af',
    textAlign: 'center',
  },
  activityList: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#374151',
    overflow: 'hidden',
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#374151',
  },
  activityIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  activityContent: {
    flex: 1,
  },
  activityMessage: {
    fontSize: 14,
    color: '#fff',
    marginBottom: 4,
  },
  activityTime: {
    fontSize: 12,
    color: '#6b7280',
  },
  resourceCard: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#374151',
  },
  resourceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#374151',
  },
  resourceLabel: {
    fontSize: 14,
    color: '#9ca3af',
  },
  resourceValue: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '500',
  },
  resourceTotal: {
    borderBottomWidth: 0,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    marginHorizontal: -16,
    paddingHorizontal: 16,
    marginBottom: -16,
    borderBottomLeftRadius: 16,
    borderBottomRightRadius: 16,
  },
  resourceTotalLabel: {
    fontSize: 14,
    color: '#fff',
    fontWeight: '600',
  },
  resourceTotalValue: {
    fontSize: 14,
    color: '#3b82f6',
    fontWeight: '600',
  },
});
