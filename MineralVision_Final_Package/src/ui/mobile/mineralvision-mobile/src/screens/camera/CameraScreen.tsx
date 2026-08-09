import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, RouteProp, useNavigation } from '@react-navigation/native';
import { RootStackParamList } from '../../navigation/RootNavigator';

type RouteProps = RouteProp<RootStackParamList, 'Camera'>;

export default function CameraScreen() {
  const route = useRoute<RouteProps>();
  const navigation = useNavigation();
  const { mode } = route.params;
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  const handleCapture = async () => {
    setIsProcessing(true);
    
    // Simulate capture and processing
    setTimeout(() => {
      setIsProcessing(false);
      if (mode === 'waldo') {
        Alert.alert(
          'WALDO Detection',
          'Detected: Core sample box\nConfidence: 94.2%\nMineralization: Visible sulfides',
          [{ text: 'Save', onPress: () => navigation.goBack() }]
        );
      } else {
        Alert.alert(
          'Photo Captured',
          'Photo saved successfully',
          [{ text: 'OK', onPress: () => navigation.goBack() }]
        );
      }
    }, 2000);
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <View style={styles.cameraContainer}>
        <View style={styles.cameraPlaceholder}>
          <Ionicons name="camera" size={64} color="#6b7280" />
          <Text style={styles.placeholderText}>Camera Preview</Text>
          <Text style={styles.placeholderSubtext}>
            {mode === 'waldo' ? 'WALDO Object Detection Mode' : 'Photo Capture Mode'}
          </Text>
        </View>

        {mode === 'waldo' && (
          <View style={styles.detectionOverlay}>
            <View style={styles.detectionBox}>
              <View style={styles.cornerTL} />
              <View style={styles.cornerTR} />
              <View style={styles.cornerBL} />
              <View style={styles.cornerBR} />
            </View>
          </View>
        )}
      </View>

      <View style={styles.controls}>
        <View style={styles.modeIndicator}>
          <Ionicons
            name={mode === 'waldo' ? 'scan' : 'camera'}
            size={20}
            color="#3b82f6"
          />
          <Text style={styles.modeText}>
            {mode === 'waldo' ? 'WALDO Detection' : 'Photo Mode'}
          </Text>
        </View>

        <View style={styles.captureRow}>
          <TouchableOpacity style={styles.sideButton}>
            <Ionicons name="images-outline" size={24} color="#fff" />
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.captureButton, isProcessing && styles.captureButtonDisabled]}
            onPress={handleCapture}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <ActivityIndicator color="#fff" size="large" />
            ) : (
              <View style={styles.captureInner} />
            )}
          </TouchableOpacity>

          <TouchableOpacity style={styles.sideButton}>
            <Ionicons name="camera-reverse-outline" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        {mode === 'waldo' && (
          <View style={styles.waldoInfo}>
            <Text style={styles.waldoInfoTitle}>WALDO Detection Classes:</Text>
            <View style={styles.waldoClasses}>
              <View style={styles.waldoClass}>
                <View style={[styles.waldoClassDot, { backgroundColor: '#10b981' }]} />
                <Text style={styles.waldoClassText}>Core Box</Text>
              </View>
              <View style={styles.waldoClass}>
                <View style={[styles.waldoClassDot, { backgroundColor: '#3b82f6' }]} />
                <Text style={styles.waldoClassText}>Sample Bag</Text>
              </View>
              <View style={styles.waldoClass}>
                <View style={[styles.waldoClassDot, { backgroundColor: '#f59e0b' }]} />
                <Text style={styles.waldoClassText}>Equipment</Text>
              </View>
              <View style={styles.waldoClass}>
                <View style={[styles.waldoClassDot, { backgroundColor: '#8b5cf6' }]} />
                <Text style={styles.waldoClassText}>Mineralization</Text>
              </View>
            </View>
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  cameraPlaceholder: {
    flex: 1,
    backgroundColor: '#1f2937',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 18,
    color: '#9ca3af',
    marginTop: 16,
  },
  placeholderSubtext: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  detectionOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  detectionBox: {
    width: 250,
    height: 250,
    position: 'relative',
  },
  cornerTL: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: 40,
    height: 40,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#3b82f6',
  },
  cornerTR: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 40,
    height: 40,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderColor: '#3b82f6',
  },
  cornerBL: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: 40,
    height: 40,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#3b82f6',
  },
  cornerBR: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 40,
    height: 40,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderColor: '#3b82f6',
  },
  controls: {
    backgroundColor: '#0f172a',
    paddingVertical: 24,
    paddingHorizontal: 16,
  },
  modeIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 24,
  },
  modeText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '500',
  },
  captureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 40,
  },
  sideButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#1f2937',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#374151',
  },
  captureButtonDisabled: {
    backgroundColor: '#6b7280',
  },
  captureInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#fff',
  },
  waldoInfo: {
    marginTop: 24,
    padding: 16,
    backgroundColor: '#1f2937',
    borderRadius: 12,
  },
  waldoInfoTitle: {
    fontSize: 14,
    color: '#9ca3af',
    marginBottom: 12,
  },
  waldoClasses: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  waldoClass: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  waldoClassDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  waldoClassText: {
    fontSize: 12,
    color: '#fff',
  },
});
