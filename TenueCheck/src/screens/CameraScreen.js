import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  Image, ActivityIndicator, ScrollView
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { api } from '../services/api';

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const cameraRef = useRef(null);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permText}>La caméra est nécessaire pour analyser les tenues</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Autoriser l'accès</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const takePicture = async () => {
    if (cameraRef.current) {
      const pic = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      setPhoto(pic.uri);
      setResult(null);
    }
  };

  const analyzePhoto = async () => {
    setAnalyzing(true);
    try {
      const data = await api.analyzeImage(photo);
      setResult(data);
    } catch (e) {
      // Mode démo
      setResult({
        violations: [{ type: 'Casquette', confidence: 0.87 }],
        status: 'Non conforme'
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const reset = () => { setPhoto(null); setResult(null); };

  // Écran résultat après photo
  if (photo) {
    const hasViolation = result?.violations?.length > 0;
    return (
      <ScrollView style={styles.container}>
        <Image source={{ uri: photo }} style={styles.preview} />

        {!result && !analyzing && (
          <TouchableOpacity style={styles.analyzeBtn} onPress={analyzePhoto}>
            <Text style={styles.analyzeBtnText}>🔍 Analyser la tenue</Text>
          </TouchableOpacity>
        )}

        {analyzing && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#667eea" />
            <Text style={styles.loadingText}>Analyse en cours...</Text>
          </View>
        )}

        {result && (
          <View style={[styles.resultCard, { borderColor: hasViolation ? '#ff416c' : '#38ef7d' }]}>
            <Text style={[styles.resultTitle, { color: hasViolation ? '#ff416c' : '#38ef7d' }]}>
              {hasViolation ? '🚫 Tenue non conforme' : '✅ Tenue conforme'}
            </Text>
            {result.violations?.map((v, i) => (
              <View key={i} style={styles.resultViolationRow}>
                <Text style={styles.resultViolationType}>• {v.type}</Text>
                <Text style={styles.resultViolationConf}>{Math.round(v.confidence * 100)}%</Text>
              </View>
            ))}
          </View>
        )}

        <TouchableOpacity style={styles.retakeBtn} onPress={reset}>
          <Text style={styles.retakeBtnText}>↩ Prendre une autre photo</Text>
        </TouchableOpacity>

        <View style={{ height: 40 }} />
      </ScrollView>
    );
  }

  // Écran caméra
  return (
    <View style={styles.container}>
    <CameraView style={styles.camera} ref={cameraRef}>
        <View style={styles.aimFrame} />
        <View style={styles.hint}>
          <Text style={styles.hintText}>Cadrez la personne entièrement</Text>
        </View>
        <TouchableOpacity style={styles.captureBtn} onPress={takePicture}>
          <View style={styles.captureInner} />
        </TouchableOpacity>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e' },
  camera: { flex: 1 },
  aimFrame: {
    position: 'absolute', top: '15%', left: '15%', right: '15%', bottom: '20%',
    borderWidth: 2, borderColor: 'rgba(102,126,234,0.6)', borderRadius: 12,
  },
  hint: {
    position: 'absolute', top: 60, alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.65)', paddingHorizontal: 20, paddingVertical: 8, borderRadius: 20,
  },
  hintText: { color: '#fff', fontSize: 13 },
  captureBtn: {
    position: 'absolute', bottom: 50, alignSelf: 'center',
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.25)',
    justifyContent: 'center', alignItems: 'center',
    borderWidth: 3, borderColor: 'rgba(255,255,255,0.6)',
  },
  captureInner: { width: 58, height: 58, borderRadius: 29, backgroundColor: '#fff' },
  preview: { width: '100%', height: 380 },
  analyzeBtn: {
    margin: 20, backgroundColor: '#667eea',
    borderRadius: 14, padding: 18, alignItems: 'center',
  },
  analyzeBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  loadingBox: { alignItems: 'center', padding: 30 },
  loadingText: { color: '#aaa', marginTop: 12, fontSize: 15 },
  resultCard: {
    margin: 20, backgroundColor: '#16213e',
    borderRadius: 14, padding: 20, borderWidth: 2,
  },
  resultTitle: { fontSize: 20, fontWeight: '800', marginBottom: 12 },
  resultViolationRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
  resultViolationType: { color: '#ff416c', fontSize: 15 },
  resultViolationConf: { color: '#aaa', fontSize: 14 },
  retakeBtn: {
    marginHorizontal: 20, backgroundColor: '#2a2a4a',
    borderRadius: 14, padding: 16, alignItems: 'center',
  },
  retakeBtnText: { color: '#aaa', fontWeight: '600', fontSize: 15 },
  permText: { color: '#eee', fontSize: 16, textAlign: 'center', margin: 30 },
  permBtn: { backgroundColor: '#667eea', margin: 20, padding: 16, borderRadius: 12, alignItems: 'center' },
  permBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
});