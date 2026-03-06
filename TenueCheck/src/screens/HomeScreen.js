import React, { useState, useEffect } from 'react';
import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, RefreshControl, StatusBar
} from 'react-native';
import { api } from '../services/api';

export default function HomeScreen({ navigation }) {
  const [stats, setStats] = useState({ total_detections: 0, total_alerts: 0, violations: [] });
  const [systemActive, setSystemActive] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [connected, setConnected] = useState(false);

  const fetchData = async () => {
    try {
      const data = await api.getViolations();
      setStats(data);
      setConnected(true);
    } catch (e) {
      setConnected(false);
      // Données de démo si backend pas connecté
      setStats({
        total_detections: 12,
        total_alerts: 3,
        violations: [
          { type: 'Casquette', confidence: 0.87, timestamp: '09:42:15' },
          { type: 'Short', confidence: 0.73, timestamp: '09:38:02' },
          { type: 'Tongs', confidence: 0.91, timestamp: '09:31:47' },
        ]
      });
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleToggle = async () => {
    try {
      await api.toggleDetection();
    } catch (e) {}
    setSystemActive(!systemActive);
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#667eea" />}
    >
      <StatusBar barStyle="light-content" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>TenueCheck</Text>
        <Text style={styles.headerSub}>ENSITECH · Article 17</Text>
        <View style={[styles.connectedBadge, { backgroundColor: connected ? 'rgba(56,239,125,0.2)' : 'rgba(255,165,0,0.2)' }]}>
          <Text style={[styles.connectedText, { color: connected ? '#38ef7d' : '#ffa500' }]}>
            {connected ? '● Connecté au serveur' : '● Mode démo (serveur déconnecté)'}
          </Text>
        </View>
      </View>

      {/* Statut système */}
      <View style={styles.statusCard}>
        <View style={styles.statusRow}>
          <View style={[styles.dot, { backgroundColor: systemActive ? '#38ef7d' : '#ff416c' }]} />
          <Text style={styles.statusText}>Système {systemActive ? 'actif' : 'en pause'}</Text>
        </View>
        <TouchableOpacity
          style={[styles.toggleBtn, { backgroundColor: systemActive ? '#ff416c' : '#38ef7d' }]}
          onPress={handleToggle}
        >
          <Text style={styles.toggleBtnText}>{systemActive ? 'Pause' : 'Activer'}</Text>
        </TouchableOpacity>
      </View>

      {/* Stats */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>{stats.total_detections}</Text>
          <Text style={styles.statLabel}>Détections</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#ff416c' }]}>{stats.total_alerts}</Text>
          <Text style={styles.statLabel}>Alertes envoyées</Text>
        </View>
      </View>

      {/* Dernières violations */}
      <Text style={styles.sectionTitle}>Dernières violations</Text>
      {stats.violations && stats.violations.length > 0 ? (
        stats.violations.slice(0, 5).map((v, i) => (
          <View key={i} style={styles.violationCard}>
            <View>
              <Text style={styles.violationType}>🚫 {v.type}</Text>
              <Text style={styles.violationTime}>{v.timestamp}</Text>
            </View>
            <Text style={styles.violationConf}>{Math.round((v.confidence || 0.8) * 100)}%</Text>
          </View>
        ))
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyText}>✅ Aucune violation détectée</Text>
        </View>
      )}

      {/* Bouton caméra */}
      <TouchableOpacity style={styles.cameraBtn} onPress={() => navigation.navigate('Caméra')}>
        <Text style={styles.cameraBtnText}>📷 Analyser une tenue</Text>
      </TouchableOpacity>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e' },
  header: { backgroundColor: '#16213e', padding: 30, paddingTop: 60, alignItems: 'center' },
  headerTitle: { fontSize: 28, fontWeight: '800', color: '#fff', letterSpacing: 1 },
  headerSub: { fontSize: 12, color: '#667eea', marginTop: 4, letterSpacing: 3, textTransform: 'uppercase' },
  connectedBadge: { marginTop: 12, paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
  connectedText: { fontSize: 12, fontWeight: '600' },
  statusCard: {
    margin: 20, backgroundColor: '#16213e', borderRadius: 16, padding: 20,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
  },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  dot: { width: 12, height: 12, borderRadius: 6 },
  statusText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  toggleBtn: { paddingHorizontal: 18, paddingVertical: 9, borderRadius: 20 },
  toggleBtnText: { color: '#fff', fontWeight: '700', fontSize: 13 },
  statsRow: { flexDirection: 'row', paddingHorizontal: 20, gap: 12, marginBottom: 10 },
  statCard: { flex: 1, backgroundColor: '#16213e', borderRadius: 16, padding: 20, alignItems: 'center' },
  statValue: { fontSize: 38, fontWeight: '800', color: '#667eea' },
  statLabel: { fontSize: 12, color: '#888', marginTop: 4 },
  sectionTitle: { color: '#eee', fontSize: 17, fontWeight: '700', marginHorizontal: 20, marginBottom: 10 },
  violationCard: {
    backgroundColor: '#16213e', marginHorizontal: 20, marginBottom: 10,
    borderRadius: 12, padding: 16, borderLeftWidth: 4, borderLeftColor: '#ff416c',
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  violationType: { color: '#ff416c', fontWeight: '700', fontSize: 15 },
  violationTime: { color: '#888', fontSize: 12, marginTop: 3 },
  violationConf: { color: '#aaa', fontSize: 14, fontWeight: '600' },
  emptyState: { backgroundColor: '#16213e', marginHorizontal: 20, borderRadius: 12, padding: 24, alignItems: 'center' },
  emptyText: { color: '#38ef7d', fontSize: 15 },
  cameraBtn: { margin: 20, backgroundColor: '#667eea', borderRadius: 16, padding: 18, alignItems: 'center' },
  cameraBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },
});