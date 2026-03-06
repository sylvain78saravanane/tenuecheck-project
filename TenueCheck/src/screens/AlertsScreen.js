import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, RefreshControl, TouchableOpacity, Alert } from 'react-native';
import { api } from '../services/api';

const PROHIBITED = [
  'Short', 'Bermuda', 'Mini-jupe', 'Crop top', 'Casquette',
  'Chapeau', 'Bonnet', 'Bandana', 'Tongs', 'Jean troué', 'Tenue de sport'
];

export default function AlertsScreen() {
  const [violations, setViolations] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    try {
      const data = await api.getViolations();
      setViolations(data.violations || []);
    } catch {
      setViolations([
        { type: 'Casquette', confidence: 0.87, timestamp: '09:42:15', high_confidence: true },
        { type: 'Short', confidence: 0.73, timestamp: '09:38:02', high_confidence: true },
        { type: 'Tongs', confidence: 0.91, timestamp: '09:31:47', high_confidence: true },
      ]);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const onRefresh = async () => { setRefreshing(true); await fetchData(); setRefreshing(false); };

  const handleTestAlert = async () => {
    try {
      await api.sendTestAlert();
      Alert.alert('✅ Alerte envoyée', 'L\'alerte de test a bien été transmise au responsable.');
    } catch {
      Alert.alert('Mode démo', 'Alerte simulée (backend non connecté).');
    }
  };

  const renderItem = ({ item }) => (
    <View style={[styles.card, { borderLeftColor: item.high_confidence ? '#ff416c' : '#ffa500' }]}>
      <View>
        <Text style={styles.cardType}>🚫 {item.type}</Text>
        <Text style={styles.cardTime}>{item.timestamp}</Text>
      </View>
      <View style={styles.cardRight}>
        <Text style={styles.cardConf}>{Math.round((item.confidence || 0.8) * 100)}%</Text>
        <Text style={[styles.cardBadge, { color: item.high_confidence ? '#ff416c' : '#ffa500' }]}>
          {item.high_confidence ? 'CERTAIN' : 'SUSPECT'}
        </Text>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Alertes</Text>
        <Text style={styles.headerSub}>{violations.length} violation(s) enregistrée(s)</Text>
      </View>

      {/* Éléments interdits */}
      <Text style={styles.sectionTitle}>Tenues surveillées</Text>
      <View style={styles.tagsContainer}>
        {PROHIBITED.map((item, i) => (
          <View key={i} style={styles.tag}>
            <Text style={styles.tagText}>{item}</Text>
          </View>
        ))}
      </View>

      {/* Bouton test alerte */}
      <TouchableOpacity style={styles.testBtn} onPress={handleTestAlert}>
        <Text style={styles.testBtnText}>📧 Envoyer une alerte de test</Text>
      </TouchableOpacity>

      {/* Liste des violations */}
      <Text style={styles.sectionTitle}>Historique</Text>
      <FlatList
        data={violations}
        keyExtractor={(_, i) => String(i)}
        renderItem={renderItem}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#667eea" />}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>✅ Aucune violation détectée</Text>
          </View>
        }
        contentContainerStyle={{ paddingBottom: 40 }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e' },
  header: { backgroundColor: '#16213e', padding: 30, paddingTop: 60, alignItems: 'center' },
  headerTitle: { fontSize: 26, fontWeight: '800', color: '#fff' },
  headerSub: { fontSize: 13, color: '#888', marginTop: 4 },
  sectionTitle: { color: '#eee', fontSize: 16, fontWeight: '700', marginHorizontal: 20, marginTop: 16, marginBottom: 8 },
  tagsContainer: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: 16, gap: 8 },
  tag: { backgroundColor: 'rgba(255,65,108,0.15)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16 },
  tagText: { color: '#ff6b5b', fontSize: 12, fontWeight: '500' },
  testBtn: {
    margin: 20, marginBottom: 0,
    backgroundColor: 'rgba(102,126,234,0.15)',
    borderWidth: 1, borderColor: '#667eea',
    borderRadius: 12, padding: 14, alignItems: 'center',
  },
  testBtnText: { color: '#667eea', fontWeight: '600', fontSize: 14 },
  card: {
    backgroundColor: '#16213e', marginHorizontal: 20, marginBottom: 10,
    borderRadius: 12, padding: 16, borderLeftWidth: 4,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  cardType: { color: '#ff416c', fontWeight: '700', fontSize: 15 },
  cardTime: { color: '#666', fontSize: 12, marginTop: 3 },
  cardRight: { alignItems: 'flex-end' },
  cardConf: { color: '#aaa', fontSize: 16, fontWeight: '700' },
  cardBadge: { fontSize: 11, fontWeight: '600', marginTop: 2 },
  empty: { padding: 40, alignItems: 'center' },
  emptyText: { color: '#38ef7d', fontSize: 15 },
});