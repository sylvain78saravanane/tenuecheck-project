import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Text } from 'react-native';
import HomeScreen from './src/screens/HomeScreen';
import CameraScreen from './src/screens/CameraScreen';
import AlertsScreen from './src/screens/AlertsScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={{
          headerShown: false,
          tabBarStyle: { backgroundColor: '#16213e', borderTopColor: '#2a2a4a', height: 65 },
          tabBarActiveTintColor: '#667eea',
          tabBarInactiveTintColor: '#555',
          tabBarLabelStyle: { fontSize: 12, fontWeight: '600', paddingBottom: 8 },
        }}
      >
        <Tab.Screen
          name="Accueil"
          component={HomeScreen}
          options={{ tabBarIcon: ({ color }) => <Text style={{ fontSize: 22 }}>🏠</Text> }}
        />
        <Tab.Screen
          name="Caméra"
          component={CameraScreen}
          options={{ tabBarIcon: ({ color }) => <Text style={{ fontSize: 22 }}>📷</Text> }}
        />
        <Tab.Screen
          name="Alertes"
          component={AlertsScreen}
          options={{ tabBarIcon: ({ color }) => <Text style={{ fontSize: 22 }}>🚨</Text> }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}