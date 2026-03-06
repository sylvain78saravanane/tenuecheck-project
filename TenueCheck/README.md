Étape 1 — Va dans le bon dossier et crée le projet Expo
cd tenuecheck-project
npx create-expo-app TenueCheck --template blank
cd TenueCheck

Étape 2 — Remplace tout le package.json par ça :
{
  "name": "tenuecheck",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "start": "expo start",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web"
  },
  "dependencies": {
    "@react-navigation/bottom-tabs": "6.6.1",
    "@react-navigation/native": "6.1.18",
    "axios": "1.7.2",
    "expo": "54.0.33",
    "expo-camera": "17.0.10",
    "expo-status-bar": "3.0.9",
    "react": "19.1.0",
    "react-native": "0.81.5",
    "react-native-safe-area-context": "5.6.0",
    "react-native-screens": "4.16.0",
    "react-dom": "19.1.0",
    "react-native-web": "0.21.0"
  },
  "private": true
}

Étape 3 — Installe les dépendances
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install --legacy-peer-deps

Étape 4 — Crée les dossiers
mkdir src\screens
mkdir src\services

Étape 5 — Crée les fichiers dans VS Code :

src/services/api.js
src/screens/HomeScreen.js
src/screens/CameraScreen.js
src/screens/AlertsScreen.js
Remplace App.js

Étape 6 — Lance
Installe Expo Go sur ton app store
npx expo start --clear