const BASE_URL = 'http://192.168.56.1:5000'; 
// ⚠️ Remplace XX par l'IP du PC de ton collègue
// Pour trouver son IP : il tape "ipconfig" dans son terminal Windows
// et cherche "Adresse IPv4" → genre 192.168.1.42

// 192.168.56.1

export const api = {

  getViolations: async () => {
    const res = await fetch(`${BASE_URL}/api/violations`);
    return res.json();
  },

  toggleDetection: async () => {
    const res = await fetch(`${BASE_URL}/api/toggle`, { method: 'POST' });
    return res.json();
  },

  sendTestAlert: async () => {
    const res = await fetch(`${BASE_URL}/api/test_alert`, { method: 'POST' });
    return res.json();
  },

  getStats: async () => {
    const res = await fetch(`${BASE_URL}/api/stats`);
    return res.json();
  },

  analyzeImage: async (imageUri) => {
    const formData = new FormData();
    formData.append('image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'capture.jpg',
    });
    const res = await fetch(`${BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.json();
  },
};