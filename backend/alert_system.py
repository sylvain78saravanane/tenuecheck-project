"""
Système d'alertes par email pour les violations du code vestimentaire
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from config import EMAIL_CONFIG, MESSAGES_ALERTE


class AlertSystem:
    """
    Système de gestion des alertes et envoi d'emails
    """

    def __init__(self):
        self.smtp_server = EMAIL_CONFIG["smtp_server"]
        self.smtp_port = EMAIL_CONFIG["smtp_port"]
        self.sender_email = EMAIL_CONFIG["sender_email"]
        self.sender_password = EMAIL_CONFIG["sender_password"]
        self.recipient_email = EMAIL_CONFIG["recipient_email"]

        # Historique des alertes envoyées (pour éviter le spam)
        self.sent_alerts = {}
        self.alert_cooldown = 60  # Secondes entre deux alertes similaires

    def create_alert_email(self, violations, image_path=None):
        """
        Crée le contenu de l'email d'alerte
        """
        msg = MIMEMultipart('related')
        msg['Subject'] = f"[ALERTE ENSITECH] Violation du code vestimentaire - {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email

        # Corps HTML de l'email
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #d32f2f; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .violation {{ background-color: #ffebee; border-left: 4px solid #d32f2f; padding: 10px; margin: 10px 0; }}
                .timestamp {{ color: #666; font-size: 12px; }}
                .footer {{ background-color: #f5f5f5; padding: 10px; text-align: center; font-size: 12px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #d32f2f; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ALERTE - Code Vestimentaire ENSITECH</h1>
            </div>
            <div class="content">
                <p><strong>Date et heure:</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>

                <h2>Violations détectées:</h2>
                <table>
                    <tr>
                        <th>Type de violation</th>
                        <th>Confiance</th>
                        <th>Message</th>
                    </tr>
        """

        for violation in violations:
            violation_type = violation['type']
            confidence = violation['confidence'] * 100
            message = MESSAGES_ALERTE.get(violation_type, f"Vêtement non autorisé: {violation_type}")

            html_content += f"""
                    <tr>
                        <td><strong>{violation_type}</strong></td>
                        <td>{confidence:.0f}%</td>
                        <td>{message}</td>
                    </tr>
            """

        html_content += """
                </table>

                <div class="violation">
                    <p><strong>Action requise:</strong> Veuillez vérifier et prendre les mesures appropriées conformément au règlement intérieur (Article 17).</p>
                </div>
        """

        if image_path:
            html_content += """
                <h3>Image capturée:</h3>
                <img src="cid:alert_image" style="max-width: 100%; border: 1px solid #ddd;">
            """

        html_content += """
            </div>
            <div class="footer">
                <p>Ce message a été généré automatiquement par le système de surveillance ENSITECH.</p>
                <p>Pour toute question, contactez l'administration.</p>
            </div>
        </body>
        </html>
        """

        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)

        # Version texte simple
        text_content = f"""
        ALERTE - Code Vestimentaire ENSITECH

        Date et heure: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}

        Violations détectées:
        """
        for violation in violations:
            text_content += f"\n- {violation['type']} (Confiance: {violation['confidence']*100:.0f}%)"

        text_content += """

        Action requise: Veuillez vérifier et prendre les mesures appropriées conformément au règlement intérieur (Article 17).

        ---
        Ce message a été généré automatiquement par le système de surveillance ENSITECH.
        """

        msg_alternative.attach(MIMEText(text_content, 'plain'))
        msg_alternative.attach(MIMEText(html_content, 'html'))

        # Attacher l'image si disponible
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-ID', '<alert_image>')
                img.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
                msg.attach(img)

        return msg

    def should_send_alert(self, violation_type):
        """
        Vérifie si une alerte doit être envoyée (évite le spam)
        """
        current_time = datetime.now()

        if violation_type in self.sent_alerts:
            last_sent = self.sent_alerts[violation_type]
            time_diff = (current_time - last_sent).total_seconds()

            if time_diff < self.alert_cooldown:
                return False

        return True

    def send_alert(self, violations, image_path=None):
        """
        Envoie l'alerte par email
        """
        # Vérifier si on doit envoyer l'alerte
        violation_types = [v['type'] for v in violations]
        if not any(self.should_send_alert(vt) for vt in violation_types):
            print("Alerte ignorée (cooldown actif)")
            return False

        try:
            msg = self.create_alert_email(violations, image_path)

            # Connexion au serveur SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            # Mettre à jour l'historique
            current_time = datetime.now()
            for violation_type in violation_types:
                self.sent_alerts[violation_type] = current_time

            print(f"Alerte envoyée avec succès à {self.recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError:
            print("Erreur d'authentification SMTP. Vérifiez vos identifiants.")
            return False
        except smtplib.SMTPException as e:
            print(f"Erreur SMTP: {e}")
            return False
        except Exception as e:
            print(f"Erreur lors de l'envoi de l'alerte: {e}")
            return False

    def log_alert(self, violations, image_path=None):
        """
        Enregistre l'alerte dans un fichier log (alternative à l'email)
        """
        log_file = "alerts/alerts_log.txt"
        os.makedirs("alerts", exist_ok=True)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"ALERTE - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")

            for violation in violations:
                f.write(f"- Type: {violation['type']}\n")
                f.write(f"  Confiance: {violation['confidence']*100:.0f}%\n")
                message = MESSAGES_ALERTE.get(violation['type'], "Violation du code vestimentaire")
                f.write(f"  Message: {message}\n")

            if image_path:
                f.write(f"Image sauvegardée: {image_path}\n")

            f.write(f"{'='*60}\n")

        print(f"Alerte enregistrée dans {log_file}")
        return True


# Fonction utilitaire pour tester le système d'alertes
def test_alert_system():
    """
    Teste le système d'alertes
    """
    alert_system = AlertSystem()

    test_violations = [
        {"type": "Casquette", "confidence": 0.85},
        {"type": "Short", "confidence": 0.72}
    ]

    # Test de logging (fonctionne toujours)
    alert_system.log_alert(test_violations)

    # Test d'email (nécessite configuration)
    # alert_system.send_alert(test_violations)


if __name__ == "__main__":
    test_alert_system()
