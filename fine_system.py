"""
FINE SYSTEM MODULE
Generates PDF invoices and sends emails
"""

import os
from datetime import datetime
import config

# Try to import PDF library
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("[FineSystem] reportlab not installed. Run: pip install reportlab")

# Email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from email.utils import make_msgid


class FineSystem:
    def __init__(self):
        self.fine_amount = config.FINE_AMOUNT
        self.smtp_config = config.SMTP_CONFIG
        self.backup_smtp = config.BACKUP_SMTP_CONFIG
        self.send_email_flag = config.SEND_EMAIL
        os.makedirs(config.INVOICES_DIR, exist_ok=True)

    def _smtp_credentials_ready(self, smtp_config):
        """Return True only when real SMTP credentials are configured."""
        username = (smtp_config or {}).get("username", "")
        password = (smtp_config or {}).get("password", "")

        if not username or not password:
            return False

        placeholder_values = {
            "your-email@gmail.com",
            "your-app-password-here",
            "Traffic Violation System <your-email@gmail.com>",
        }
        if username in placeholder_values or password in placeholder_values:
            return False

        return True
    
    def generate_invoice_pdf(self, owner_name, plate_number, violation_type, 
                              violation_date, evidence_path, fine_amount, 
                              violation_id):
        """Generate PDF invoice for fine"""
        if not REPORTLAB_OK:
            print("[FineSystem] Cannot generate PDF: reportlab not installed")
            return None
        
        invoice_number = f"INV-{violation_id}-{datetime.now().strftime('%Y%m%d')}"
        pdf_path = os.path.join(config.INVOICES_DIR, f"{invoice_number}.pdf")
        
        try:
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                                   topMargin=20*mm, bottomMargin=20*mm,
                                   leftMargin=20*mm, rightMargin=20*mm)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#cc0000'),
                spaceAfter=30,
                alignment=1  # Center
            )
            
            header_style = ParagraphStyle(
                'Header',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#333333'),
                spaceAfter=10
            )
            
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6
            )
            
            # Build content
            story = []
            
            # Header
            story.append(Paragraph("TRAFFIC VIOLATION NOTICE", title_style))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"Invoice Number: {invoice_number}", normal_style))
            story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
            story.append(Spacer(1, 20))
            
            # Vehicle Owner Details
            story.append(Paragraph("Vehicle Owner Details", header_style))
            owner_data = [
                ["Owner Name:", owner_name],
                ["Vehicle Number:", plate_number],
                ["Violation Type:", violation_type],
                ["Violation Date:", violation_date],
            ]
            owner_table = Table(owner_data, colWidths=[80*mm, 100*mm])
            owner_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(owner_table)
            story.append(Spacer(1, 20))
            
            # Fine Details
            story.append(Paragraph("Fine Details", header_style))
            fine_data = [
                ["Description", "Amount (₹)"],
                [f"No Helmet Violation - {plate_number}", f"₹{fine_amount:.2f}"],
                ["Processing Fee", "₹0.00"],
                ["Total Amount Due", f"₹{fine_amount:.2f}"],
            ]
            fine_table = Table(fine_data, colWidths=[120*mm, 60*mm])
            fine_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ffeeee')),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(fine_table)
            story.append(Spacer(1, 20))
            
            # Payment Instructions
            story.append(Paragraph("Payment Instructions", header_style))
            story.append(Paragraph("Please pay the fine within 15 days to avoid additional penalties.", normal_style))
            story.append(Paragraph("Payment can be made online or at any traffic police station.", normal_style))
            story.append(Spacer(1, 10))
            story.append(Paragraph("Online Payment:", normal_style))
            story.append(Paragraph("• UPI ID: trafficfine@gov.in", normal_style))
            story.append(Paragraph("• Net Banking: Account No: 123456789, IFSC: GOVT001", normal_style))
            story.append(Spacer(1, 20))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_text = "This is an automatically generated notice. If you believe this is an error, please contact the traffic department."
            story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)))
            
            # Build PDF
            doc.build(story)
            print(f"[FineSystem] PDF generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"[FineSystem] PDF generation error: {e}")
            return None

    def _build_email_message(self, to_email, owner_name, plate_number,
                             violation_type, fine_amount, pdf_path,
                             evidence_image_path=None, original_image_path=None, test_mode=False):
        """Build a rich HTML email with both original and annotated evidence images."""
        subject = f"Traffic Violation Notice - {plate_number}"
        if test_mode:
            subject = f"[TEST] {subject}"

        msg = MIMEMultipart("related")
        msg['From'] = self.smtp_config['from_email']
        msg['To'] = to_email
        msg['Subject'] = subject

        alt = MIMEMultipart("alternative")
        msg.attach(alt)

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #f6f7fb; padding: 18px;">
            <div style="max-width: 720px; margin: auto; background: white; border: 1px solid #e5e7eb; border-radius: 14px; overflow: hidden;">
                <div style="background: #b91c1c; color: white; padding: 18px 22px;">
                    <div style="font-size: 22px; font-weight: bold;">⚠️ Traffic Violation Notice</div>
                </div>
                <div style="padding: 22px; color: #111827;">
                    <p style="font-size: 14px; margin: 0 0 10px 0;"><strong>Vehicle Number:</strong> <span style="font-size: 18px; color: #b91c1c;">{plate_number}</span></p>
                    <p style="font-size: 14px; margin: 0 0 10px 0;"><strong>Violation:</strong> {violation_type}</p>
                    <p style="font-size: 14px; margin: 0 0 10px 0;"><strong>Fine Amount:</strong> <span style="font-size: 16px; font-weight: bold; color: #b91c1c;">₹{fine_amount:.2f}</span></p>
                    
                    <div style="margin: 16px 0; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fcfcfd;">
                        <p style="font-size: 13px; margin: 0; color: #6b7280;">Original and annotated evidence images with invoice PDF are attached. Please pay the fine within 15 days.</p>
                    </div>
                    
                    <p style="font-size: 10px; color: #9ca3af; margin-top: 16px; border-top: 1px solid #e5e7eb; padding-top: 12px;"><em>This is a system-generated notice. If this is not you, please reply to this email and we apologize for the inconvenience.</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        alt.attach(MIMEText(html_body, 'html'))

        text_body = (
            f"⚠️ TRAFFIC VIOLATION NOTICE\n\n"
            f"Vehicle Number: {plate_number}\n"
            f"Violation: {violation_type}\n"
            f"Fine Amount: ₹{fine_amount:.2f}\n\n"
            f"Original and annotated evidence images with invoice PDF are attached.\n"
            f"Pay within 15 days to avoid penalties.\n\n"
            f"This is a system-generated notice. If this is not you, please reply to this email and we apologize for the inconvenience."
        )
        alt.attach(MIMEText(text_body, 'plain'))

        # Attach PDF
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_attachment = MIMEBase('application', 'octet-stream')
                pdf_attachment.set_payload(f.read())
                encoders.encode_base64(pdf_attachment)
                pdf_attachment.add_header('Content-Disposition',
                                          f'attachment; filename="{os.path.basename(pdf_path)}"')
                msg.attach(pdf_attachment)

        # Attach original image
        if original_image_path and os.path.exists(original_image_path):
            with open(original_image_path, 'rb') as f:
                original_bytes = f.read()
            img = MIMEImage(original_bytes)
            img.add_header('Content-Disposition', 'attachment', filename='original_image.jpg')
            msg.attach(img)

        # Attach annotated/evidence image with violation bounding box
        if evidence_image_path and os.path.exists(evidence_image_path):
            with open(evidence_image_path, 'rb') as f:
                evidence_bytes = f.read()
            img = MIMEImage(evidence_bytes)
            img.add_header('Content-Disposition', 'attachment', filename='violation_annotated.jpg')
            msg.attach(img)

        # Ensure Message-ID exists so we can log it after send
        if 'Message-ID' not in msg:
            msg['Message-ID'] = make_msgid()

        return msg, subject
    
    def send_violation_email(self, to_email, owner_name, plate_number, 
                             violation_type, fine_amount, pdf_path, 
                             evidence_image_path=None, original_image_path=None, test_mode=False):
        """Send violation notice email with PDF and images (original + annotated)"""
        if not self.send_email_flag and not test_mode:
            print(f"[FineSystem] Email disabled. Would send to: {to_email}")
            return False

        if not self._smtp_credentials_ready(self.smtp_config):
            print("[FineSystem] SMTP credentials are missing or still placeholder values in .env")
            print("[FineSystem] Set SMTP_USER and SMTP_PASSWORD to a real Gmail app password before sending")
            return False
        
        # Build message once
        msg, subject = self._build_email_message(
            to_email=to_email,
            owner_name=owner_name,
            plate_number=plate_number,
            violation_type=violation_type,
            fine_amount=fine_amount,
            pdf_path=pdf_path,
            evidence_image_path=evidence_image_path,
            original_image_path=original_image_path,
            test_mode=test_mode,
        )

        # Attempt primary SMTP first
        try:
            server = smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port'], timeout=30)
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            # Use authenticated account as envelope and From header to avoid spoofing/DMARC issues
            try:
                if 'From' in msg:
                    del msg['From']
            except Exception:
                pass
            msg['From'] = self.smtp_config.get('username')
            send_result = server.send_message(msg, from_addr=self.smtp_config.get('username'), to_addrs=[to_email])
            print(f"[FineSystem] Primary send_result: {send_result}")
            try:
                print(f"[FineSystem] Primary server features: {server.esmtp_features}")
            except Exception:
                pass
            server.quit()
            print(f"[FineSystem] Email sent to {to_email} via primary SMTP | Message-ID: {msg.get('Message-ID')}")
            return True
        except Exception as e_primary:
            print(f"[FineSystem] Primary SMTP error: {e_primary}")

            # If backup configured, try backup SMTP
            try:
                if self.backup_smtp and self.backup_smtp.get('username'):
                    b = self.backup_smtp
                    server = smtplib.SMTP(b['server'], int(b['port']), timeout=30)
                    server.starttls()
                    server.login(b['username'], b['password'])
                    # Use backup authenticated account as envelope and From header
                    try:
                        if 'From' in msg:
                            del msg['From']
                    except Exception:
                        pass
                    msg['From'] = b.get('username')
                    send_result = server.send_message(msg, from_addr=b.get('username'), to_addrs=[to_email])
                    print(f"[FineSystem] Backup send_result: {send_result}")
                    try:
                        print(f"[FineSystem] Backup server features: {server.esmtp_features}")
                    except Exception:
                        pass
                    server.quit()
                    print(f"[FineSystem] Email sent to {to_email} via backup SMTP | Message-ID: {msg.get('Message-ID')}")
                    return True
                else:
                    print("[FineSystem] No backup SMTP configured; skipping backup send")
            except Exception as e_backup:
                print(f"[FineSystem] Backup SMTP error: {e_backup}")

            # Fallback behavior: save .eml when test or demo
            if test_mode or to_email == config.TEST_EMAIL:
                demo_outbox = os.path.join(config.OUTPUT_DIR, "demo_outbox")
                os.makedirs(demo_outbox, exist_ok=True)
                filename = f"{plate_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eml"
                fallback_path = os.path.join(demo_outbox, filename)
                with open(fallback_path, "w", encoding="utf-8") as f:
                    f.write(msg.as_string())
                print(f"[FineSystem] Demo email saved locally (NOT sent): {fallback_path}")

            return False

    def test_send_via_both(self, to_email):
        """Test method: attempt primary then backup and return detailed results."""
        result = {"primary": False, "backup": False, "primary_error": None, "backup_error": None}

        if not self._smtp_credentials_ready(self.smtp_config):
            result["primary_error"] = "SMTP credentials are missing or still placeholder values in .env"

        msg, _ = self._build_email_message(
            to_email=to_email,
            owner_name="Test",
            plate_number="TEST1234",
            violation_type="SMTP Connectivity Test",
            fine_amount=0.0,
            pdf_path=None,
            evidence_image_path=None,
            original_image_path=None,
            test_mode=True,
        )
        # Primary
        if self._smtp_credentials_ready(self.smtp_config):
            try:
                s = smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port'], timeout=30)
                s.starttls()
                s.login(self.smtp_config['username'], self.smtp_config['password'])
                # Ensure From/envelope match authenticated account
                try:
                    if 'From' in msg:
                        del msg['From']
                except Exception:
                    pass
                msg['From'] = self.smtp_config.get('username')
                send_result = s.send_message(msg, from_addr=self.smtp_config.get('username'), to_addrs=[to_email])
                print(f"[FineSystem] Primary test send_result: {send_result}")
                s.quit()
                result['primary'] = True
            except Exception as e:
                result['primary_error'] = repr(e)

        # Backup
        try:
            b = self.backup_smtp
            if b and self._smtp_credentials_ready(b):
                s = smtplib.SMTP(b['server'], int(b['port']), timeout=30)
                s.starttls()
                s.login(b['username'], b['password'])
                try:
                    if 'From' in msg:
                        del msg['From']
                except Exception:
                    pass
                msg['From'] = b.get('username')
                send_result = s.send_message(msg, from_addr=b.get('username'), to_addrs=[to_email])
                print(f"[FineSystem] Backup test send_result: {send_result}")
                s.quit()
                result['backup'] = True
            else:
                result['backup_error'] = 'No backup configured'
        except Exception as e:
            result['backup_error'] = repr(e)

        return result


def issue_fine(violation_id, vehicle_info, violation_type, violation_date, 
               evidence_path, fine_amount, send_email=True, recipient_email=None,
               original_image_path=None, test_mode=False):
    """
    Complete fine issuance: PDF + Email with original and annotated images
    """
    system = FineSystem()
    
    # Generate PDF
    pdf_path = system.generate_invoice_pdf(
        owner_name=vehicle_info['owner_name'],
        plate_number=vehicle_info['plate_number'],
        violation_type=violation_type,
        violation_date=violation_date,
        evidence_path=evidence_path,
        fine_amount=fine_amount,
        violation_id=violation_id
    )
    
    # Send email
    email_sent = False
    if send_email and pdf_path:
        email_to = recipient_email or vehicle_info['owner_email']
        email_sent = system.send_violation_email(
            to_email=email_to,
            owner_name=vehicle_info['owner_name'],
            plate_number=vehicle_info['plate_number'],
            violation_type=violation_type,
            fine_amount=fine_amount,
            pdf_path=pdf_path,
            evidence_image_path=evidence_path,
            original_image_path=original_image_path,
            test_mode=test_mode
        )
    
    return {
        "pdf_path": pdf_path,
        "email_sent": email_sent
    }


if __name__ == "__main__":
    # Test
    system = FineSystem()
    print(f"Fine amount: ₹{system.fine_amount}")
    print(f"Email sending: {'ENABLED' if system.send_email_flag else 'DISABLED (set SEND_EMAIL=True in config)'}")