import smtplib
import os
from dotenv import load_dotenv

load_dotenv()

# Use environment variables instead of hardcoded credentials
sender = os.getenv("SMTP_USER", "your-email@gmail.com")
password = os.getenv("SMTP_PASSWORD", "")  # Set in .env file
receiver = os.getenv("TEST_EMAIL", "test@example.com")

if not password:
    print("❌ ERROR: SMTP_PASSWORD not set in .env file")
    print("Please create a .env file with your Gmail App Password")
    exit(1)

message = """Subject: Test Mail

Hello from Python!
"""

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, password)
    server.sendmail(sender, receiver, message)
    server.quit()
    print("✓ Email sent successfully!")
except Exception as e:
    print(f"❌ Error sending email: {e}")