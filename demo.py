import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from main_pipeline import ViolationPipeline
from fine_system import FineSystem


DEFAULT_EMAIL = "suman15sep2004@gmail.com"
# DEFAULT_EMAIL = "harshshaw600@gmail.com"


def main():
    print("\n" + "="*60)
    print("HELMET VIOLATION DETECTION SYSTEM - DEMO")
    print("="*60)
    
    # Get image path from argument or prompt user
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\n📸 Enter the image path:")
        image_path = input("Path: ").strip()
    
    # Validate image path
    if not image_path:
        print("❌ No image path provided.")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(2)
    
    print(f"\n✅ Image found: {image_path}")
    print("\n🔍 Processing image...")
    
    # Demo mode: force email flow to the default address
    config.TEST_MODE = True
    config.TEST_EMAIL = DEFAULT_EMAIL
    config.SEND_EMAIL = True
    
    # Run pipeline without auto-email (we'll handle it)
    pipeline = ViolationPipeline(suppress_emails=True)
    result = pipeline.process_image(image_path, show=False, save=True)
    
    if not result:
        print("❌ No result returned.")
        return
    
    # Check if violation detected
    violation_count = result['detection']['violation_count']
    safe_count = result['detection']['safe_count']
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if violation_count > 0:
        print(f"⚠️  VIOLATION DETECTED!")
        print(f"   Violations found: {violation_count}")
        print(f"   Safe riders: {safe_count}")
        
        # Extract plate info from violation records
        plate_text = "NOT_DETECTED"
        plate_confidence = 0
        
        if 'violation_records' in result and result['violation_records']:
            # Get the first violation record
            record = result['violation_records'][0]
            plate_text = record.get('plate_text', 'NOT_DETECTED')
            plate_confidence = record.get('ocr_conf', 0)
        
        if plate_text != "NOT_DETECTED":
            capped_confidence = min(plate_confidence, 1.0)  # Cap at 100%
            print(f"\n   ✅ Plate Detected: {plate_text}")
            if capped_confidence < plate_confidence:
                print(f"   Confidence: 100% (capped, raw: {plate_confidence:.0%})")
            else:
                print(f"   Confidence: {capped_confidence:.0%}")
        else:
            print(f"\n   ❌ Plate: NOT_DETECTED")
        
        # Send email with violation details
        print(f"\n📧 Sending violation email to {DEFAULT_EMAIL}...")
        
        try:
            fine_system = FineSystem()
            
            # Use the annotated image path from the result (current detection)
            annotated_path = result.get('annotated_path')
            
            # Find the most recent evidence/invoice file
            evidence_files = glob.glob(os.path.join(config.EVIDENCE_DIR, "*.jpg"))
            invoice_files = glob.glob(os.path.join(config.INVOICES_DIR, "*.pdf"))
            
            evidence_path = evidence_files[-1] if evidence_files else None
            invoice_path = invoice_files[-1] if invoice_files else None
            
            # Prepare email info
            owner_name = "Traffic Violator"
            violation_type = "No Helmet Violation"
            fine_amount = config.FINE_AMOUNT
            
            # Cap confidence at 100%
            capped_confidence = min(plate_confidence, 1.0)
            
            # Send email with detected plate, using annotated image from current detection
            email_sent = fine_system.send_violation_email(
                to_email=DEFAULT_EMAIL,
                owner_name=owner_name,
                plate_number=plate_text,
                violation_type=violation_type,
                fine_amount=fine_amount,
                pdf_path=invoice_path,
                evidence_image_path=annotated_path,  # Use current annotated image
                original_image_path=image_path,
                test_mode=True
            )
            
            if email_sent:
                print("✅ Email sent successfully!")
                print(f"   📄 Invoice: {os.path.basename(invoice_path) if invoice_path else 'N/A'}")
                print(f"   📸 Original: original_image.jpg")
                print(f"   📸 Annotated: violation_annotated.jpg (with bounding box)")
                print(f"   🚗 Plate: {plate_text}")
                if capped_confidence < plate_confidence:
                    print(f"   ⚠️  Confidence: 100% (capped, raw: {plate_confidence:.0%})")
                else:
                    print(f"   📊 Confidence: {capped_confidence:.0%}")
            else:
                print("⚠️  Email could not be sent (check SMTP config)")
                
        except Exception as e:
            print(f"⚠️  Email sending failed: {e}")
    else:
        print(f"✅ SAFE - No violations detected")
        print(f"   Safe riders: {safe_count}")
        print(f"\n📧 No email sent (traffic is safe)")
    
    print("\n" + "="*60)
    print("Evidence saved to: outputs/evidence/")
    print("Annotated image saved to: outputs/annotated/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
