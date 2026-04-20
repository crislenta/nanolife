#!/usr/bin/env python3
"""
Check email outbox and send queued emails via Nebula email system.
This script should be called periodically to process outbox.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Import Nebula's email capability (available in sandbox environment)
# This is a placeholder - the actual implementation will use the email tool
def send_email(to: str, subject: str, body: str):
    """Send email via Nebula email tool."""
    # In actual use, this would call the Nebula email send function
    # For now, print to stdout so orchestrator can see it
    print(f"EMAIL_SEND_REQUEST:")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"---BODY---")
    print(body)
    print(f"---END BODY---")
    return True

def main():
    project_root = Path(__file__).parent
    outbox_file = project_root / "autonomous_logs" / "email_outbox.json"
    
    if not outbox_file.exists():
        print("No emails in outbox")
        return 0
    
    try:
        with open(outbox_file) as f:
            email_data = json.load(f)
        
        # Send the email
        success = send_email(
            to=email_data["to"],
            subject=email_data["subject"],
            body=email_data["body"]
        )
        
        if success:
            # Archive the sent email
            archive_dir = project_root / "autonomous_logs" / "sent_emails"
            archive_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = archive_dir / f"email_{timestamp}.json"
            
            with open(archive_file, 'w') as f:
                json.dump(email_data, f, indent=2)
            
            # Remove from outbox
            outbox_file.unlink()
            
            print(f"Email sent and archived to {archive_file}")
            return 0
        else:
            print("Failed to send email")
            return 1
            
    except Exception as e:
        print(f"Error processing email: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
