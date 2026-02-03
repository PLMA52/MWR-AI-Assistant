"""
AuraDB Keep-Alive Script
Prevents AuraDB Free instance from pausing due to inactivity.
Runs a simple write/delete query to maintain active status.

Schedule: Run every 2 days via GitHub Actions
"""

import os
from neo4j import GraphDatabase
from datetime import datetime

# AuraDB Connection - Uses GitHub Secrets when run via Actions
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def keepalive():
    """Execute heartbeat query to keep AuraDB active."""
    
    if not NEO4J_PASSWORD:
        print("❌ ERROR: NEO4J_PASSWORD not set")
        return False
    
    driver = None
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            # Create and immediately delete a heartbeat node
            result = session.run("""
                CREATE (h:Heartbeat {timestamp: datetime(), source: 'keepalive'})
                WITH h
                DELETE h
                RETURN 'success' as status
            """)
            
            record = result.single()
            
            if record and record["status"] == "success":
                print(f"✅ AuraDB keepalive successful at {datetime.now().isoformat()}")
                return True
            else:
                print("❌ Keepalive query returned unexpected result")
                return False
                
    except Exception as e:
        print(f"❌ Keepalive failed: {e}")
        return False
        
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    success = keepalive()
    exit(0 if success else 1)