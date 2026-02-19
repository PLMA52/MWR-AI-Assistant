from neo4j import GraphDatabase

d = GraphDatabase.driver('neo4j+s://551c1b37.databases.neo4j.io', auth=('neo4j','RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g'))
s = d.session()

r = s.run('MATCH (z:ZipCode) RETURN keys(z) AS k LIMIT 1')
rec = r.single()
all_keys = sorted(rec['k'])

print("ALL properties on ZipCode nodes:")
print("=" * 50)
for k in all_keys:
    print(f"  {k}")

print("\n\nAge/pct related properties:")
print("=" * 50)
for k in all_keys:
    if 'pct' in k.lower() or 'age' in k.lower() or 'under' in k.lower() or 'over' in k.lower():
        print(f"  {k}")

d.close()
